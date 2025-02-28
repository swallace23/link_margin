import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline
from scipy.interpolate import RBFInterpolator as rbf
from scipy.interpolate import RectBivariateSpline as rbs
import scipy.interpolate as ip
from scipy.ndimage import uniform_filter1d
import pandas as pd
from PyNEC import *
import ppigrf as pp
from datetime import date
############################################ MATH UTILITIES ############################################
# einsum - einstein summation convention - here, row-wise dot product
def clip_norm_dots(vec1, vec2):
     return np.clip(np.einsum('ij,ij->i',vec1, vec2),0,0.9999)
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

############################################ POLARIZATION LOSS FUNCTIONS ############################################
# Calculates polarization loss factor per Alex Mule's formula
def get_polarization_loss(receiver, source, separation):

    receiver_hat = unit_vector(receiver)
    source_hat = unit_vector(source)
    separation_hat = unit_vector(separation)
    dot_rs = clip_norm_dots(receiver_hat,source_hat)
    dot_rsep = clip_norm_dots(receiver_hat,separation_hat)
    dot_ssep = clip_norm_dots(source_hat,separation_hat)
    denominator = np.sqrt(1-dot_rsep**2)*np.sqrt(1-dot_ssep**2)
    denominator = np.maximum(denominator,1e-6)
    ploss = (dot_rs - (dot_rsep*dot_ssep))/denominator
    return ploss

# convert vector from spherical to cartesian coordinates
def s_c_vec_conversion(spherical_vec):
    x = spherical_vec[0] * np.sin(spherical_vec[1]) * np.cos(spherical_vec[2])
    y = spherical_vec[0] * np.sin(spherical_vec[1]) * np.sin(spherical_vec[2])
    z = spherical_vec[0] * np.cos(spherical_vec[1])
    return np.array([x, y, z])
# cartesian to spherical
def c_s_vec_conversion(cartesian_vec):
    r = np.sqrt(cartesian_vec[0]**2+cartesian_vec[1]**2+cartesian_vec[2]**2)
    theta = np.arccos(cartesian_vec[2]/r)
    phi = np.arctan(cartesian_vec[1]/cartesian_vec[0])
    return np.array([r,theta,phi])

############################################ INTERPOLATION FUNCTIONS ############################################
# Gets time slice with even spacing
def interp_time(times, sample_rate, start_time=0, end_time=None):
    if end_time is None:
        end_time = times[-1]
    new_times = np.arange(times[0], times[-1],1/sample_rate)
    interval_indices = np.where((new_times >= start_time) & (new_times <= end_time))[0]
    return new_times[interval_indices]

# Interpolates Nyquist rocket position to given sample rate and time slice. 
# Interpolation over cartesian coordinates with respect to arc length to improve data fidelity over non-uniform time grid.
def interp_position(times, sample_rate, position_vector, start_time=0, end_time= None, coord_system = "spherical",new_times=None):
    #times with parameterized sample rate
    if end_time is None:
        end_time = times[-1]
    if new_times is None:
        new_times = np.arange(times[0], times[-1],1/sample_rate)
    position_vector[:, 1] = np.radians(position_vector[:, 1])  
    position_vector[:, 2] = np.radians(position_vector[:, 2])
    # Find the indices corresponding to the time interval
    unique_indices = np.unique(times, return_index=True)[1]
    times = times[unique_indices]
    position_vector = position_vector[unique_indices]
    

    interval_indices = np.where((new_times >= start_time) & (new_times <= end_time))[0]
    time_slice = new_times[interval_indices]

    positions_cartesian = np.apply_along_axis(s_c_vec_conversion, 1, np.copy(position_vector))
    arc_length = np.cumsum(np.linalg.norm(np.diff(positions_cartesian, axis=0), axis=1))
    arc_length = np.insert(arc_length, 0, 0)

    arc_length, unique_arc_indices = np.unique(arc_length, return_index=True)
    positions_cartesian = positions_cartesian[unique_arc_indices]

    num_samples = len(times)

    arc_length_uniform = np.linspace(arc_length[0], arc_length[-1], num_samples)
    pos_interp_funcs = [ip.PchipInterpolator(arc_length, positions_cartesian[:,i]) for i in range(3)]
    positions_cartesian_resampled = np.column_stack([func(arc_length_uniform) for func in pos_interp_funcs])

    pos_interp_funcs_time = [ip.PchipInterpolator(times, positions_cartesian_resampled[:,i]) for i in range(3)]
    pos_interp_cart = np.column_stack([func(time_slice) for func in pos_interp_funcs_time])
    pos_interp = np.apply_along_axis(c_s_vec_conversion, 1, np.copy(pos_interp_cart))

    new_times = new_times[interval_indices]
    if coord_system == "spherical":
        return pos_interp
    elif coord_system == "cartesian":
        return pos_interp_cart
    else:
        raise ValueError("Invalid coordinate system specified. Must be 'spherical' or 'cartesian'.")
    
# Interpolate Nyquist lat-lon coordinates for IGRF alignment
def interp_lat_lon(times, traj_array, start_time=0, end_time=None, sample_rate=30):
    #times with parameterized sample rate
    if end_time is None:
        end_time = times[-1]
    
    new_times = np.arange(times[0], times[-1],1/sample_rate)
    unique_indices = np.unique(times, return_index=True)[1]
    times = times[unique_indices]
    interval_indices = np.where((new_times >= start_time) & (new_times <= end_time))[0]
    lat = np.array(traj_array["Latgd"])
    lon = np.array(traj_array["Long"])
    alt = np.array(traj_array["Altkm"])
    lat = lat[unique_indices]
    lon = lon[unique_indices]
    alt = alt[unique_indices]
    lat_interp_func = CubicSpline(times, lat, extrapolate=True)
    lon_interp_func = CubicSpline(times, lon, extrapolate=True)
    alt_interp_func = CubicSpline(times, alt, extrapolate=True)

    lat_interp = lat_interp_func(new_times[interval_indices])
    lon_interp = lon_interp_func(new_times[interval_indices])
    alt_interp = alt_interp_func(new_times[interval_indices])
    LLA_interp = np.column_stack([lat_interp, lon_interp, alt_interp])

    return LLA_interp

# Get IGRF for Lat-Lon-Altiude coordinates
def get_mag(LLA):
    td = date.today()
    td = pd.Timestamp(td)
    #east, north, up in nano teslas
    Be, Bn, Bu = pp.igrf(LLA[:,1], LLA[:,0], LLA[:,2], td)
    return np.array([Be, Bn, Bu]).squeeze().T

# Align transmit vector with IGRF
def align_with_mag(LLA, transmitters):
    Bs = get_mag(LLA)
    B_mag = np.linalg.norm(Bs, axis=1)
    #inclination angle
    thetas = np.arccos(Bs[:,2]/ B_mag)
    thetas = np.pi-np.abs(thetas)
    #declination angle - positive means tilts east, negative means tilts west
    declination = np.arctan2(Bs[:,0], Bs[:,1])
    
    thetas_signed = np.sign(declination)*thetas
    x_axis = np.full((len(thetas_signed),3),[1,0,0])
    #reshape for rotation vector
    rot_vecs = thetas_signed[:,np.newaxis]*x_axis

    rot = R.from_rotvec(rot_vecs)
    return rot.apply(transmitters)


# spins transmit vector at given frequency
#TODO: Vectorize this
def spin_whip(times, angular_frequency=2*np.pi, whip_unit_vec=[1,0,0]):
    transmitters = np.full((len(times),3),whip_unit_vec).astype(np.float64)
    z_axis = np.array([0,0,1]).astype(np.float64)
    rot_angles = np.zeros((len(times),3))
    for i in range(len(times)):
        spin_angle = (angular_frequency*times[i])%(2*np.pi)
        rot_angles[i] = spin_angle*z_axis

    spin_rots = R.from_rotvec(rot_angles)

    for i in range(len(times)):
        transmitters[i] = spin_rots[i].apply(transmitters[i])

    return transmitters

############################################ ANTENNA GAIN FUNCTIONS ############################################
# Get NEC data from excel spreadsheet
def data_from_excel(sheet_name):
	signal_data = pd.read_excel(sheet_name)
	# convert angle scales to match traj data
	signal_data['THETA']=signal_data['THETA'].abs()
	# convert complete loss value to avoid screwing up the interpolation
	signal_data.loc[signal_data['TOTAL']<=-900, 'TOTAL']=-20
	return(signal_data)

# Get receiver gain interpolation function 
def rbf_nec_data(signal_data):
	theta_vals = signal_data[['THETA']].to_numpy().flatten()
	phi_vals = signal_data[['PHI']].to_numpy().flatten()
	totals = signal_data['TOTAL'].to_numpy()
	interp = rbf(list(zip(theta_vals, phi_vals)), totals)
	return lambda thet, ph: interp(np.array([[thet, ph]])).item()

# NOTE - New PyNEC installations on unix are broken. Must install version 1.7.3.4.
# Get dipole gain pattern for given frequency, length
def pynec_dipole_gain(frequency, length):
    context = nec_context()
    geo = context.get_geometry()
    center = np.array([0,0,0])
    wire_radius = 0.01e-3
    length = 1
    nr_segments = 11
    half = np.array([length/2,0,0])
    pt1 = center-half
    pt2 = center + half
    wire_tag = 1
    geo.wire(tag_id=wire_tag, segment_count = nr_segments, xw1=pt1[0], yw1=pt1[1], zw1=pt1[2], xw2=pt2[0], yw2=pt2[1], zw2=pt2[2], rad=wire_radius, rdel=1.0, rrad=1.0)
    context.geometry_complete(0)
    context.gn_card(-1,0,0,0,0,0,0,0)
    context.ex_card(0, wire_tag, int(nr_segments/2), 0, 0, 0, 0, 0, 0, 0, 0)
    context.fr_card(ifrq=0,nfrq=1,freq_hz=frequency,del_freq=0)
    context.rp_card(calc_mode=0,n_theta=90,n_phi=180,output_format=0,normalization=5,D=0,A=0,theta0=0.0,phi0=0.0,delta_theta=1.0,delta_phi=5,radial_distance=0.0,gain_norm=0.0)
    rp = context.get_radiation_pattern(0)
    return rp.get_gain()

# Interpolate gain over trajectory
def interpolate_pynec(frequency, length, r_rocket):
    gains = pynec_dipole_gain(frequency, length)
    result = np.zeros(len(r_rocket))
    thetas = np.linspace(0,90,90)
    phis = np.linspace(0,180,180)
    interp = rbs(thetas, phis, gains)
    theta_vals = r_rocket[:,2]
    phi_vals = r_rocket[:,1]
    result = interp(theta_vals, phi_vals, grid=False)
    return result

#TODO: Vectorize this, switch to RBS interpolation
# Get receiver gain from Alexx Lipschultz' NEC data file
def get_receiver_gain(r_rocket, nec_sheet_name):
    signal_data = data_from_excel(nec_sheet_name)
    rbf_f = rbf_nec_data(signal_data)
	# gain calculations
    gains = np.zeros(len(r_rocket))
    for i in range(len(r_rocket)):
         gains[i] = rbf_f(r_rocket[i][2],r_rocket[i][1])
    return gains

############################################ SIGNAL POWER ############################################

freq = 162.990625e6  # transmit frequency (Hz)
#freq = 150e6
Gtx = 2.1  # dBi for transmitter gain
txPwr = 2 # Transmit power in Watts
Bn = 20e3  # Bandwidth
NFrx_dB = 0.5  # Noise figure in dB with a pre-amp
NFrx = 10.0 ** (NFrx_dB / 10)  # Convert to dimensionless quantity
c_speed = 3.0e8  # Speed of light in m/s
kB = 1.38e-23  # Boltzmann constant

def path_loss(rocket_pos):
    return (4*np.pi*rocket_pos[:,0]*freq/c_speed)**2

def calc_received_power(rocket_pos, gains_rx, gains_tx, ploss):
     result_watts = (txPwr * np.multiply(np.multiply(gains_tx,gains_rx),ploss**2))/path_loss(rocket_pos)
     result_watts[result_watts<=0]=1e-100
     result_watts = result_watts.astype(np.float64)
     result_dBm = 10*np.log10(result_watts)+30
     return result_dBm
