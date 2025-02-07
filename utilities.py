import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline
from scipy.interpolate import RBFInterpolator as rbf
from scipy.interpolate import RectBivariateSpline as rbs
import pandas as pd
from PyNEC import *

############################################ POLARIZATION LOSS FUNCTIONS ############################################
def clip_norm_dots(vec1, vec2):
     return np.clip(np.dot(vec1, vec2),0,0.9999)

def get_polarization_loss(receiver, source, separation):
	receiver_hat = receiver / np.linalg.norm(receiver)
	source_hat = source / np.linalg.norm(source)
	separation_hat = separation / np.linalg.norm(separation)
	return (clip_norm_dots(receiver_hat,source_hat)-(clip_norm_dots(receiver_hat,separation_hat)*clip_norm_dots(source_hat,separation_hat)))/(np.sqrt(1-clip_norm_dots(receiver_hat,separation_hat)**2)*np.sqrt(1-clip_norm_dots(source_hat,separation_hat)**2))

# convert numpy vector from spherical to cartesian coordinates
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

#interpolates Nyquist position to given sample rate. Optionally specify time interval for performance.
def interp_position(times, sample_rate, position_vector, start_time=0, end_time= None, coord_system = "spherical"):
    #times with parameterized sample rate
    if end_time is None:
        end_time = times[-1]
    
    new_times = np.arange(times[0], times[-1],1/sample_rate)

    position_vector[:, 1] = np.radians(position_vector[:, 1])  
    position_vector[:, 2] = np.radians(position_vector[:, 2])

    # Find the indices corresponding to the time interval
    unique_indices = np.unique(times, return_index=True)[1]
    times = times[unique_indices]
    position_vector = position_vector[unique_indices]

    interval_indices = np.where((new_times >= start_time) & (new_times <= end_time))[0]


    # Interpolate position functions for the specified time interval
    pos_interp_funcs = [CubicSpline(times, position_vector[:,i], extrapolate=True) for i in range(position_vector.shape[1])]

    pos_interp = np.column_stack([func(new_times[interval_indices]) for func in pos_interp_funcs])

    # Get cartesian versions for the specified time interval
    pos_interp_cart = np.apply_along_axis(s_c_vec_conversion, 1, np.copy(pos_interp))

    """# Interpolate lat/lon/alt for IGRF alignment for the specified time interval
    lat = np.array(traj_arraysrt["Latgd"])
    lon = np.array(traj_arraysrt["Long"])
    alt = np.array(traj_arraysrt["Altkm"])
    lat = lat[unique_indices]
    lon = lon[unique_indices]
    alt = alt[unique_indices]
    lat_interp_func = CubicSpline(times, lat, extrapolate=True)
    lon_interp_func = CubicSpline(times, lon, extrapolate=True)
    alt_interp_func = CubicSpline(times, alt, extrapolate=True)

    lat_interp = lat_interp_func(new_times[interval_indices])
    lon_interp = lon_interp_func(new_times[interval_indices])
    alt_interp = alt_interp_func(new_times[interval_indices])"""

    new_times = new_times[interval_indices]
    if coord_system == "spherical":
        return new_times, pos_interp
    elif coord_system == "cartesian":
        return new_times, pos_interp_cart
    else:
        raise ValueError("Invalid coordinate system specified. Must be 'spherical' or 'cartesian'.")
    
# spins unit vector at given frequency
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

def interpolate_pynec(frequency, length, r_rocket):
    gains = pynec_dipole_gain(frequency, length)
    result = np.zeros(len(r_rocket))
    thetas = np.linspace(0,90,90)
    phis = np.linspace(0,180,180)
    theta_grid, phi_grid = np.meshgrid(thetas,phis,indexing='ij')
    y = np.column_stack([theta_grid.ravel(),phi_grid.ravel()])
    data = gains.ravel()
    #interp = rbf(y,data)
    interp = rbs(thetas, phis, gains)
    theta_vals = r_rocket[:,2]
    phi_vals = r_rocket[:,1]
    result = interp(theta_vals, phi_vals, grid=False)
    return result

# Get receiver gain 
def get_receiver_gain(r_rocket, nec_sheet_name):
    signal_data = data_from_excel(nec_sheet_name)
    rbf_f = rbf_nec_data(signal_data)
	# gain calculations
    gains = np.zeros(len(r_rocket))
    for i in range(len(r_rocket)):
         gains[i] = rbf_f(r_rocket[i][2],r_rocket[i][1])
    return gains

############################################ SIGNAL VISUALIZATION FUNCTIONS ############################################

freq = 162.990625e6  # transmit frequency (Hz)
Gtx = 2.1  # dBi for transmitter gain
txPwr = 2 # Transmit power in Watts
Bn = 20e3  # Bandwidth
NFrx_dB = 0.5  # Noise figure in dB with a pre-amp
NFrx = 10.0 ** (NFrx_dB / 10)  # Convert to dimensionless quantity
c_speed = 3.0e8  # Speed of light in m/s
kB = 1.38e-23  # Boltzmann constant

def calc_received_power(rocket_pos, gains_rx, gains_tx, ploss):
    result = np.zeros(len(rocket_pos))
    for i in range(len(rocket_pos)):
        Lpath = (4.0*np.pi*rocket_pos[i][0]*freq/c_speed)**2

        # Power in Watts
        Pwr_rx = (txPwr * gains_tx[i] * gains_rx[i]*(ploss[i]**2))/Lpath

        if Pwr_rx<=0:
            Pwr_rx = 1e-100
        # Convert to dBW
        Pwr_rx_dBW = 10 * np.log10(Pwr_rx)
        # Convert to dBm
        result[i] = Pwr_rx_dBW+30
    return result