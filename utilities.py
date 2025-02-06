import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline
from scipy.interpolate import RBFInterpolator as rbf
import pandas as pd

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

############################################ RECEIVER GAIN FUNCTIONS ############################################
# Get NEC data from excel spreadsheet
def data_from_excel(sheet_name):
	signal_data = pd.read_excel(sheet_name)
	# convert angle scales to match traj data
	signal_data['THETA']=signal_data['THETA'].abs()
	# convert complete loss value to avoid screwing up the interpolation
	signal_data.loc[signal_data['TOTAL']<=-900, 'TOTAL']=-20
	return(signal_data)

# Get interpolated function with Radial Basis Function 
def rbf_nec_data(signal_data):
	theta_vals = signal_data[['THETA']].to_numpy().flatten()
	phi_vals = signal_data[['PHI']].to_numpy().flatten()
	totals = signal_data['TOTAL'].to_numpy()
	interp = rbf(list(zip(theta_vals, phi_vals)), totals)
	return lambda thet, ph: interp(np.array([[thet, ph]])).item()
#interpolate trajectory data

def get_receiver_gain(r_rocket, nec_sheet_name):
    signal_data = data_from_excel(nec_sheet_name)
	# interpolate function from NEC data
    rbf_f = rbf_nec_data(signal_data)
	# gain calculations
    gains = np.zeros(len(r_rocket))
    for i in range(len(r_rocket)):
         gains[i] = rbf_f(r_rocket[i][2],r_rocket[i][1])
    #for phi, theta in zip(r_rocket[:,2], r_rocket[:,1]):
    #    gains.append(rbf_f(theta, phi))
    return gains

############################################ SIGNAL VISUALIZATION FUNCTIONS ############################################

freq = 162.990625e6  # transmit frequency (Hz)
Gtx = 2.1  # dBi for transmitter gain
txPwr = 1.5
Bn = 20e3  # Bandwidth
NFrx_dB = 0.5  # Noise figure in dB with a pre-amp
NFrx = 10.0 ** (NFrx_dB / 10)  # Convert to dimensionless quantity
c_speed = 3.0e8  # Speed of light in m/s
kB = 1.38e-23  # Boltzmann constant

def calc_received_power(rocket_pos, gains, ploss):
    result = np.zeros(len(rocket_pos))
    for i in range(len(rocket_pos)):
        Lpath = (4.0*np.pi*rocket_pos[i][0]*freq/c_speed)**2

        Pwr_rx = (txPwr * Gtx * gains[i]*(ploss[i]**2))/Lpath

        if Pwr_rx<=0:
            Pwr_rx = 1e-100
        Pwr_rx_dBW = 10 * np.log10(Pwr_rx)
        result[i] = Pwr_rx_dBW+30
    """
    for radius, gain, pol_loss in zip(rocket_pos, gains, ploss):
        # Path loss calculation
        Lpath = (4.0 * np.pi * radius * freq / c_speed) ** 2

        # Received power calculation in Watts

        Pwr_rx = (txPwr * Gtx * gain*(pol_loss**2)) / Lpath
        if Pwr_rx <= 0:
            Pwr_rx = 1e-100
        Pwr_rx_dBW = 10 * np.log10(Pwr_rx)
        # Append the calculated power to the list, converting to dbm
        result.append(Pwr_rx_dBW+30)
    """
    return result