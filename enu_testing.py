"""
- TODO: appears that whip is not always orthogonal to the magnetic field
late night idea - for spin function - assume cylindrical coordinate system centered on magnetic field vector, calculate angle from spin rate, r is one, z is zero.
calculate general change of coordinates matrix for this coordinate transformation and then apply it to the whole data. maybe easier as composition of two transformations:
first rotating the z axis, then transforming from standard cylindrical to cartesian (equivalent to ENU given mag vectors in local ENU)
"""
"""
Data format: 
- Geodetic (LLA) coordinates structured latitude, longitude, altitude. Altitude in meters.
- ENU = East, North, Up. Calculations are done in ENU coordinates centered at the receiver.
"""
# testing link budget calculations with ENU coordinates
import numpy as np
import matplotlib.pyplot as plt
import utilities as ut
import nyquist as ny
import pymap3d as pm

############################################ Global Parameters ############################################
# Receiving sites -- from SDR document
# PF == poker flat
lat_pf = 65.1192
long_pf = -147.43
lat_tl = 68.627
long_tl = -149.598
sample_rate = 50

nec_sheet_name = "10152024_nec_data.xlsx"

# ellipsoid model for geodetic coordinates
ell_grs = pm.Ellipsoid.from_name('grs80')

############################################# Data Generation ############################################
# read trajectory data
traj_arrays = ny.read_traj_data("Traj_Right.txt")
times = ny.get_times(traj_arrays)
rocket_lla = np.zeros((len(times),3))
rocket_lla[:,0] = traj_arrays["Latgd"]
rocket_lla[:,1] = traj_arrays["Long"]
rocket_lla[:,2] = traj_arrays["Altkm"] * 1000

# transform rocket coordinates from geodetic to ENU with PF origin
#rocket_enu = np.column_stack(pm.geodetic2enu(rocket_lla[:,0], rocket_lla[:,1], rocket_lla[:,2], lat_pf, long_pf, 0, ell=ell_grs))
# toolik instead
rocket_enu = np.column_stack(pm.geodetic2enu(rocket_lla[:,0], rocket_lla[:,1], rocket_lla[:,2], lat_tl, long_tl, 0, ell=ell_grs))
# remove duplicate time steps
duplicate_indices = [i for i in range(1, len(times)) if times[i] <= times[i-1]]
rocket_enu = np.delete(rocket_enu, duplicate_indices, axis=0)
rocket_lla = np.delete(rocket_lla,duplicate_indices, axis=0)
times = np.delete(times, duplicate_indices)

times_interp, rocket_enu_interp = ut.interp_time_position(times, sample_rate, rocket_enu)
rocket_lla_interp = ut.interp_time_position(times, sample_rate, rocket_lla)[1]

# define receiver vectors
receivers_ew = np.full((len(times_interp),3),[1,0,0]).astype(np.float64)
receivers_ns = np.full((len(times_interp),3),[0,1,0]).astype(np.float64)


# numpy array of receiver coordinates for correct broadcasting
# rec_lats = np.full(len(times_interp), lat_pf)
# rec_lons = np.full(len(times_interp), long_pf)
# toolik instead
rec_lats = np.full(len(times_interp), lat_tl)
rec_lons = np.full(len(times_interp), long_tl)

# Constant magnetic field approximation
mag_vec_spherical = np.array([1, np.radians(14.5694), np.radians(90+77.1489)])
transmitters = ut.get_transmitters(mag_vec_spherical, times_interp, np.pi)
# transmitters = ut.spin(times_interp, r_aligned, 0.5)

# get receiver and transmitter gains along trajectory
thetas = ut.get_thetas(rocket_enu_interp)
phis = ut.get_phis(rocket_enu_interp)
radius = np.linalg.norm(rocket_enu_interp, axis=1)

rx_gains = ut.get_rx_gain(nec_sheet_name, thetas, phis)
tx_gains = ut.get_tx_gain(162.99, 1, thetas, phis)


# get polarization losses for each receiver
losses_ew = ut.get_polarization_loss(receivers_ew,transmitters,rocket_enu_interp)
losses_ns = ut.get_polarization_loss(receivers_ns,transmitters,rocket_enu_interp)

# get signal power
#temp_tx_gains = np.full(len(times_interp), 1)
signal_ew = ut.calc_received_power(radius, rx_gains, tx_gains, losses_ew)
signal_ns = ut.calc_received_power(radius, rx_gains, tx_gains, losses_ns)

plt.title("Signal Strength at Toolik")
plt.xlabel("Time (s)")
plt.ylabel("Signal Strength (dBm)")
plt.plot(times_interp, signal_ew, label="EW")
plt.plot(times_interp, signal_ns, label="NS")
plt.ylim(-200,-115)
plt.legend()
plt.show()