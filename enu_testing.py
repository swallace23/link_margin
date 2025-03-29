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
sample_rate = 50

nec_sheet_name = "10152024_nec_data.xlsx"

# ellipsoid model for geodetic coordinates
ell_grs = pm.Ellipsoid.from_name('grs80')


traj_arrays = ny.read_traj_data("Traj_Right.txt")
times = ny.get_times(traj_arrays)
# get geodetic rocket coordinates
rocket_lla = np.zeros((len(times),3))
rocket_lla[:,0] = traj_arrays["Latgd"]
rocket_lla[:,1] = traj_arrays["Long"]
rocket_lla[:,2] = traj_arrays["Altkm"] * 1000

# transform rocket coordinates from geodetic to ENU with PF origin
rocket_enu = np.column_stack(pm.geodetic2enu(rocket_lla[:,0], rocket_lla[:,1], rocket_lla[:,2], lat_pf, long_pf, 0, ell=ell_grs))

# remove duplicate time steps
duplicate_indices = [i for i in range(1, len(times)) if times[i] <= times[i-1]]
rocket_enu = np.delete(rocket_enu, duplicate_indices, axis=0)
rocket_lla = np.delete(rocket_lla,duplicate_indices, axis=0)
times = np.delete(times, duplicate_indices)

times_interp, rocket_enu_interp = ut.interp_time_position(times, sample_rate, rocket_enu)
rocket_lla_interp = ut.interp_time_position(times, sample_rate, rocket_lla)[1]
rocket_spherical_interp = np.apply_along_axis(ut.c_s_vec_conversion,1,rocket_enu_interp)

receivers_ew = np.full((len(times_interp),3),[1,0,0]).astype(np.float64)
receivers_ns = np.full((len(times_interp),3),[0,1,0]).astype(np.float64)


# numpy array of receiver coordinates for correct broadcasting
rec_lats = np.full(len(times_interp), lat_pf)
rec_lons = np.full(len(times_interp), long_pf)

r_aligned = ut.align_mag(rocket_lla_interp, rec_lats, rec_lons)

transmitters = ut.spin(times_interp, r_aligned, 0.5)
thetas = ut.get_thetas(rocket_enu_interp)
phis = ut.get_phis(rocket_enu_interp)
radius = np.linalg.norm(rocket_enu_interp, axis=1)
rx_gains = ut.get_rx_gain(nec_sheet_name, thetas, phis)

# get polarization losses for each receiver
losses_ew=np.zeros(len(times_interp))
losses_ns=np.zeros(len(times_interp))
losses_ew = ut.get_polarization_loss(receivers_ew,transmitters,rocket_enu_interp)
losses_ns = ut.get_polarization_loss(receivers_ns,transmitters,rocket_enu_interp)

# gains
tx_gains = ut.interpolate_pynec(162.99, 1, thetas, phis)
signal_ew = ut.calc_received_power(radius, rx_gains, tx_gains, losses_ew)
signal_ns = ut.calc_received_power(radius, rx_gains, tx_gains, losses_ns)
plt.plot(times_interp, signal_ew, label="EW")
plt.plot(times_interp, signal_ns, label="NS")
plt.ylim(-200,-115)
plt.legend()
plt.show()