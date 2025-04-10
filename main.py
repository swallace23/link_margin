import numpy as np
import matplotlib.pyplot as plt
import utilities as ut
import nyquist as ny
import pymap3d as pm
from enum import Enum

############################################ Global Parameters ############################################

# Receiver enum - comment out receivers you don't want to plot
class Receiver(Enum):
    # PF = 0
    # VT = 1
    TL = 2
    # AV = 3

# Receiving sites -- from SDR document
# PF == poker flat
lat_pf = 65.1192
long_pf = -147.43

# VT == venetie
lat_vt = 67.013
long_vt = -146.407

# BV == beaver
lat_bv = 66.36
long_bv = -147.4

# AV == arctic village 
lat_av = 68.113
long_av = -147.575
# TL == toolik
lat_tl = 68.627
long_tl = -149.598

coords = np.array([[lat_pf, long_pf], [lat_vt, long_vt], [lat_tl, long_tl], [lat_av, long_av]])

# sample rate for interpolation
sample_rate = 50
# rocket spin freq
omega = np.pi/2
# ellipsoid model for geodetic coordinate conversions
ell_grs = pm.Ellipsoid.from_name('grs80')
# receiver gain - from Alexx' NEC model
nec_sheet_name = "10152024_nec_data.xlsx"
# Giraff GPS data
gps_sheet_name = "giraff_gps.xlsx"

# Giraff vs GNEISS trajectory
isGiraff = True
# consider receiver gain pattern
receiveGain = True
############################################# Data Generation ############################################
if isGiraff:
    times, raw_lla = ny.read_gps_file(gps_sheet_name)
else:
    traj_arrays = ny.read_traj_data("Traj_Right.txt")
    times = ny.get_times(traj_arrays)
    # must convert km to m for all the calculations
    raw_lla = np.stack([traj_arrays["Latgd"], traj_arrays["Long"], traj_arrays["Altkm"] * 1000], axis=1)

# Remove duplicate time steps
valid_indices = np.where(np.diff(times, prepend=times[0] - 1) > 0)[0]
times = times[valid_indices]
raw_lla = raw_lla[valid_indices]


# Interpolate trajectory
times_interp, rocket_lla_interp = ut.interp_time_position(times, sample_rate, raw_lla)

# Constant magnetic field approximation from NOAA calculator.
mag_vec_spherical = np.array([1, np.radians(14.5694), np.radians(90 + 77.1489)])
# generate spinning transmitter orthogonal to magnetic field approximation
transmitters = ut.get_transmitters(mag_vec_spherical, times_interp, omega)

# store signals at each receiver in a dictionary
signals = {}


for recv in Receiver:
    lat, lon = coords[recv.value]

    # convert LLA to local ENU (cartesian) coordinates centered at receiver
    rocket_enu = np.column_stack(pm.geodetic2enu(
        rocket_lla_interp[:, 0],
        rocket_lla_interp[:, 1],
        rocket_lla_interp[:, 2],
        lat, lon, 0, ell=ell_grs
    ))
    # adjust altitude wrt receiver above sea level
    offset = rocket_enu[0,2]
    rocket_enu[:,2] = rocket_enu[:,2] - offset
    # avoid division by zero
    idx = np.where(rocket_enu[:,2] == 0)[0]
    rocket_enu[idx,2] = 0.0001
    # spherical coordinates for path length and gain calculations
    radius = np.linalg.norm(rocket_enu, axis=1)
    thetas = ut.get_thetas(rocket_enu)
    phis = ut.get_phis(rocket_enu)
    rx_gains = ut.get_rx_gain(nec_sheet_name, thetas, phis)
    tx_gains = ut.get_tx_gain(162.99, 1, thetas, phis)
    rec_ew = np.full_like(rocket_enu, [1, 0, 0], dtype=np.float64)
    rec_ns = np.full_like(rocket_enu, [0, 1, 0], dtype=np.float64)

    # account for transmitter - receiver orientation
    losses_ew = ut.get_polarization_loss(rec_ew, transmitters, rocket_enu)
    losses_ns = ut.get_polarization_loss(rec_ns, transmitters, rocket_enu)
    if receiveGain:
        signal_ew = ut.calc_received_power(radius, rx_gains, tx_gains, losses_ew)
        signal_ns = ut.calc_received_power(radius, rx_gains, tx_gains, losses_ns)
    else:
        signal_ew = ut.calc_received_power(radius, 1, tx_gains, losses_ew)
        signal_ns = ut.calc_received_power(radius, 1, tx_gains, losses_ns)

    signals[recv] = (signal_ew, signal_ns)

############################################## Plotting ##################################################
fig, axs = plt.subplots(2,2)
axs[0,0].plot(times_interp, radius)
axs[0,0].set_title("Radius")
axs[0,0].set_ylabel("Radius (m)")
axs[1,0].plot(times_interp, np.abs(np.abs(thetas)-np.pi/2))
axs[1,0].set_title("(Absolute) Elevation Angle")
axs[1,0].set_ylabel("Angle (rad)")
for recv in Receiver:
    ew, ns = signals[recv]
    axs[0,1].plot(times_interp, ew, label=f"{recv.name} EW")
    axs[0,1].plot(times_interp, ns, label=f"{recv.name} NS")
axs[0,1].set_title("Received Power")
axs[0,1].set_ylabel("Watts")
axs[1,1].plot(times_interp, ew+ns)
axs[1,1].set_title("Sum of Received Power")
axs[1,1].set_ylabel("Watts")
# plt.gca().set_ylim(bottom=-130)
axs[1,0].set_xlabel("Time (s)")
axs[1,1].set_xlabel("Time (s)")
plt.xlim(0,600)
plt.show()
