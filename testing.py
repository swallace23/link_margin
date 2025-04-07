import numpy as np
import matplotlib.pyplot as plt
import utilities as ut
import nyquist as ny
import pymap3d as pm
from enum import Enum

############################################ Global Parameters ############################################
# Receiver coordinates
class Receiver(Enum):
    PF = 0
    VT = 1
    TL = 2

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
coords = np.array([[lat_pf, long_pf], [lat_vt, long_vt], [lat_tl, long_tl]])

sample_rate = 50
ell_grs = pm.Ellipsoid.from_name('grs80')
nec_sheet_name = "10152024_nec_data.xlsx"

############################################# Data Generation ############################################
traj_arrays = ny.read_traj_data("Traj_Right.txt")
times = ny.get_times(traj_arrays)
raw_lla = np.stack([traj_arrays["Latgd"], traj_arrays["Long"], traj_arrays["Altkm"] * 1000], axis=1)

# Remove duplicate time steps
valid_indices = np.where(np.diff(times, prepend=times[0] - 1) > 0)[0]
times = times[valid_indices]
raw_lla = raw_lla[valid_indices]

# Interpolate trajectory
times_interp, rocket_lla_interp = ut.interp_time_position(times, sample_rate, raw_lla)

# Constant magnetic field approximation
mag_vec_spherical = np.array([1, np.radians(14.5694), np.radians(90 + 77.1489)])
transmitters = ut.get_transmitters(mag_vec_spherical, times_interp, np.pi)
thetas = ut.get_thetas(rocket_lla_interp)
phis = ut.get_phis(rocket_lla_interp)
radius = np.linalg.norm(rocket_lla_interp, axis=1)

rx_gains = ut.get_rx_gain(nec_sheet_name, thetas, phis)
tx_gains = ut.get_tx_gain(162.99, 1, thetas, phis)

# Initialize signal storage
signals = {}

for recv in Receiver:
    lat, lon = coords[recv.value]

    # Recalculate ENU coordinates relative to this receiver
    rocket_enu = np.column_stack(pm.geodetic2enu(
        rocket_lla_interp[:, 0],
        rocket_lla_interp[:, 1],
        rocket_lla_interp[:, 2],
        lat, lon, 0, ell=ell_grs
    ))

    # Compute spherical coordinates relative to this receiver
    radius = np.linalg.norm(rocket_enu, axis=1)
    thetas = ut.get_thetas(rocket_enu)
    phis = ut.get_phis(rocket_enu)

    rx_gains = ut.get_rx_gain(nec_sheet_name, thetas, phis)
    tx_gains = ut.get_tx_gain(162.99, 1, thetas, phis)

    rec_ew = np.full_like(rocket_enu, [1, 0, 0], dtype=np.float64)
    rec_ns = np.full_like(rocket_enu, [0, 1, 0], dtype=np.float64)

    losses_ew = ut.get_polarization_loss(rec_ew, transmitters, rocket_enu)
    losses_ns = ut.get_polarization_loss(rec_ns, transmitters, rocket_enu)

    signal_ew = ut.calc_received_power(radius, rx_gains, tx_gains, losses_ew)
    signal_ns = ut.calc_received_power(radius, rx_gains, tx_gains, losses_ns)

    signals[recv] = (signal_ew, signal_ns)

############################################### Plotting ##################################################
plt.title("Signal Strength at Receivers")
plt.xlabel("Time (s)")
plt.ylabel("Signal Strength (dBm)")
for recv in Receiver:
    ew, ns = signals[recv]
    plt.plot(times_interp, ew, label=f"{recv.name} EW")
    plt.plot(times_interp, ns, label=f"{recv.name} NS")
plt.ylim(-200, -115)
plt.legend()
plt.show()
