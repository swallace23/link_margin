# This script is designed to work as an executable to generate plots.
import argparse
parser = argparse.ArgumentParser(description='Generate link budget plots', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("start_time", type=int, help="Start time for plot")
parser.add_argument("end_time", type=int, help="End time for plot")
parser.add_argument('-sr', '--sample_rate', type=int, default=10, help="Sample rate for interpolation. Reduce to improve performance for larger time slices.")
parser.add_argument('-ymin', type=int, default=-95, help="Minimum y-axis value for plot")
parser.add_argument('-ymax', type=int, default=-70, help="Maximum y-axis value for plot")
args = vars(parser.parse_args())

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import utilities as ut
import nyquist as ny
import animation as ani

############################################ Global Parameters ############################################
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

nec_sheet_name = "10152024_nec_data.xlsx"

start_time = args["start_time"]
end_time = args["end_time"]
sample_rate = 30

############################################ Data Generation ############################################


traj_arrays = ny.read_traj_data("Traj_Right.txt")
times, r_pf = ny.get_spherical_position(lat_pf, long_pf, traj_arrays)
r_vt = ny.get_spherical_position(lat_vt, long_vt, traj_arrays)[1]
r_tl = ny.get_spherical_position(lat_tl, long_tl, traj_arrays)[1]

times_interp, r_pf_interp = ut.interp_position(times, sample_rate, r_pf, start_time, end_time)
r_pf_interp_cart = ut.interp_position(times, sample_rate, r_pf, start_time, end_time, coord_system="cartesian")[1]
r_vt_interp = ut.interp_position(times, sample_rate, r_vt, start_time=start_time, end_time=end_time)[1]
r_vt_interp_cart = ut.interp_position(times, sample_rate, r_vt,start_time=start_time, end_time=end_time, coord_system="cartesian")[1]
r_tl_interp = ut.interp_position(times, sample_rate, r_tl,start_time=start_time,end_time=end_time)[1]
r_tl_interp_cart = ut.interp_position(times, sample_rate, r_tl,start_time=start_time,end_time=end_time, coord_system="cartesian")[1]




receivers_ew = np.full((len(times_interp),3),[1,0,0]).astype(np.float64)
receivers_ns = np.full((len(times_interp),3),[0,1,0]).astype(np.float64)

transmitters = ut.spin_whip(times_interp, np.pi, [1,0,0])
pf_losses_ew = np.zeros(len(times_interp))
pf_losses_ns = np.zeros(len(times_interp))
vt_losses_ew = np.zeros(len(times_interp))
vt_losses_ns = np.zeros(len(times_interp))
tl_losses_ew = np.zeros(len(times_interp))
tl_losses_ns = np.zeros(len(times_interp))

############################################ Calculations ############################################
# polarization losses
for i in range(len(times_interp)):
    pf_losses_ew[i]=ut.get_polarization_loss(receivers_ew[i],transmitters[i],r_pf_interp_cart[i])
    pf_losses_ns[i]=ut.get_polarization_loss(receivers_ns[i],transmitters[i],r_pf_interp_cart[i])
    vt_losses_ew[i]=ut.get_polarization_loss(receivers_ew[i],transmitters[i],r_vt_interp_cart[i])
    vt_losses_ns[i]=ut.get_polarization_loss(receivers_ns[i],transmitters[i],r_vt_interp_cart[i])
    tl_losses_ew[i]=ut.get_polarization_loss(receivers_ew[i],transmitters[i],r_tl_interp_cart[i])
    tl_losses_ns[i]=ut.get_polarization_loss(receivers_ns[i],transmitters[i],r_tl_interp_cart[i])
# gains
pf_gains = ut.get_receiver_gain(r_pf_interp, nec_sheet_name)
vt_gains = ut.get_receiver_gain(r_vt_interp, nec_sheet_name)
tl_gains = ut.get_receiver_gain(r_tl_interp, nec_sheet_name)

transmit_gains = ut.interpolate_pynec(162.99, 1, r_pf_interp)


# signal strength
pf_signal_strength_ew = ut.calc_received_power(r_pf_interp,pf_gains, pf_losses_ew, transmit_gains)
pf_signal_strength_ns = ut.calc_received_power(r_pf_interp,pf_gains, pf_losses_ns, transmit_gains)
vt_signal_strength_ew = ut.calc_received_power(r_vt_interp,vt_gains, vt_losses_ew, transmit_gains)
vt_signal_strength_ns = ut.calc_received_power(r_vt_interp,vt_gains, vt_losses_ns, transmit_gains)
tl_signal_strength_ew = ut.calc_received_power(r_tl_interp,tl_gains, tl_losses_ew, transmit_gains)
tl_signal_strength_ns = ut.calc_received_power(r_tl_interp,tl_gains, tl_losses_ns, transmit_gains)

# Plots
plt.ylim(-95,-70)
plt.title('Signal Strength across Receivers')
plt.ylabel('Signal Strength (dBm)')
plt.xlabel('Time (s)')
plt.plot(times_interp,pf_signal_strength_ew, label='Poker Flat EW')
plt.plot(times_interp,pf_signal_strength_ns, label='Poker Flat NS', linestyle='dashed')
plt.plot(times_interp,vt_signal_strength_ew, label='Venetie EW')
plt.plot(times_interp,vt_signal_strength_ns, label='Venetie NS', linestyle='dashed')
plt.plot(times_interp,tl_signal_strength_ew, label='Toolik EW')
plt.plot(times_interp,tl_signal_strength_ns, label='Toolik NS', linestyle='dashed')
plt.legend()
plt.savefig("output.png")

