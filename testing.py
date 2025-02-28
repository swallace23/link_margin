# Core Link Budget Calculations
# note - rocket position variables go r, theta phi and x, y, z
import numpy as np
import matplotlib.pyplot as plt
import utilities as ut
import nyquist as ny
# import animation as ani
from enum import Enum

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

start_time = 200
end_time = 220
sample_rate = 50

class Receiver(Enum):
    PF = 0
    VT = 1
    TL = 2

coords = np.array([[lat_pf, long_pf], [lat_vt, long_vt], [lat_tl, long_tl]])

############################################ Data Generation ############################################

traj_arrays = ny.read_traj_data("Traj_Right.txt")
times = ny.get_times(traj_arrays)
positions = np.zeros((len(times),len(Receiver),3))
for i in range(len(Receiver)):
    positions[:,i] = ny.get_spherical_position(coords[i][0], coords[i][1], traj_arrays)

times_interp = ut.interp_time(times, sample_rate, start_time, end_time)
positions_interp = np.zeros((len(times_interp),len(Receiver),3))
positions_interp_cartesian = np.zeros((len(times_interp),len(Receiver),3))
for i in range(len(Receiver)):
    positions_interp[:,i] = ut.interp_position(times, sample_rate, positions[:,i], start_time, end_time)
    positions_interp_cartesian[:,i] = ut.interp_position(times, sample_rate, positions[:,i], start_time, end_time, coord_system="cartesian")

lat_lon = ut.interp_lat_lon(times, traj_arrays, start_time, end_time, sample_rate)


receivers_ew = np.full((len(times_interp),3),[1,0,0]).astype(np.float64)
receivers_ns = np.full((len(times_interp),3),[0,1,0]).astype(np.float64)

transmitters = ut.spin_whip(times_interp, np.pi, [1,0,0])
transmitters_aligned = ut.align_with_mag(lat_lon, transmitters).astype(np.float64)
losses = np.zeros((len(times_interp),len(Receiver),2))
for i in range(len(Receiver)):
    losses[:,i,0] = ut.get_polarization_loss(receivers_ew,transmitters_aligned,positions_interp_cartesian[:,i])
    losses[:,i,1] = ut.get_polarization_loss(receivers_ns,transmitters_aligned,positions_interp_cartesian[:,i])

############################################ Calculations ############################################
# polarization losses
gains = np.zeros((len(times_interp),len(Receiver)))
for i in range(len(Receiver)):
    gains[:,i] = ut.get_receiver_gain(positions_interp[:,i], nec_sheet_name)

# gains
transmit_gains = ut.interpolate_pynec(162.99, 1, positions_interp[:,Receiver.PF.value])



# signal strength
signal_strengths = np.zeros((len(times_interp),len(Receiver),2))
for i in range(len(Receiver)):
    signal_strengths[:,i,0] = ut.calc_received_power(positions_interp[:,i], gains[:,i], transmit_gains, losses[:,i,0])
    signal_strengths[:,i,1] = ut.calc_received_power(positions_interp[:,i], gains[:,i], transmit_gains, losses[:,i,1])

# Plots
plt.ylim(-200,-115)
plt.title('Signal Strength across Receivers')
plt.ylabel('Signal Strength (dBm)')
plt.xlabel('Time (s)')
plt.plot(times_interp,signal_strengths[:,Receiver.PF.value,0], label='Poker Flat EW')
plt.plot(times_interp,signal_strengths[:,Receiver.PF.value,1], label='Poker Flat NS', linestyle='dashed')
plt.plot(times_interp,signal_strengths[:,Receiver.VT.value,0], label='Venetie EW')
plt.plot(times_interp,signal_strengths[:,Receiver.VT.value,1], label='Venetie NS', linestyle='dashed')
plt.plot(times_interp,signal_strengths[:,Receiver.TL.value,0], label='Toolik EW')
plt.plot(times_interp,signal_strengths[:,Receiver.TL.value,1], label='Toolik NS', linestyle='dashed')
plt.legend()
plt.show()



################### Sample Rate Debugging ###################
# import numpy as np
# import matplotlib.pyplot as plt
# import utilities as ut
# import nyquist as ny
# from enum import Enum

# # Define sample rates to compare
# sample_rates = [50,100,200]

# # Receiving site coordinates
# lat_pf, long_pf = 65.1192, -147.43
# lat_vt, long_vt = 67.013, -146.407
# lat_tl, long_tl = 68.627, -149.598
# coords = np.array([[lat_pf, long_pf], [lat_vt, long_vt], [lat_tl, long_tl]])

# nec_sheet_name = "10152024_nec_data.xlsx"
# start_time = 240
# end_time = 250

# class Receiver(Enum):
#     PF = 0
#     VT = 1
#     TL = 2

# # Read trajectory data
# traj_arrays = ny.read_traj_data("Traj_Right.txt")
# times = ny.get_times(traj_arrays)
# positions = np.zeros((len(times), len(Receiver), 3))

# for i in range(len(Receiver)):
#     positions[:, i] = ny.get_spherical_position(coords[i][0], coords[i][1], traj_arrays)
# # Create subplots
# fig, axes = plt.subplots(len(sample_rates), figsize=(8, 2 * len(sample_rates)), sharex=True)
# import scipy.stats as stats
# for idx, sample_rate in enumerate(sample_rates):
#     # Interpolate times and positions
#     times_interp = ut.interp_time(times, sample_rate, start_time, end_time)
#     positions_interp = np.zeros((len(times_interp), len(Receiver), 3))
#     positions_interp_cartesian = np.zeros((len(times_interp),len(Receiver),3))
#     for i in range(len(Receiver)):
#         positions_interp[:, i] = ut.interp_position(times, sample_rate, positions[:, i], start_time, end_time)
#         positions_interp_cartesian[:,i] = ut.interp_position(times, sample_rate, positions[:,i], start_time, end_time, coord_system="cartesian")

#     lat_lon = ut.interp_lat_lon(times, traj_arrays, start_time, end_time, sample_rate)

#     receivers_ew = np.full((len(times_interp), 3), [1, 0, 0]).astype(np.float64)
#     receivers_ns = np.full((len(times_interp),3),[0,1,0]).astype(np.float64)
#     transmitters = ut.spin_whip(times_interp, np.pi, [1, 0, 0])
#     transmitters_aligned = ut.align_with_mag(lat_lon, transmitters).astype(np.float64)

#     losses = np.zeros((len(times_interp),len(Receiver),2))
#     for i in range(len(Receiver)):
#         losses[:,i,0] = ut.get_polarization_loss(receivers_ew,transmitters_aligned,positions_interp_cartesian[:,i])
#         losses[:,i,1] = ut.get_polarization_loss(receivers_ns,transmitters_aligned,positions_interp_cartesian[:,i])

#     ############################################ Calculations ############################################
#     # polarization losses
#     gains = np.zeros((len(times_interp),len(Receiver)))
#     for i in range(len(Receiver)):
#         gains[:,i] = ut.get_receiver_gain(positions_interp[:,i], nec_sheet_name)

#     # gains
#     transmit_gains = ut.interpolate_pynec(162.99, 1, positions_interp[:,Receiver.PF.value])



#     # signal strength
#     signal_strengths = np.zeros((len(times_interp),len(Receiver),2))
#     for i in range(len(Receiver)):
#         signal_strengths[:,i,0] = ut.calc_received_power(positions_interp[:,i], gains[:,i], transmit_gains, losses[:,i,0])
#         signal_strengths[:,i,1] = ut.calc_received_power(positions_interp[:,i], gains[:,i], transmit_gains, losses[:,i,1])

# # plt.ylim(-200,-115)
# # plt.title('Signal Strength across Receivers')
# # plt.ylabel('Signal Strength (dBm)')
# # plt.xlabel('Time (s)')
# # plt.plot(times_interp,signal_strengths[:,Receiver.PF.value,0], label='Poker Flat EW')
# # plt.plot(times_interp,signal_strengths[:,Receiver.PF.value,1], label='Poker Flat NS', linestyle='dashed')
# # plt.plot(times_interp,signal_strengths[:,Receiver.VT.value,0], label='Venetie EW')
# # plt.plot(times_interp,signal_strengths[:,Receiver.VT.value,1], label='Venetie NS', linestyle='dashed')
# # plt.plot(times_interp,signal_strengths[:,Receiver.TL.value,0], label='Toolik EW')
# # plt.plot(times_interp,signal_strengths[:,Receiver.TL.value,1], label='Toolik NS', linestyle='dashed')
#     # Plot for the current sample rate
#     axes[idx].plot(times_interp, signal_strengths[:,Receiver.PF.value, 0], label='Poker Flat ew')
#     axes[idx].plot(times_interp, signal_strengths[:,Receiver.PF.value, 1], label='Poker Flat ns')
#     axes[idx].set_title(f'Sample Rate: {sample_rate} Hz')
#     axes[idx].legend()
#     #print(f"Sample Rate: {sample_rate}: Correlation between theta and phi: {stats.pearsonr(positions_interp[:,Receiver.PF.value,1], positions_interp[:,Receiver.PF.value,2])[0]}")
#     # print(f"Sample rate: {sample_rate}: mean(x): {np.mean(positions_interp_cartesian[:,Receiver.PF.value,0])}, std(x): {np.std(positions_interp_cartesian[:,Receiver.PF.value,0])}")
#     # print(f"Sample rate: {sample_rate}: mean(y): {np.mean(positions_interp_cartesian[:,Receiver.PF.value,1])}, std(y): {np.std(positions_interp_cartesian[:,Receiver.PF.value,1])}")
#     # print(f"Sample rate: {sample_rate}: mean(z): {np.mean(positions_interp_cartesian[:,Receiver.PF.value,2])}, std(z): {np.std(positions_interp_cartesian[:,Receiver.PF.value,2])}")
# # Final formatting
# plt.xlabel('Time')
# for ax in axes:
#     ax.set_ylim(-200, -90)
# plt.tight_layout()
# plt.show()
