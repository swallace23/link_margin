import numpy as np
import scipy.interpolate as ip
import pandas as pd
from PyNEC import *
# for IGRF, currently not used
#import ppigrf as pp
#from datetime import date
from scipy.spatial.transform import Rotation as R
from scipy import linalg as la
import matplotlib.pyplot as plt

############################################ MATH UTILITIES ############################################
# Row wize L2 normalization
def unit_vectors(array):
    return array/np.linalg.norm(array, axis=1)[:, np.newaxis]
# Row wise dot product of 2D arrays
def dots(arr1, arr2):
    return np.einsum('ij,ij->i',arr1,arr2)

############################################ POLARIZATION LOSS FUNCTIONS ############################################
# Calculates polarization loss factor per Alex Mule's formula
def get_polarization_loss(receiver, source, separation):
    # get all the necessary unit vectors
    receiver_hat = unit_vectors(receiver)
    source_hat = unit_vectors(source)
    separation_hat = unit_vectors(separation)
    dot_rs = dots(receiver_hat,source_hat)
    dot_rsep = dots(receiver_hat,separation_hat)
    dot_ssep = dots(source_hat,separation_hat)
    # calculation
    denominator = np.sqrt(1-dot_rsep**2)*np.sqrt(1-dot_ssep**2)
    # avoid division by zero
    denominator = np.maximum(denominator,1e-6)
    ploss = (dot_rs - (dot_rsep*dot_ssep))/denominator
    return ploss

# get angles from ENU coordinates
def get_thetas(r_enu):
    return np.arctan((np.sqrt(r_enu[:,0]**2 + r_enu[:,1]**2))/r_enu[:,2])
def get_phis(r_enu):
    return np.arctan(r_enu[:,1]/r_enu[:,0])

############################################ MAG FIELD FUNCTIONS ############################################
# Get IGRF vector at LLA position in ENU coordinates. currently using uniform approximation.
# def get_mag(LLA):
#     td = date.today()
#     td = pd.Timestamp(td)
#     LLA_kilom = LLA
#     LLA_kilom[:,2] = LLA_kilom[:,2]/1000
#     #east, north, up in nano teslas
#     Be, Bn, Bu = pp.igrf(LLA_kilom[:,1], LLA_kilom[:,0], LLA_kilom[:,2], td)
#     return np.array([Be, Bn, Bu]).squeeze().T

# rotate vector about z axis over time at omega radians per second
def rotate_whip(times, omega):
    initial_vecs = np.full((len(times), 3), [1,0,0]).astype(np.float64)
    r = R.from_euler('z', omega * times, degrees=False)
    rotated_vecs = r.apply(initial_vecs)
    return rotated_vecs

# the null space of the magnetic field vector is the plane of rotation. 
# the matrix of orthonormal basis vectors is the change of basis matrix from 
# standard ENU coordinates to an orthonormal cartesian basis with the z-axis along the magnetic field vector.
def get_orth_basis_for_mag(mag_vec):
    mag_unit = mag_vec / np.linalg.norm(mag_vec)
    ns = la.null_space(mag_unit.reshape(1, -1))
    # Orthonormalize null space
    Q, _ = np.linalg.qr(ns)
    # treats the magnetic field unit vector as the z-axis/up vector
    orth_basis = np.column_stack((Q, mag_unit))
    return orth_basis

# convert spherical to ENU
def s2ENU(vec):
    r, phi, theta = vec
    east = r * np.sin(phi) * np.cos(theta)
    north = r * np.sin(phi) * np.sin(theta)
    up = r * np.cos(phi)
    return np.array([east, north, up])

# get array of unit vectors rotating about magnetic field vector to represent transmitters.
def get_transmitters(mag_vec, times, omega):
    mag_vec_ENU = s2ENU(mag_vec)
    basis = get_orth_basis_for_mag(mag_vec_ENU)
    whips_mag_frame = rotate_whip(times, omega)
    whips_ENU = whips_mag_frame @ basis.T
    return whips_ENU

############################################ INTERPOLATION FUNCTIONS ############################################
# Interpolate trajectory data to a given sample rate
def interp_time_position(times, sample_rate, rocket_pos):
    f = ip.CubicSpline(times,rocket_pos)
    times_interp = np.arange(times[0], times[-1], 1/sample_rate)
    return times_interp, f(times_interp)

############################################ ANTENNA GAIN FUNCTIONS ############################################
# Get and interpolate NEC data from Alexx' calculations
def get_rx_gain(sheet_name, thetas, phis):
    signal_data = pd.read_excel(sheet_name)

    # Match NEC angle scales to trajectory data
    signal_data['THETA']=signal_data['THETA'].abs()
    thetas = np.abs(np.degrees(thetas))
    phis = np.degrees(phis)+90

    # convert absolute loss value to avoid screwing up the interpolation
    signal_data.loc[signal_data['TOTAL']<=-900, 'TOTAL']=-20
    # convert to numpy
    signal_data = signal_data[['THETA', 'PHI', 'TOTAL']].to_numpy()

    # grid for 2D interpolation
    grid = np.column_stack((signal_data[:,0], signal_data[:,1]))
    f = ip.LinearNDInterpolator(grid, signal_data[:,2])
    gains = f(thetas, phis)
    return(gains)

# NOTE - New PyNEC installations on unix are broken. Must install version 1.7.3.4.
# Get dipole gain pattern for given frequency, length
# Documentation - https://tmolteno.github.io/necpp/index.html, in particular https://tmolteno.github.io/necpp/classnec__context.html
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
    # free space ground condition
    context.gn_card(-1,0,0,0,0,0,0,0)
    # specify radio excitation source as voltage
    context.ex_card(0, wire_tag, int(nr_segments/2), 0, 0, 0, 0, 0, 0, 0, 0)
    # specify frequency in MHz. parameter is called freq_hz in the python library for some reason, but NEC uses MHz
    context.fr_card(ifrq=0,nfrq=1,freq_hz=frequency,del_freq=0)
    # radiation parameters: normal mode in free space, output major, minor axes and total gain, power gain, no averaging, antenna at origin, normalization factor 0
    context.rp_card(calc_mode=0,n_theta=90,n_phi=180,output_format=0,normalization=5,D=0,A=0,theta0=0.0,phi0=0.0,delta_theta=1.0,delta_phi=5,radial_distance=0.0,gain_norm=0.0)
    rp = context.get_radiation_pattern(0)
    return rp.get_gain()

# Interpolate gain over trajectory
def get_tx_gain(frequency, length, thetas, phis):
    if len(thetas) != len(phis):
        raise ValueError("The lengths of thetas and phis must be equal.")
    gains = pynec_dipole_gain(frequency, length)
    result = np.zeros(len(thetas))
    theta_grid = np.linspace(0,90,90)
    phi_grid = np.linspace(0,180,180)
    # I don't remember why I used RBS here - cubic spline should also be fine 
    f = ip.RectBivariateSpline(theta_grid, phi_grid, gains)
    result = f(thetas, phis, grid=False)
    return result

############################################ SIGNAL POWER ############################################

# constants:
freq = 162.990625e6  # transmit frequency (Hz)
txPwr = 2 # Transmit power in Watts
c_speed = 3.0e8  # Speed of light in m/s

def path_loss(radius):
    return (4*np.pi*radius*freq/c_speed)**2

# Standard Friis transmission equation with polarization loss factor. 
# Polarization loss is squared because it represents electromagnetic energy projected onto receiver.
def calc_received_power(radius, gains_rx, gains_tx, ploss, unit):
    if unit != 1 and unit != 0:
        raise ValueError("Invalid unit specification. 0=dBm, 1=Watts")
    result_watts = (txPwr * np.multiply(np.multiply(gains_tx,gains_rx),ploss**2))/path_loss(radius)
     # eliminate negative powers
    result_watts[result_watts<=0]=1e-100
    if unit == 0:
        # specify float to avoid rounding
        result_watts = result_watts.astype(np.float64)
        # convert to dbm
        result_dBm = 10*np.log10(result_watts)+30
        return result_dBm
    else:     
        return result_watts
    
def showPlots(times_interp, radius, receive_enum, signals, powerUnits, startTime, endTime, thetas, power, powerSum, trajectory):
    #convert radius to km
    radius = radius/1000
    if power and powerSum and trajectory:
        fig, axs = plt.subplots(2,2)
        axs[0,0].plot(times_interp, radius)
        axs[0,0].set_title("Radius")
        axs[0,0].set_ylabel("Radius (km)")
        axs[1,0].plot(times_interp, np.abs(np.abs(thetas)-np.pi/2))
        axs[1,0].set_title("(Absolute) Elevation Angle")
        axs[1,0].set_ylabel("Angle (rad)")
        for recv in receive_enum:
            ew, ns = signals[recv]
            axs[0,1].plot(times_interp, ew, label=f"{recv.name} EW")
            axs[0,1].plot(times_interp, ns, label=f"{recv.name} NS")
        axs[0,1].set_title("Received Power")
        if powerUnits == 0:
            axs[0,1].set_ylabel("dBm")
        elif powerUnits == 1:
            axs[0,1].set_ylabel("Watts")
        else:
            raise ValueError("powerUnits must be 0 (dBm) or 1 (Watts)")
        axs[0,1].legend()
        axs[1,1].plot(times_interp, ew+ns)
        axs[1,1].set_title("Sum of Received Power")
        axs[1,1].set_ylabel("Watts")
        # plt.gca().set_ylim(bottom=-130)
        axs[1,0].set_xlabel("Time (s)")
        axs[1,1].set_xlabel("Time (s)")
        plt.xlim(startTime,endTime)
        plt.subplots_adjust(left=0.1, right=0.9,hspace=0.7, wspace=0.4)
        plt.show()
        return

    elif power and trajectory and not powerSum:
        fig, axs = plt.subplots(3)
        axs[0].plot(times_interp, radius)
        axs[0].set_title("Radius")
        axs[0].set_ylabel("Radius (km)")
        axs[1].plot(times_interp, np.abs(np.abs(thetas)-np.pi/2))
        axs[1].set_title("(Absolute) Elevation Angle")
        axs[1].set_ylabel("Angle (rad)")
        for recv in receive_enum:
            ew, ns = signals[recv]
            axs[2].plot(times_interp, ew, label=f"{recv.name} EW")
            axs[2].plot(times_interp, ns, label=f"{recv.name} NS")
        axs[2].set_title("Received Power")
        if powerUnits == 0:
            axs[0,1].set_ylabel("dBm")
        elif powerUnits == 1:
            axs[0,1].set_ylabel("Watts")
        else:
            raise ValueError("powerUnits must be 0 (dBm) or 1 (Watts)")
        axs[2].legend()
        # plt.gca().set_ylim(bottom=-130)
        axs[2].set_xlabel("Time (s)")
        plt.xlim(startTime,endTime)
        plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.5)
        plt.show()
        return

    elif power and powerSum and not trajectory:
        fig, axs = plt.subplots(2)
        for recv in receive_enum:
            ew, ns = signals[recv]
            axs[0].plot(times_interp, ew, label=f"{recv.name} EW")
            axs[0].plot(times_interp, ns, label=f"{recv.name} NS")
        axs[0].set_title("Received Power")
        axs[0].set_ylabel("Watts")
        axs[1].plot(times_interp, ew+ns)
        axs[1].set_title("Sum of Received Power")
        if powerUnits == 0:
            axs[0,1].set_ylabel("dBm")
        elif powerUnits == 1:
            axs[0,1].set_ylabel("Watts")
        else:
            raise ValueError("powerUnits must be 0 (dBm) or 1 (Watts)")
        # plt.gca().set_ylim(bottom=-130)
        axs[1].set_xlabel("Time (s)")
        plt.xlim(startTime,endTime)
        plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.5)
        plt.show()
        return
    
    elif trajectory and powerSum and not power:
        fig, axs = plt.subplots(3)
        axs[0].plot(times_interp, radius)
        axs[0].set_title("Radius")
        axs[0].set_ylabel("Radius (km)")
        axs[1].plot(times_interp, np.abs(np.abs(thetas)-np.pi/2))
        axs[1].set_title("(Absolute) Elevation Angle")
        axs[1].set_ylabel("Angle (rad)")
        for recv in receive_enum:
            ew, ns = signals[recv]
            axs[2].plot(times_interp, ew+ns)
        axs[2].set_title("Received Power")
        if powerUnits == 0:
            axs[2].set_ylabel("dBm")
        elif powerUnits == 1:
            axs[2].set_ylabel("Watts")
        else:
            raise ValueError("powerUnits must be 0 (dBm) or 1 (Watts)")
        axs[2].set_xlabel("Time (s)")
        plt.xlim(startTime,endTime)
        plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.7)
        plt.show()
        return
        
    elif power:
        for recv in receive_enum:
            ew, ns = signals[recv]
            plt.plot(times_interp, ew, label=f"{recv.name} EW")
            plt.plot(times_interp, ns, label=f"{recv.name} NS")
        plt.title("Received Power")
        if powerUnits == 0:
            plt.ylabel("dBm")
        elif powerUnits == 1:
            plt.ylabel("Watts")
        else:
            raise ValueError("powerUnits must be 0 (dBm) or 1 (Watts)")
        plt.xlabel("Time (s)")
        plt.xlim(startTime,endTime)
        plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.5)
        plt.legend()
        plt.show()
        return
    
    elif trajectory:
        fig, axs = plt.subplots(2)
        axs[0].plot(times_interp, radius)
        axs[0].set_title("Radius")
        axs[0].set_ylabel("Radius (km)")
        axs[1].plot(times_interp, np.abs(np.abs(thetas)-np.pi/2))
        axs[1].set_title("(Absolute) Elevation Angle")
        axs[1].set_ylabel("Angle (rad)")
        axs[1].set_xlabel("Time (s)")
        plt.xlim(startTime,endTime)
        plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.5)
        plt.show()
        return
    
    elif powerSum:
        for recv in receive_enum:
            ew, ns = signals[recv]
        plt.plot(times_interp, ew+ns, label=f"{recv.name} EW+NS")
        plt.title("Received Power Sum")
        if powerUnits == 0:
            plt.ylabel("dBm")
        elif powerUnits == 1:
            plt.ylabel("Watts")
        else:
            raise ValueError("powerUnits must be 0 (dBm) or 1 (Watts)")
        plt.xlabel("Time (s)")
        plt.xlim(startTime,endTime)
        plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.5)
        plt.legend()
        plt.show()
        return
    else:
        print("I had thought that going into space would be the ultimate catharsis of that connection I had been looking for between all living things—that being up there would be the next beautiful step to understanding the harmony of the universe. In the film 'Contact,' when Jodie Foster’s character goes to space and looks out into the heavens, she lets out an astonished whisper, 'They should’ve sent a poet.' I had a different experience, because I discovered that the beauty isn’t out there, it’s down here, with all of us. Leaving that behind made my connection to our tiny planet even more profound. It was among the strongest feelings of grief I have ever encountered. The contrast between the vicious coldness of space and the warm nurturing of Earth below filled me with overwhelming sadness. Every day, we are confronted with the knowledge of further destruction of Earth at our hands: the extinction of animal species, of flora and fauna . . . things that took five billion years to evolve, and suddenly we will never see them again because of the interference of mankind. It filled me with dread. My trip to space was supposed to be a celebration; instead, it felt like a funeral.")
        print("\n\n\n\n\n")
        print
        import time
        time.sleep(5)
        raise ValueError("No data :(")