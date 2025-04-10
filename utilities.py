import numpy as np
import scipy.interpolate as ip
import pandas as pd
from PyNEC import *
# for IGRF, currently not used
#import ppigrf as pp
#from datetime import date
from scipy.spatial.transform import Rotation as R
from scipy import linalg as la

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
def calc_received_power(radius, gains_rx, gains_tx, ploss):
     result_watts = (txPwr * np.multiply(np.multiply(gains_tx,gains_rx),ploss**2))/path_loss(radius)
     # eliminate negative powers
     result_watts[result_watts<=0]=1e-100
     # specify float to avoid rounding
     result_watts = result_watts.astype(np.float64)
     # convert to dbm
     result_dBm = 10*np.log10(result_watts)+30
     return result_dBm