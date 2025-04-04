import numpy as np
from scipy.interpolate import RectBivariateSpline as rbs
import scipy.interpolate as ip
import pandas as pd
from PyNEC import *
import ppigrf as pp
from datetime import date
from scipy.spatial.transform import Rotation as R
from scipy import linalg as la

############################################ MATH UTILITIES ############################################
# einsum - einstein summation convention - here, row-wise dot product
def clip_norm_dots(vec1, vec2):
     return np.clip(np.einsum('ij,ij->i',vec1, vec2),0,0.9999)
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

############################################ POLARIZATION LOSS FUNCTIONS ############################################
# Calculates polarization loss factor per Alex Mule's formula
def get_polarization_loss(receiver, source, separation):

    receiver_hat = unit_vector(receiver)
    source_hat = unit_vector(source)
    separation_hat = unit_vector(separation)
    dot_rs = clip_norm_dots(receiver_hat,source_hat)
    dot_rsep = clip_norm_dots(receiver_hat,separation_hat)
    dot_ssep = clip_norm_dots(source_hat,separation_hat)
    denominator = np.sqrt(1-dot_rsep**2)*np.sqrt(1-dot_ssep**2)
    denominator = np.maximum(denominator,1e-6)
    ploss = (dot_rs - (dot_rsep*dot_ssep))/denominator
    return ploss

def get_thetas(r_enu):
    return np.arctan((np.sqrt(r_enu[:,0]**2 + r_enu[:,1]**2))/r_enu[:,2])
def get_phis(r_enu):
    return np.arctan(r_enu[:,1]/r_enu[:,0])

############################################ MAG FIELD FUNCTIONS ############################################
# Get IGRF at LLA position in ENU coordinates
def get_mag(LLA):
    td = date.today()
    td = pd.Timestamp(td)
    LLA_kilom = LLA
    LLA_kilom[:,2] = LLA_kilom[:,2]/1000
    #east, north, up in nano teslas
    Be, Bn, Bu = pp.igrf(LLA_kilom[:,1], LLA_kilom[:,0], LLA_kilom[:,2], td)
    return np.array([Be, Bn, Bu]).squeeze().T

# ENU to ECEF rotation matrix at each trajectory point
# reference - https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
def enu_to_ecef_vec_matrix_batch(r_lat, r_lon):
    if len(r_lat) != len(r_lon):
        raise ValueError("r_lat and r_lon must have the same length")
    sin_lat, cos_lat = np.sin(r_lat), np.cos(r_lat)
    sin_lon, cos_lon = np.sin(r_lon), np.cos(r_lon)
    R = np.empty((len(r_lat), 3, 3))
    R[:,0,0] = -sin_lon
    R[:,0,1] = -cos_lat * sin_lon
    R[:,0,2] =  cos_lat * cos_lon

    R[:,1,0] =  cos_lon
    R[:,1,1] = -cos_lat * cos_lon
    R[:,1,2] =  cos_lat * sin_lon

    R[:,2,0] = 0
    R[:,2,1] =  sin_lat
    R[:,2,2] =  cos_lat
    return R

# rotate IGRF vectors from transmitter frame to receiver frame in local ENU coordinates
def translate_mag_vecs(mag_vec, r_lat, r_lon, rec_lat, rec_lon):
    R_enu2ecef = enu_to_ecef_vec_matrix_batch(r_lat, r_lon)      # shape (N, 3, 3)
    R_ecef2enu = np.transpose(enu_to_ecef_vec_matrix_batch(rec_lat, rec_lon), axes=(0, 2, 1))  # inverse is transpose - see reference

    v_ecef = np.einsum('nij,nj->ni', R_enu2ecef, mag_vec)
    v_rx_enu = np.einsum('nij,nj->ni', R_ecef2enu, v_ecef)
    return v_rx_enu

def align_mag(r_lla, rec_lat, rec_lon):
    B = get_mag(r_lla)
    # convert to receiver-centered ENU
    B = translate_mag_vecs(B, r_lla[:,0], r_lla[:,1], rec_lat, rec_lon)
    # normalize. where=norms!=0 prevents division by zero
    norms = np.linalg.norm(B, axis=1, keepdims=True)
    B_hat = np.divide(B, norms, where=norms != 0)
    return B_hat

def rotate_whip(times, omega):
    initial_vecs = np.full((len(times), 3), [1,0,0]).astype(np.float64)
    r = R.from_euler('z', omega * times, degrees=False)
    rotated_vecs = r.apply(initial_vecs)
    return rotated_vecs


def get_orth_basis_for_mag(mag_vec):
    mag_unit = mag_vec / np.linalg.norm(mag_vec)
    ns = la.null_space(mag_unit.reshape(1, -1))
    # Orthonormalize null space
    Q, _ = np.linalg.qr(ns)
    orth_basis = np.column_stack((Q, mag_unit))
    return orth_basis

def s2ENU(vec):
    """Convert spherical coordinates to ENU."""
    r, phi, theta = vec
    east = r * np.sin(phi) * np.cos(theta)
    north = r * np.sin(phi) * np.sin(theta)
    up = r * np.cos(phi)
    return np.array([east, north, up])


def get_transmitters(mag_vec, times, omega):
    mag_vec_ENU = s2ENU(mag_vec)
    basis = get_orth_basis_for_mag(mag_vec_ENU)
    whips_mag_frame = rotate_whip(times, omega)
    whips_ENU = whips_mag_frame @ basis.T
    return whips_ENU

############################################ INTERPOLATION FUNCTIONS ############################################
def interp_time_position(times, sample_rate, rocket_pos):
    f = ip.CubicSpline(times,rocket_pos)
    times_interp = np.arange(times[0], times[-1], 1/sample_rate)
    return times_interp, f(times_interp)

def spin(times, aligned_rocket, angular_frequency_hz):
    z_axes = aligned_rocket/np.linalg.norm(aligned_rocket, axis=1, keepdims=True)
    # get the plane of whip rotation (orthogonal complement to magnetic field vector)
    omega = 2*np.pi*angular_frequency_hz
    vectors = np.zeros((len(times),3))  
    for i in range(len(times)):
        v = z_axes[i]
        basis = la.null_space(v.reshape(1, -1))  # shape (3,2)
        u1, u2 = basis[:, 0], basis[:, 1]
        c, s = np.cos(omega * times[i]), np.sin(omega * times[i])
        vectors[i] = c * u1 + s * u2

    return vectors
############################################ ANTENNA GAIN FUNCTIONS ############################################
# Get NEC data from excel spreadsheet
def get_rx_gain(sheet_name, thetas, phis):
    signal_data = pd.read_excel(sheet_name)
    # Match angle scales
    signal_data['THETA']=signal_data['THETA'].abs()
    thetas = np.abs(np.degrees(thetas))
    phis = np.degrees(phis)+90
    # convert complete loss value to avoid screwing up the interpolation
    signal_data.loc[signal_data['TOTAL']<=-900, 'TOTAL']=-20
    signal_data = signal_data[['THETA', 'PHI', 'TOTAL']].to_numpy()
    grid = np.column_stack((signal_data[:,0], signal_data[:,1]))
    f = ip.LinearNDInterpolator(grid, signal_data[:,2])
    gains = f(thetas, phis)
    return(gains)

# NOTE - New PyNEC installations on unix are broken. Must install version 1.7.3.4.
# Get dipole gain pattern for given frequency, length
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

# Interpolate gain over trajectory
def get_tx_gain(frequency, length, thetas, phis):
    if len(thetas) != len(phis):
        raise ValueError("The lengths of thetas and phis must be equal.")
    gains = pynec_dipole_gain(frequency, length)
    result = np.zeros(len(thetas))
    theta_grid = np.linspace(0,90,90)
    phi_grid = np.linspace(0,180,180)
    interp = rbs(theta_grid, phi_grid, gains)
    result = interp(thetas, phis, grid=False)
    return result

############################################ SIGNAL POWER ############################################

freq = 162.990625e6  # transmit frequency (Hz)
#freq = 150e6
Gtx = 2.1  # dBi for transmitter gain
txPwr = 2 # Transmit power in Watts
Bn = 20e3  # Bandwidth
NFrx_dB = 0.5  # Noise figure in dB with a pre-amp
NFrx = 10.0 ** (NFrx_dB / 10)  # Convert to dimensionless quantity
c_speed = 3.0e8  # Speed of light in m/s
kB = 1.38e-23  # Boltzmann constant

def path_loss(radius):
    return (4*np.pi*radius*freq/c_speed)**2

def calc_received_power(radius, gains_rx, gains_tx, ploss):
     result_watts = (txPwr * np.multiply(np.multiply(gains_tx,gains_rx),ploss**2))/path_loss(radius)
     result_watts[result_watts<=0]=1e-100
     result_watts = result_watts.astype(np.float64)
     result_dBm = 10*np.log10(result_watts)+30
     return result_dBm

"""
# align unit vector a onto unit vector b
# formula from https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
def align(a,b):
    v= np.cross(a,b)
    s = np.linalg.norm(v)
    c = np.dot(a,b)
    skew = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    I= np.eye(3)
    R = I + skew + np.dot(skew,skew) * (1-c)/(s**2)
    return np.matmul(R,a)

"""