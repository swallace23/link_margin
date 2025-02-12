# credit to Shreya Gandhi for some of this code
import numpy as np

R_EARTH = 6371000  # Earth radius in meters

def read_traj_data(filename):
    trajectoryrt = np.genfromtxt(filename, skip_header=1, dtype=float)

    # Read the header separately
    with open('Traj_Right.txt', 'r') as file:
        headers = file.readline().strip().split()

    # Create arrays for each column using the header titles
    traj_arraysrt = {header: trajectoryrt[:, i] for i, header in enumerate(headers)}
    for title, array in traj_arraysrt.items():
        title = title + "_rttraj"
    return traj_arraysrt

def get_times(traj_arrays):
    return traj_arrays["Time"]

def spherical_to_cartesian(r, theta, phi):
	x = r * np.sin(theta) * np.cos(phi)
	y = r * np.sin(theta) * np.sin(phi)
	z = r * np.cos(theta)
	return np.array([x, y, z])

# Convert Cartesian to spherical coordinates
def cartesian_to_spherical(x, y, z):
	r = np.sqrt(x**2 + y**2 + z**2)
	theta = np.arcsin(z / r)  # Polar angle
	phi = np.arctan2(y, x)    # Azimuthal angle
	return np.array([r, theta, phi])

# Translate a point in spherical coordinates relative to a reference point
def translate_point_spherical(r1, theta1, phi1, r2, theta2, phi2):
	# Convert reference point and satellite point to Cartesian
	p1_cartesian = spherical_to_cartesian(r1, theta1, phi1)
	p2_cartesian = spherical_to_cartesian(r2, theta2, phi2)
	# Calculate the translated Cartesian coordinates
	translated_cartesian = p2_cartesian - p1_cartesian
	# Convert back to spherical coordinates
	return cartesian_to_spherical(*translated_cartesian)

def get_spherical_position(receiver_lat, receiver_lon, traj_arrays):
    pos_vec = np.zeros((len(traj_arrays["Time"]),3))
    times = traj_arrays["Time"]
# Process trajectory data
    for i, time in enumerate(traj_arrays["Time"]):
        latitude = traj_arrays["Latgd"][i]
        longitude = traj_arrays["Long"][i]
        altitude = traj_arrays["Altkm"][i] * 1000  # Convert altitude to meters
        time = traj_arrays["Time"][i]
        
        # Poker Flat as origin
        r, theta, phi = translate_point_spherical(
            R_EARTH, np.pi/2 - np.radians(receiver_lat), np.radians(receiver_lon),
            R_EARTH + altitude, np.pi/2 - np.radians(latitude), np.radians(longitude)
        )
        pos_vec[i][0] = r
        pos_vec[i][1] = (np.degrees(np.arccos(altitude/r)))
        phi = np.degrees(phi)
        if phi < 0:
            phi += 180
        
        pos_vec[i][2] = (phi)
    return pos_vec