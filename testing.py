import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def clip_norm_dots(vec1, vec2):
     return np.clip(np.dot(vec1, vec2),0,0.9999)

def get_polarization_loss(receiver, source, separation, times):
	receiver_hat = receiver / np.linalg.norm(receiver)
	source_hat = source / np.linalg.norm(source)
	separation_hat = separation / np.linalg.norm(separation)
	return (np.dot(receiver_hat,source_hat)-(np.dot(receiver_hat,separation_hat)*np.dot(source_hat,separation_hat)))/(np.sqrt(1-clip_norm_dots(receiver_hat,separation_hat)**2)*np.sqrt(1-clip_norm_dots(source_hat,separation_hat)**2))

trajectoryrt = np.genfromtxt("Traj_Right.txt", skip_header=1, dtype=float)

# Read the header separately
with open('Traj_Right.txt', 'r') as file:
    headers = file.readline().strip().split()

# Create arrays for each column using the header titles
traj_arraysrt = {header: trajectoryrt[:, i] for i, header in enumerate(headers)}
for title, array in traj_arraysrt.items():
    title = title + "_rttraj"

lat_pf = 65.1192
long_pf = -147.43

# Constants
R_EARTH = 6371000  # Earth radius in meters

# Convert spherical to Cartesian coordinates
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



# Initialize arrays
r_pf = np.zeros((len(traj_arraysrt["Time"]),3))
times = traj_arraysrt["Time"]
# Process trajectory data
for i, time in enumerate(traj_arraysrt["Time"]):
	latitude = traj_arraysrt["Latgd"][i]
	longitude = traj_arraysrt["Long"][i]
	altitude = traj_arraysrt["Altkm"][i] * 1000  # Convert altitude to meters
	time = traj_arraysrt["Time"][i]
	
	# Poker Flat as origin
	rpf, tpf, ppf = translate_point_spherical(
	    R_EARTH, np.pi/2 - np.radians(lat_pf), np.radians(long_pf),
	    R_EARTH + altitude, np.pi/2 - np.radians(latitude), np.radians(longitude)
	)
	r_pf[i][0] = rpf
	r_pf[i][1] = (np.degrees(np.arccos(altitude/rpf)))
	ppf = np.degrees(ppf)
	if ppf < 0:
		ppf += 180
	
	r_pf[i][2] = (ppf)

l1 = np.full((len(times),3),[1,0,0])
l2 = np.full((len(times),3),[0,1,0])
losses = np.zeros(len(times))
for i in range(len(times)):
	l = get_polarization_loss(l1[i], l2[i], r_pf[i], times[i])
	losses[i] = l

plt.plot(times, losses**2)
plt.show()