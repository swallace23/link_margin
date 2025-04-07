# credit to Shreya Gandhi for some of this code
import numpy as np
import pandas as pd

# Reads Nyquist trajectory file
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

# read rocket GPS file
def read_gps_file(filename):
    df = pd.read_excel(filename)
    # strip white space from column names
    df.columns = df.columns.str.strip()
    df = df[['Flight Time', 'Lat', 'Long', 'Alt']]
    df[['Alt']]=df[['Alt']]*1000
    arr = df.to_numpy()
    # last three rows have negative time, removed for safety
    arr = np.delete(arr, np.s_[len(arr)-3:len(arr)], axis=0)
    # shift times to be positive
    arr[:,0] = arr[:,0] + np.abs(arr[0,0])

    times = arr[:,0]
    lla = arr[:,1:arr.shape[1]]
    return times, lla



def get_times(traj_arrays):
    return traj_arrays["Time"]