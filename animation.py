import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
def animate(data_vec, title="Unnamed Vector", reference_label="Z-axis"):
    global ax
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    max_range = np.max(np.abs(data_vec)) * 1.1
    ax.set_xlim([-max_range,max_range])
    ax.set_ylim([-max_range,max_range])
    ax.set_zlim([-max_range,max_range])

    ax.quiver(0,0,0,0,0,max_range, color='b', arrow_length_ratio = 0.1,label=reference_label)
    ax.legend()    
    #quiver = ax.quiver (0,0,0, data_vec[0,0], data_vec[0,1], data_vec[0,2], color='r', arrow_length_ratio = 0.1)
    #(axis_vec,) = ax.plot([],[],[], 'r-', linewidth=2, label=label)
    quiver = [None]
    ax.set_title(title)
    ani = animation.FuncAnimation(fig, update, frames=len(data_vec), fargs=(quiver, data_vec), interval=50, blit=False)
    plt.show()

def update(frame, quiver, data_vec):
	#axis_vec.set_data([0, data_vec[frame,0]],[0,data_vec[frame,1]])
	#axis_vec.set_3d_properties([0,data_vec[frame,2]])
    if quiver[0] is not None:
        quiver[0].remove()
    quiver[0] = ax.quiver(0,0,0,data_vec[frame,0],data_vec[frame,1],data_vec[frame,2], color='r', arrow_length_ratio = 0.1)
    return quiver[0],