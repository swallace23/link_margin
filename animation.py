# animates vectors in time. sometimes useful for debugging/sanity checking
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
def update(frame, quiver_vec, data_vec, quiver_axis, axis_vecs):
    v = data_vec[frame]
    if quiver_vec[0] is not None:
        quiver_vec[0].remove()
    quiver_vec[0] = ax.quiver(0, 0, 0, v[0], v[1], v[2], color='r', arrow_length_ratio=0.1)

    if axis_vecs is not None:
        a = axis_vecs[frame]
        if quiver_axis[0] is not None:
            quiver_axis[0].remove()
        quiver_axis[0] = ax.quiver(0, 0, 0, a[0], a[1], a[2], color='b', arrow_length_ratio=0.1)

def animate(data_vec, title="Unnamed Vector", reference_label="Z-axis", axis_vecs=None):
    global ax
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    max_range = np.max(np.abs(data_vec)) * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_title(title)

    quiver_vec = [None]
    quiver_axis = [None]

    if axis_vecs is not None:
        axis_vecs = np.asarray(axis_vecs)
        # Add legend handle for initial frame (placeholder)
        quiver_axis[0] = ax.quiver(0, 0, 0, 0, 0, 0, color='b', arrow_length_ratio=0.1, label=reference_label)
        ax.legend()

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(data_vec),
        fargs=(quiver_vec, data_vec, quiver_axis, axis_vecs),
        interval=50,
        blit=False
    )
    plt.show()
