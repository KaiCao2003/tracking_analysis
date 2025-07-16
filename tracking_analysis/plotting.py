import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_trajectory_2d(pos, times, time_markers, out_path):
    fig, ax = plt.subplots()
    # Plot trajectory as a continuous colored line
    segments = [ [pos[i], pos[i+1]] for i in range(len(pos)-1) ]
    lc = LineCollection(segments, array=times, cmap='rainbow', linewidth=2)
    ax.add_collection(lc)
    # Add colorbar
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label('Time (s)')
    ax.autoscale()
    # Add markers on the trajectory
    for tm in time_markers:
        if 0 <= tm < len(times):
            ax.scatter(pos[tm,0], pos[tm,1], marker='v', s=60, edgecolor='black')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Trajectory')
    fig.savefig(out_path)
    plt.close(fig)

def plot_trajectory_3d(pos, times, time_markers, out_path):
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    # Plot trajectory as a continuous colored 3D line
    segments3d = [
        [(pos[i,0], pos[i,1], pos[i,2]), (pos[i+1,0], pos[i+1,1], pos[i+1,2])]
        for i in range(len(pos)-1)
    ]
    lc = Line3DCollection(segments3d, array=times, cmap='rainbow', linewidth=2)
    ax.add_collection(lc)
    # Add colorbar
    cbar = fig.colorbar(lc, ax=ax, pad=0.1)
    cbar.set_label('Time (s)')
    # Autoscale axes to data limits
    ax.auto_scale_xyz(pos[:,0], pos[:,1], pos[:,2])
    for tm in time_markers:
        if 0 <= tm < len(times):
            ax.scatter(
                pos[tm,0], pos[tm,1], pos[tm,2],
                marker='v', s=60, edgecolor='black'
            )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory')
    fig.savefig(out_path)
    plt.close(fig)

def plot_time_series(values, times, ylabel, time_markers, out_path):
    fig, ax = plt.subplots()
    ax.plot(times, values)
    for tm in time_markers:
        if 0 <= tm < len(times):
            ax.axvline(x=times[tm], linestyle='--', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs Time")
    fig.savefig(out_path)
    plt.close(fig)