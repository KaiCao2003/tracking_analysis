import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_trajectory_2d(pos, times, time_markers, out_path):
    fig, ax = plt.subplots()
    sc = ax.scatter(pos[:,0], pos[:,1], c=times, cmap='rainbow')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Time (s)')
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
    sc  = ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=times, cmap='rainbow')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Time (s)')
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