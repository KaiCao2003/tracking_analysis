import argparse
import os

from tracking_analysis.config import Config
from tracking_analysis.reader import load_data
from tracking_analysis.filtering import filter_missing
from tracking_analysis.grouping import group_entities
from tracking_analysis.kinematics import (
    compute_linear_velocity,
    compute_angular_speed
)
from tracking_analysis.plotting import (
    plot_trajectory_2d,
    plot_trajectory_3d,
    plot_time_series
)

def main():
    parser = argparse.ArgumentParser(description="Tracking analysis pipeline")
    parser.add_argument(
        '-c', '--config',
        default='config.yaml',
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    # Load configuration
    cfg = Config(args.config)

    # Load data + frame/time columns
    df, frame_col, time_col = load_data(cfg.get('input_file'))

    # Interval selection (frames)
    start = cfg.get('interval', 'start_frame')
    end   = cfg.get('interval', 'end_frame')

    # Static/missing filtering
    missing = filter_missing(df, start, end)
    if cfg.get('output', 'export_filtered'):
        os.makedirs(cfg.get('output','output_dir'), exist_ok=True)
        with open(os.path.join(cfg.get('output','output_dir'),
                               'missing.txt'), 'w') as f:
            f.write('\n'.join(missing))

    # Build ID→entity groups
    groups      = group_entities(df)
    selected_ids = cfg.get('groups') or list(groups.keys())

    # Prepare output directory
    out_dir = cfg.get('output','output_dir')
    os.makedirs(out_dir, exist_ok=True)

    # Full timestamp array
    times_full = df[time_col].values

    for id_ in selected_ids:
        if id_ not in groups:
            continue

        # Slice by frame indices
        if end == float('inf'):
            sub   = df.iloc[start:]
            times = times_full[start:]
        else:
            sub   = df.iloc[start:end+1]
            times = times_full[start:end+1]

        # --- extract data for this entity (drops level 0) ---
        ent_df = sub.xs(id_, level=0, axis=1)

        # --- Position (X,Y,Z) extraction ---
        pos_block = ent_df.xs('Position', level=1, axis=1)
        pos_df    = pos_block.droplevel(0, axis=1)
        pos       = pos_df[['X','Y','Z']].values

        # --- Rotation (X,Y,Z,W) extraction ---
        rot_block = ent_df.xs('Rotation', level=1, axis=1)
        rot_df    = rot_block.droplevel(0, axis=1)
        quat      = rot_df[['X','Y','Z','W']].values

        # Kinematics settings
        smoothing    = cfg.get('kinematics','smoothing')
        window       = cfg.get('kinematics','smoothing_window')
        polyorder    = cfg.get('kinematics','smoothing_polyorder')
        time_markers = cfg.get('time_markers') or []

        # Compute speeds
        speed, t_v   = compute_linear_velocity(pos, times,
                                               smoothing, window, polyorder)
        ang_spd, t_a = compute_angular_speed(quat, times,
                                             smoothing, window, polyorder)

        # Plot & save
        if cfg.get('output','plots'):
            plot_trajectory_2d(
                pos, times, time_markers,
                os.path.join(out_dir, f"{id_}_traj2d.png")
            )
            plot_trajectory_3d(
                pos, times, time_markers,
                os.path.join(out_dir, f"{id_}_traj3d.png")
            )
            plot_time_series(
                speed, t_v, 'Linear Speed', time_markers,
                os.path.join(out_dir, f"{id_}_speed.png")
            )
            plot_time_series(
                ang_spd, t_a, 'Angular Speed', time_markers,
                os.path.join(out_dir, f"{id_}_angular.png")
            )

if __name__ == '__main__':
    main()