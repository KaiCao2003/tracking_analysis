import argparse
import os
from datetime import datetime
import numpy as np
import pandas as pd

from tracking_analysis.config import Config
from tracking_analysis.reader import load_data, preprocess_csv
from tracking_analysis.filtering import (
    filter_missing,
    filter_anomalies,
    apply_ranges,
    compute_stats,
    filter_position,
    merge_ranges,
)
from tracking_analysis.grouping import group_entities
from tracking_analysis.kinematics import (
    compute_linear_velocity,
    compute_angular_speed
)
from tracking_analysis.plotting import (
    plot_trajectory_2d,
    plot_trajectory_3d,
    plot_time_series,
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

    # Prepare run-specific output directory
    base_out = cfg.get('output', 'output_dir')
    out_dir = os.path.join(base_out, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)

    # Save the config used for this run
    with open(args.config, 'r') as f_in, \
            open(os.path.join(out_dir, 'config_used.txt'), 'w') as f_out:
        f_out.write(f_in.read())

    # Optionally preprocess the CSV
    input_path = cfg.get('input_file')
    pre_cfg = cfg.get('preprocess') or {}
    if pre_cfg.get('enable'):
        out_file = pre_cfg.get('output_file', os.path.join(out_dir, 'trimmed.csv'))
        summary_file = pre_cfg.get('summary_file', os.path.join(out_dir, 'summary.txt'))
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        preprocess_csv(
            input_path,
            out_file,
            summary_file,
        )
        # Continue with the trimmed file
        input_path = out_file

    # Load data + frame/time columns from the original file
    df, frame_col, time_col = load_data(input_path)

    # Convert time-based interval to frame indices
    start_time = cfg.get('interval', 'start_time', default=0.0)
    end_time   = cfg.get('interval', 'end_time')
    times_full = df[time_col].values
    start = int(np.searchsorted(times_full, start_time, side='left'))
    if end_time == float('inf'):
        end = float('inf')
    else:
        end = int(np.searchsorted(times_full, end_time, side='right')) - 1

    # Static/missing filtering
    missing = filter_missing(df, start, end)
    if cfg.get('output', 'export_filtered'):
        with open(os.path.join(out_dir, 'missing.txt'), 'w') as f:
            f.write('\n'.join(missing))

    # Build ID→entity groups
    groups      = group_entities(df)
    selected_ids = cfg.get('groups') or list(groups.keys())

    # Drop entities that were deemed missing/static
    selected_ids = [sid for sid in selected_ids if sid not in missing]

    # Export the filtered data subset when requested
    if cfg.get('output', 'export_filtered'):
        cols = [c for c in df.columns if c[0] in selected_ids]
        if end == float('inf'):
            filtered_df = df.loc[start:, cols]
        else:
            filtered_df = df.loc[start:end, cols]
        filtered_df.to_csv(os.path.join(out_dir, 'filtered.csv'))

    # Optionally export the marker grouping
    if cfg.get('output', 'export_marker_list'):
        with open(os.path.join(out_dir, 'markers.txt'), 'w') as f:
            for gid, info in groups.items():
                line = ",".join([gid] + info['markers'])
                f.write(line + "\n")

    # out_dir already prepared above

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

        # Skip entities lacking required measurements
        if ('Position' not in ent_df.columns.get_level_values(1)
                or 'Rotation' not in ent_df.columns.get_level_values(1)):
            continue

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
        method       = cfg.get('kinematics','smoothing_method', default='savgol')
        time_markers = cfg.get('time_markers') or []  # seconds
        filt_cfg     = cfg.get('filtering') or {}

        # Convert marker times to relative indices within this slice
        tmarkers = []
        for tm in time_markers:
            idx = int(np.searchsorted(times, tm, side='left'))
            if 0 <= idx < len(times):
                tmarkers.append(idx)

        # Compute speeds
        speed, t_v   = compute_linear_velocity(
            pos,
            times,
            smoothing=smoothing,
            window=window,
            polyorder=polyorder,
            method=method,
        )

        ang_spd, t_a = compute_angular_speed(
            quat, times, smoothing, window, polyorder
        )

        # Optional range filtering
        if filt_cfg.get('enable'):
            start_frames = start + 1
            speed_ranges = []
            ang_ranges = []
            pos_ranges = []

            if (
                filt_cfg.get('speed_lower') is not None
                or filt_cfg.get('speed_upper') is not None
            ):
                speed, speed_ranges = filter_anomalies(
                    speed,
                    start_frames,
                    filt_cfg.get('speed_lower'),
                    filt_cfg.get('speed_upper'),
                )

            if (
                filt_cfg.get('angular_speed_lower') is not None
                or filt_cfg.get('angular_speed_upper') is not None
            ):
                ang_spd, ang_ranges = filter_anomalies(
                    ang_spd,
                    start_frames,
                    filt_cfg.get('angular_speed_lower'),
                    filt_cfg.get('angular_speed_upper'),
                )

            if any(
                filt_cfg.get(f"{axis}_{b}") is not None
                for axis in ("x", "y", "z")
                for b in ("lower", "upper")
            ):
                pos, pos_ranges = filter_position(
                    pos,
                    start,
                    filt_cfg.get("x_lower"),
                    filt_cfg.get("x_upper"),
                    filt_cfg.get("y_lower"),
                    filt_cfg.get("y_upper"),
                    filt_cfg.get("z_lower"),
                    filt_cfg.get("z_upper"),
                )

            # cross apply ranges
            ang_spd = apply_ranges(ang_spd, start_frames, speed_ranges)
            speed = apply_ranges(speed, start_frames, ang_ranges)

            if pos_ranges:
                rng_conv = [(max(start_frames, s), e + 1) for s, e in pos_ranges]
                speed = apply_ranges(speed, start_frames, rng_conv)
                ang_spd = apply_ranges(ang_spd, start_frames, rng_conv)

            pos = apply_ranges(pos, start, [(s - 1, e) for s, e in speed_ranges])
            pos = apply_ranges(pos, start, [(s - 1, e) for s, e in ang_ranges])

            speed_pos_ranges = [(s - 1, e) for s, e in speed_ranges]
            ang_pos_ranges = [(s - 1, e) for s, e in ang_ranges]
            all_ranges = merge_ranges(speed_pos_ranges + ang_pos_ranges + pos_ranges)
            speed_ranges = [(s - start_frames, e - start_frames) for s, e in speed_ranges]
            ang_ranges = [(s - start_frames, e - start_frames) for s, e in ang_ranges]
            traj_ranges = [(s - start, e - start) for s, e in all_ranges]
        else:
            speed_ranges = []
            ang_ranges = []
            traj_ranges = []

        frames_v = np.arange(start + 1, start + 1 + len(speed))
        frames_a = np.arange(start + 1, start + 1 + len(ang_spd))

        # Summary statistics
        stats_v = compute_stats(speed, frames_v, t_v)
        stats_a = compute_stats(ang_spd, frames_a, t_a)

        if cfg.get('output', 'export_speed_stats'):
            with open(os.path.join(out_dir, f"{id_}_speed_stats.txt"), "w") as f:
                for k, v in stats_v.items():
                    f.write(f"{k}: {v}\n")
        if cfg.get('output', 'export_angular_stats'):
            with open(os.path.join(out_dir, f"{id_}_angular_stats.txt"), "w") as f:
                for k, v in stats_a.items():
                    f.write(f"{k}: {v}\n")

        # Plot & save
        if cfg.get('output', 'plot_trajectory_2d'):
            plot_trajectory_2d(
                pos,
                times,
                tmarkers,
                os.path.join(out_dir, f"{id_}_traj2d.svg"),
                anomalies=traj_ranges,
                full_size=cfg.get('output', 'full_size_plots', default=False),
            )
        if cfg.get('output', 'plot_trajectory_3d'):
            plot_trajectory_3d(
                pos,
                times,
                tmarkers,
                os.path.join(out_dir, f"{id_}_traj3d.svg"),
                anomalies=traj_ranges,
                full_size=cfg.get('output', 'full_size_plots', default=False),
            )
        if cfg.get('output', 'plot_linear_speed'):
            plot_time_series(
                speed,
                t_v,
                'Linear Speed',
                tmarkers,
                os.path.join(out_dir, f"{id_}_speed.svg"),
                anomalies=speed_ranges,
                full_size=cfg.get('output', 'full_size_plots', default=False),
                x_limit=cfg.get('output', 'x_limit'),
                y_limit=cfg.get('output', 'y_limit'),
            )
        if cfg.get('output', 'plot_angular_speed'):
            plot_time_series(
                ang_spd,
                t_a,
                'Angular Speed',
                tmarkers,
                os.path.join(out_dir, f"{id_}_angular.svg"),
                anomalies=ang_ranges,
                full_size=cfg.get('output', 'full_size_plots', default=False),
                x_limit=cfg.get('output', 'x_limit'),
                y_limit=cfg.get('output', 'y_limit'),
            )


        # Export raw speed data when requested
        if cfg.get('output', 'export_speed'):
            df_v = pd.DataFrame({
                'frame': frames_v,
                'time': t_v,
                'speed': speed,
            })
            df_v.to_csv(
                os.path.join(out_dir, f"{id_}_speed.csv"),
                index=False,
                float_format="%.8f",
            )
        if cfg.get('output', 'export_angular_speed'):
            df_a = pd.DataFrame({
                'frame': frames_a,
                'time': t_a,
                'angular_speed': ang_spd,
            })
            df_a.to_csv(
                os.path.join(out_dir, f"{id_}_angular_speed.csv"),
                index=False,
                float_format="%.8f",
            )

if __name__ == '__main__':
    main()
