# Tracking Analysis

This project provides a small pipeline for analyzing motion tracking CSV files. Configuration is managed entirely through `config.yaml`.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the analysis:
   ```bash
   python -m tracking_analysis.cli --config config.yaml
   ```
   The script will automatically trim the CSV when `preprocess.enable` is `true`.

All plots, trimmed files and statistics are written to the directory specified by `output.output_dir` (default `results/`). The trimming step also produces `summary.txt` describing the input CSV.

Each run creates a timestamped subfolder inside `output.output_dir`. The exact configuration used is saved as `config_used.txt` within that folder.

## Configuration Options

Below is an overview of the keys supported in `config.yaml`.

### `input_file`
Path to the CSV file to process.

### `interval`
`start_time` and `end_time` specify the slice of the recording in seconds. Use `null` for `end_time` to include the entire file.

### `time_markers`
List of timestamps (seconds) to highlight on the plots. Markers are drawn as small red triangles so the same event can be located across all figures.

### `groups`
List of rigid-body base names to analyse. An empty list means all available groups.

### `kinematics`
Options for smoothing the computed velocities (`smoothing`, `smoothing_window`, `smoothing_polyorder`, `smoothing_method`).

### `filtering`
Enable range filtering and define upper/lower thresholds for linear or angular speed.

### `output`
Controls which figures and exports are produced. `full_size_plots` enlarges plots to 16x10 inches. `x_limit` and `y_limit` specify the axis maxima for time-series plots (set to `null` for automatic scaling).

### `preprocess`
If `enable` is true, the CSV is trimmed before analysis and a brief `summary_file` is written.

You can also run the trimming step manually:

```bash
python -m tracking_analysis.trim <input.csv> [output.csv] --summary summary.txt
```



