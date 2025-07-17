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


