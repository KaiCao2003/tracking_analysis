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
   The script will optionally trim the CSV when `preprocess.enable` is `true`.

All plots and exported data are written to the directory specified by `output.output_dir` (default `results/`).


