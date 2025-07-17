import pandas as pd

def preprocess_csv(in_path, out_path, summary_path):
    """Trim unused header/data lines and save a summary."""
    lines_to_remove = {1, 2, 3, 6}
    with open(in_path, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()

    # Save removed lines as summary
    with open(summary_path, 'w', encoding='utf-8') as fout:
        for i, line in enumerate(lines, start=1):
            if i in lines_to_remove:
                fout.write(line)

    with open(out_path, 'w', encoding='utf-8') as fout:
        for i, line in enumerate(lines, start=1):
            if i not in lines_to_remove:
                fout.write(line)

    # Load the trimmed file to compute frequency
    df, _, time_col = load_data(out_path)
    times = df[time_col].values
    if len(times) > 1:
        freq = 1.0 / float(pd.Series(times).diff().dropna().mean())
        with open(summary_path, 'a', encoding='utf-8') as fout:
            fout.write(f"Estimated frequency: {freq:.3f} Hz\n")

def load_data(filepath):
    """
    Load a tracking CSV with a 4-row header into a DataFrame.

    Returns:
        df          – DataFrame with a 4-level MultiIndex on columns
        frame_col   – tuple for the “Frame” column
        time_col    – tuple for the “Time (Seconds)” column
    """
    df = pd.read_csv(filepath, header=[0,1,2,3])

    # locate the frame & time columns by their 4th-level name
    frame_col = next(col for col in df.columns if col[3] == 'Frame')
    time_col  = next(col for col in df.columns if col[3] == 'Time (Seconds)')

    return df, frame_col, time_col
