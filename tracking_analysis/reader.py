import pandas as pd

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
