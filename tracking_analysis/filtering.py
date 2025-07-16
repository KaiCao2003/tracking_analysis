# File: tracking_analysis/filtering.py
import pandas as pd

def filter_missing(df, start_frame, end_frame):
    """
    Identify entities whose X coordinate never changes in the given frame interval.
    Handles end_frame = float('inf') as “up through the last frame.”
    Returns a list of entity names whose Position→X is static.
    """
    # Slice the DataFrame by frame index
    if end_frame == float('inf'):
        sub = df.iloc[start_frame:]
    else:
        sub = df.iloc[start_frame:end_frame + 1]

    missing = []
    # Top-level entity names
    entities = df.columns.get_level_values(0).unique()

    for entity in entities:
        try:
            # 1) select only this entity (drops level 0)
            ent_df    = sub.xs(entity, level=0, axis=1)
            # 2) select the 'Position' measurement (now level 1 of ent_df)
            pos_block = ent_df.xs('Position', level=1, axis=1)
            # 3) drop the ID-level (first level) to leave components X,Y,Z
            pos_df    = pos_block.droplevel(0, axis=1)
            # 4) extract the X-coordinate series
            x_series  = pos_df['X']
        except KeyError:
            # no Position→X for this entity
            continue

        # Robust unique count handling, even if x_series is accidentally a
        # DataFrame due to unexpected column structure. Flatten the values to a
        # 1D array, drop NaNs, and measure the number of unique entries.
        vals = pd.unique(pd.Series(x_series.values.ravel()).dropna())
        if len(vals) <= 1:
            missing.append(entity)

    return missing