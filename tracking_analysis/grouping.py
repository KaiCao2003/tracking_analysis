def group_entities(df):
    """
    Build a mapping of:
        ID -> {'rigid_body': ID, 'markers': [ID:Marker 001, …]}
    Only includes entities that actually have Position data.
    """
    all_ids = df.columns.get_level_values(0).unique()

    # Keep only base IDs (no colon) that have a Position channel
    data_ids = [
        e for e in all_ids
        if ':' not in e and any(col[0] == e and col[2] == 'Position' for col in df.columns)
    ]

    groups = {}
    for id_ in data_ids:
        # find any markers named "ID:Marker …"
        markers = [e2 for e2 in all_ids if e2.startswith(f"{id_}:")]
        groups[id_] = {
            'rigid_body': id_,
            'markers': markers
        }
    return groups
