import yaml
import os

class Config:
    """Loads and provides access to YAML configuration."""
    def __init__(self, path="config.yaml"):
        with open(path, 'r') as f:
            self._cfg = yaml.safe_load(f)

        # Normalize time_markers to a list of floats
        tmarkers = self._cfg.get('time_markers')
        if isinstance(tmarkers, str):
            try:
                self._cfg['time_markers'] = [float(t) for t in tmarkers.split(',') if t]
            except ValueError:
                self._cfg['time_markers'] = []

        # Interpret `null` end_time as “all data”
        if self._cfg['interval'].get('end_time') is None:
            self._cfg['interval']['end_time'] = float('inf')

        # Make output_dir absolute
        out = self._cfg['output'].get('output_dir', './results')
        self._cfg['output']['output_dir'] = os.path.abspath(out)

    def get(self, *keys, default=None):
        """Retrieve nested config values: cfg.get('output','plots')"""
        node = self._cfg
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node if node is not None else default
