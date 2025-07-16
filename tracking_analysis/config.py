import yaml
import os

class Config:
    """Loads and provides access to YAML configuration."""
    def __init__(self, path="config.yaml"):
        with open(path, 'r') as f:
            self._cfg = yaml.safe_load(f)

        # Interpret `null` end_frame as “all frames”
        if self._cfg['interval'].get('end_frame') is None:
            self._cfg['interval']['end_frame'] = float('inf')

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