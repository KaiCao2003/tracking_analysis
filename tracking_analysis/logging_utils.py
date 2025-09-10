import logging
import os


def setup_logging(out_dir: str, level: str = 'INFO', filename: str = 'run.log') -> logging.Logger:
    """Configure root logger to write to console and a run-specific log file."""
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, filename)
    level_num = getattr(logging, str(level).upper(), logging.INFO)

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    logger = logging.getLogger('tracking_analysis')
    logger.setLevel(level_num)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
