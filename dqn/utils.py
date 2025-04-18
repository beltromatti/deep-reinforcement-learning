# dqn_rl/utils.py
import logging

def setup_logger(name, level=logging.INFO):
    """Setup a logger for the package.

    Args:
        name (str): Logger name.
        level: Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger