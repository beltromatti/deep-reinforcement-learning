# dqn_rl/config.py
import yaml

def load_config(config_path=None):
    """Load configuration from a YAML file or return default configuration.

    Args:
        config_path (str, optional): Path to YAML configuration file.

    Returns:
        dict: Configuration parameters.
    """
    default_config = {
        "episodes": 1000,
        "batch_size": 64,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.02,
        "epsilon_decay": 0.995,
        "memory_size": 20000,
        "learning_rate": 0.0005,
        "target_update": 10,
        "save_checkpoint_every": 50,
        "use_prioritized_replay": False,
        "per_alpha": 0.6,
        "per_beta": 0.4,
        "use_double_dqn": False,
        "max_grad_norm": 1.0,
        "checkpoint_path": "dqn_checkpoint.pth",
        "plot_path": "rewards_plot.png"
    }
    if config_path:
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
        default_config.update(user_config)
    return default_config