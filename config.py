import yaml
import json
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Dictionary containing configuration
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save the config file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Configuration saved to {save_path}")


def merge_args_with_config(args, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge command-line arguments with config file, with CLI args taking precedence.

    Args:
        args: argparse.Namespace object
        config: Configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    # Convert args to dict
    args_dict = vars(args)

    # Override config with non-None CLI arguments
    if 'dataset_root' in args_dict and args_dict['dataset_root']:
        config['dataset']['root'] = args_dict['dataset_root']

    if 'train_limit' in args_dict and args_dict['train_limit']:
        config['dataset']['train_limit'] = args_dict['train_limit']

    if 'test_limit' in args_dict and args_dict['test_limit']:
        config['dataset']['test_limit'] = args_dict['test_limit']

    if 'train_partitions' in args_dict and args_dict['train_partitions']:
        config['dataset']['train_partitions'] = args_dict['train_partitions']

    if 'test_partitions' in args_dict and args_dict['test_partitions']:
        config['dataset']['test_partitions'] = args_dict['test_partitions']

    if 'image_size' in args_dict and args_dict['image_size']:
        config['dataset']['image_size'] = args_dict['image_size']

    # CNN parameters
    if 'cnn_epochs' in args_dict and args_dict['cnn_epochs']:
        config['cnn']['epochs'] = args_dict['cnn_epochs']

    if 'batch_size' in args_dict and args_dict['batch_size']:
        config['cnn']['batch_size'] = args_dict['batch_size']

    if 'learning_rate' in args_dict and args_dict['learning_rate']:
        config['cnn']['learning_rate'] = args_dict['learning_rate']

    if 'early_stopping_patience' in args_dict and args_dict['early_stopping_patience']:
        config['cnn']['early_stopping_patience'] = args_dict['early_stopping_patience']

    if 'checkpoint_dir' in args_dict and args_dict['checkpoint_dir']:
        config['experiment']['checkpoint_dir'] = args_dict['checkpoint_dir']

    # Experiment flags
    if 'skip_learning_curves' in args_dict:
        config['experiment']['skip_learning_curves'] = args_dict['skip_learning_curves']

    if 'skip_lbp' in args_dict:
        config['experiment']['skip_lbp'] = args_dict['skip_lbp']


    if 'skip_hog' in args_dict:
        config['experiment']['skip_hog'] = args_dict['skip_hog']

    if 'skip_cnn' in args_dict:
        config['experiment']['skip_cnn'] = args_dict['skip_cnn']

    if 'skip_resnet' in args_dict:
        config['experiment']['skip_resnet'] = args_dict['skip_resnet']

    # Classical method flags
    if 'skip_zernike' in args_dict:
        config['experiment']['skip_zernike'] = args_dict['skip_zernike']

    if 'skip_projection' in args_dict:
        config['experiment']['skip_projection'] = args_dict['skip_projection']

    # Preprocessing parameters
    if 'preprocessing' in args_dict and args_dict['preprocessing'] is not None:
        config['preprocessing']['enabled'] = args_dict['preprocessing']

    if 'threshold_method' in args_dict and args_dict['threshold_method']:
        config['preprocessing']['threshold_method'] = args_dict['threshold_method']

    if 'no_deskew' in args_dict and args_dict['no_deskew']:
        config['preprocessing']['deskew'] = False

    # Classical method hyperparameters
    if 'knn_neighbors' in args_dict and args_dict['knn_neighbors']:
        config['projection']['n_neighbors'] = args_dict['knn_neighbors']

    if 'zernike_degree' in args_dict and args_dict['zernike_degree']:
        config['zernike']['degree'] = args_dict['zernike_degree']

    # Augmentation parameters
    if 'augmentation_rotation' in args_dict and args_dict['augmentation_rotation']:
        config['augmentation']['rotation_range'] = args_dict['augmentation_rotation']

    if 'augmentation_elastic' in args_dict and args_dict['augmentation_elastic']:
        config['augmentation']['elastic_alpha'] = 34  # Enable elastic distortion

    return config


def print_config(config: Dict[str, Any]):
    """Print configuration in a readable format."""
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    print("="*60)
