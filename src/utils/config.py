"""Configuration utilities for the Mining ML project."""

import yaml
import os
from pathlib import Path


def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    config_file = project_root / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_data_path(config, data_type="raw"):
    """
    Get the path to data directory.
    
    Args:
        config (dict): Configuration dictionary
        data_type (str): Type of data ('raw', 'processed', 'external')
        
    Returns:
        Path: Path to the data directory
    """
    project_root = Path(__file__).parent.parent.parent
    
    if data_type == "raw":
        return project_root / config['data']['raw_data_path']
    elif data_type == "processed":
        return project_root / config['data']['processed_data_path']
    elif data_type == "external":
        return project_root / config['data']['external_data_path']
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def get_model_path(config):
    """
    Get the path to model directory.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        Path: Path to the model directory
    """
    project_root = Path(__file__).parent.parent.parent
    return project_root / config['model']['model_path']


def setup_logging(config):
    """
    Setup logging configuration.
    
    Args:
        config (dict): Configuration dictionary
    """
    import logging
    
    log_level = getattr(logging, config['logging']['level'])
    log_format = config['logging']['format']
    
    # Create logs directory if it doesn't exist
    project_root = Path(__file__).parent.parent.parent
    log_file = project_root / config['logging']['file']
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    print("Configuration loaded successfully!")
    print(f"Project: {config['project']['name']}")
    print(f"Version: {config['project']['version']}")
