"""
Configuration management utilities for MobileNet recycling classification.
"""
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for loading and accessing YAML config files."""
    
    def __init__(self, config_path: str):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.architecture')
            default: Default value if key not found
            
        Returns:
            Configuration value
        
        Example:
            >>> config = Config('configs/mobilenet_config.yaml')
            >>> config.get('model.num_classes')
            4
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Access config via dictionary syntax."""
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return self.get(key) is not None
    
    @property
    def model(self) -> Dict:
        """Get model configuration."""
        return self._config.get('model', {})
    
    @property
    def data(self) -> Dict:
        """Get data configuration."""
        return self._config.get('data', {})
    
    @property
    def training(self) -> Dict:
        """Get training configuration."""
        return self._config.get('training', {})
    
    @property
    def paths(self) -> Dict:
        """Get paths configuration."""
        return self._config.get('paths', {})
    
    @property
    def class_mapping(self) -> Dict:
        """Get class mapping configuration."""
        return self._config.get('class_mapping', {})
    
    def save(self, save_path: str = None):
        """
        Save current configuration to YAML file.
        
        Args:
            save_path: Path to save configuration. If None, overwrites original.
        """
        save_path = save_path or self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
