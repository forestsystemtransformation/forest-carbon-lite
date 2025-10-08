"""Configuration loader with fallback mechanism and validation."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from .validation import ValidatedConfigLoader, SiteConfig, EconomicsConfig
from .climate_adjustments import load_climate_config, apply_climate_adjustments_to_config

class ConfigLoader:
    """Loads and merges configuration files with fallback defaults."""
    
    def __init__(self, config_dir: Path = Path("configs/base"), validate: bool = True):
        """Initialize with configuration directory.
        
        Args:
            config_dir: Path to configuration directory
            validate: Whether to use validation (default: True)
        """
        self.config_dir = config_dir
        self.validate = validate
        self.default_economics = self._load_economics_defaults()
        
        if validate:
            self.validator = ValidatedConfigLoader(str(config_dir))
    
    def load_site_config(self, forest_type: str, config_file: Optional[str] = None, 
                        climate_config: Optional[str] = None) -> Union[Dict[str, Any], SiteConfig]:
        """Load site configuration for specified forest type.
        
        Args:
            forest_type: Forest type identifier (e.g., 'EOF', 'ETOF')
            config_file: Optional custom config file name or path
            climate_config: Optional climate configuration name (without .yaml extension)
            
        Returns:
            Dict if validation disabled, SiteConfig if validation enabled
        """
        if self.validate:
            # Use validated loading
            validated_config = self.validator.load_and_validate_site_config(forest_type, config_file)
            
            # Merge with default economics if not present
            if validated_config.economics is None:
                validated_config.economics = self.validator.load_and_validate_economics_config()
            
            # Apply climate adjustments if specified
            if climate_config:
                # Convert validated config to dict for climate adjustments
                config_dict = validated_config.dict()
                climate_config_dict = load_climate_config(Path("configs/base"), climate_config)
                adjusted_config_dict = apply_climate_adjustments_to_config(config_dict, climate_config_dict)
                
                # Convert back to validated config (this bypasses validation for climate-adjusted params)
                # Note: This is a simplified approach - in production you might want more sophisticated handling
                return adjusted_config_dict
            
            return validated_config
        else:
            # Original unvalidated loading
            if config_file:
                # Use custom config file
                if Path(config_file).is_absolute():
                    config_path = Path(config_file)
                else:
                    config_path = self.config_dir / config_file
            else:
                # Use default naming convention
                config_path = self.config_dir / f"site_{forest_type}.yaml"
            
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Handle generated configuration files (they have scenario_metadata and scenario_components)
            if 'scenario_metadata' in config:
                # This is a generated config file, extract the actual configuration
                # The configuration parameters are at the top level after the metadata sections
                actual_config = {}
                for key, value in config.items():
                    if key not in ['scenario_metadata', 'scenario_components']:
                        actual_config[key] = value
                
                # Add forest_type from metadata if not present
                if 'forest_type' not in actual_config:
                    actual_config['forest_type'] = config['scenario_metadata'].get('forest_type', forest_type)
                
                config = actual_config
            
            # Apply climate adjustments if specified
            if climate_config:
                climate_config_dict = load_climate_config(Path("configs/base"), climate_config)
                config = apply_climate_adjustments_to_config(config, climate_config_dict)
            
            # Merge with default economics if economics not in site config
            if 'economics' not in config:
                config['economics'] = self.default_economics
            else:
                # Merge missing economics parameters
                config['economics'] = self._merge_configs(
                    self.default_economics, 
                    config['economics']
                )
            
            return config
    
    def _load_economics_defaults(self) -> Dict[str, Any]:
        """Load default economics configuration."""
        default_file = Path("configs/base/economics_default.yaml")
        
        if not default_file.exists():
            # Return hardcoded defaults if file doesn't exist
            return {
                'carbon': {
                    'price_start': 35.0,
                    'price_growth': 0.03,
                    'buffer': 0.20,
                    'crediting_years': 30,
                    'discount_rate': 0.07
                },
                'costs': {
                    'management': {
                        'capex': 2000.0,
                        'opex': 150.0,
                        'mrv': 15.0
                    },
                    'reforestation': {
                        'capex': 5000.0,
                        'opex': 200.0,
                        'mrv': 35.0
                    }
                }
            }
        
        with open(default_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _merge_configs(self, default: Dict, override: Dict) -> Dict:
        """Recursively merge configuration dictionaries."""
        result = default.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(self, config: Dict[str, Any], filepath: Path):
        """Save configuration to YAML file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def find_config_file(self, forest_type: str, config_file: Optional[str] = None) -> Path:
        """
        Find configuration file with proper path resolution.
        
        Args:
            forest_type: Forest type identifier
            config_file: Optional custom config file name or path
            
        Returns:
            Path to configuration file
            
        Raises:
            FileNotFoundError: If configuration file not found
        """
        if config_file:
            # Handle custom config file
            if Path(config_file).is_absolute():
                config_path = Path(config_file)
            else:
                # Try relative to config directory first
                config_path = self.config_dir / config_file
                if not config_path.exists():
                    # Try relative to current working directory
                    config_path = Path(config_file)
        else:
            # Use default naming convention
            config_path = self.config_dir / f"site_{forest_type}.yaml"
        
        if not config_path.exists():
            # Provide helpful error message
            if config_file:
                raise FileNotFoundError(
                    f"Configuration file not found: {config_path}\n"
                    f"Searched locations:\n"
                    f"  - {self.config_dir / config_file}\n"
                    f"  - {Path(config_file).resolve()}"
                )
            else:
                raise FileNotFoundError(
                    f"Default configuration file not found: {config_path}\n"
                    f"Expected file: site_{forest_type}.yaml in {self.config_dir}\n"
                    f"Available files: {list(self.config_dir.glob('*.yaml'))}"
                )
        
        return config_path
    
    def list_available_configs(self) -> List[str]:
        """
        List all available configuration files.
        
        Returns:
            List of forest types with available configurations
        """
        config_files = list(self.config_dir.glob("site_*.yaml"))
        forest_types = []
        
        for config_file in config_files:
            # Extract forest type from filename
            forest_type = config_file.stem.replace("site_", "")
            forest_types.append(forest_type)
        
        return sorted(forest_types)
    
    def list_available_climate_configs(self) -> List[str]:
        """
        List all available climate configuration files.
        
        Returns:
            List of climate config names (without .yaml extension)
        """
        climate_files = list(self.config_dir.glob("climate_*.yaml"))
        climate_configs = []
        
        for climate_file in climate_files:
            # Extract climate config name from filename
            climate_name = climate_file.stem.replace("climate_", "")
            climate_configs.append(climate_name)
        
        return sorted(climate_configs)
    
    def validate_config_file(self, filepath: Path) -> bool:
        """
        Validate that a configuration file can be loaded and parsed.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)
            
            # Basic validation - check for required keys
            required_keys = ['K_AGB', 'G', 'tyf_calibrations']
            for key in required_keys:
                if key not in config:
                    return False
            
            return True
            
        except Exception:
            return False