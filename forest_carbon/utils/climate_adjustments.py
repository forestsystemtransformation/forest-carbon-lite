"""Climate adjustment functions for forest carbon modeling.

This module provides functions to adjust forest parameters based on climate change
scenarios, including temperature increases and rainfall reductions.
"""

from typing import Dict, Any, Optional
import yaml
import copy
from pathlib import Path


def calculate_climate_dsev(baseline_dsev: float, temp_increase: float, rainfall_reduction_mm: float) -> float:
    """
    Calculate disturbance severity adjustment for climate change.
    
    Args:
        baseline_dsev: Baseline disturbance severity (0-1)
        temp_increase: Temperature increase in degrees Celsius
        rainfall_reduction_mm: Rainfall reduction in mm
        
    Returns:
        Adjusted disturbance severity (0-1)
    """
    # Temperature effect (2% increase per degree)
    temp_effect = temp_increase * 0.02
    
    # Rainfall effect (1.2% increase per 100mm reduction)  
    rain_effect = (rainfall_reduction_mm / 100) * 0.012
    
    # Combined disturbance severity
    climate_dsev = baseline_dsev + temp_effect + rain_effect
    
    # Apply reasonable bounds
    climate_dsev = max(0.02, min(0.35, climate_dsev))  # 2% min, 35% max
    
    return climate_dsev


def adjust_climate_parameters(baseline_params: Dict[str, float], 
                            temp_increase: float, 
                            rainfall_reduction_mm: float) -> Dict[str, float]:
    """
    Comprehensive climate parameter adjustment.
    
    Args:
        baseline_params: dict with 'fpi', 'mortality', 'pdist', 'dsev'
        temp_increase: degrees C
        rainfall_reduction_mm: mm less rainfall
    
    Returns:
        dict with adjusted parameters
    """
    
    # FPI adjustment (multiplicative)
    temp_factor = 1.0 - (temp_increase * 0.10)
    rain_factor = 1.0 - (rainfall_reduction_mm / 100 * 0.08)
    adjusted_fpi = baseline_params['fpi'] * temp_factor * rain_factor
    adjusted_fpi = max(0.4, min(1.2, adjusted_fpi))
    
    # Mortality adjustment (additive)
    temp_mort = temp_increase * 0.012
    rain_mort = (rainfall_reduction_mm / 100) * 0.008
    adjusted_mortality = baseline_params['mortality'] + temp_mort + rain_mort
    adjusted_mortality = max(0.005, min(0.080, adjusted_mortality))
    
    # Disturbance probability adjustment (additive)
    temp_pdist = temp_increase * 0.03
    rain_pdist = (rainfall_reduction_mm / 100) * 0.02
    adjusted_pdist = baseline_params['pdist'] + temp_pdist + rain_pdist
    adjusted_pdist = max(0.01, min(0.25, adjusted_pdist))
    
    # Disturbance severity adjustment (additive)
    temp_dsev = temp_increase * 0.02
    rain_dsev = (rainfall_reduction_mm / 100) * 0.012
    adjusted_dsev = baseline_params['dsev'] + temp_dsev + rain_dsev
    adjusted_dsev = max(0.02, min(0.35, adjusted_dsev))
    
    return {
        'fpi': adjusted_fpi,
        'mortality': adjusted_mortality,
        'pdist': adjusted_pdist,
        'dsev': adjusted_dsev
    }


def load_climate_config(config_dir: Path, climate_name: str) -> Dict[str, Any]:
    """
    Load climate configuration from YAML file.
    
    Args:
        config_dir: Path to config directory
        climate_name: Name of climate config (without .yaml extension)
        
    Returns:
        Climate configuration dictionary
        
    Raises:
        FileNotFoundError: If climate config file doesn't exist
    """
    # Handle "normal" climate - no adjustments
    if climate_name in ['normal', 'default']:
        return {
            'temperature_increase': 0.0,
            'rainfall_reduction_mm': 0.0,
            'climate_sensitivity': {
                'fpi_temperature_factor_per_degree': 0.0,
                'fpi_rainfall_factor_per_100mm': 0.0,
                'mortality_temperature_per_degree': 0.0,
                'mortality_rainfall_per_100mm': 0.0,
                'pdist_temperature_per_degree': 0.0,
                'pdist_rainfall_per_100mm': 0.0,
                'dsev_temperature_per_degree': 0.0,
                'dsev_rainfall_per_100mm': 0.0
            },
            'parameter_bounds': {
                'fpi_min': 0.40,
                'fpi_max': 1.20,
                'mortality_min': 0.005,
                'mortality_max': 0.080,
                'pdist_min': 0.01,
                'pdist_max': 0.25,
                'dsev_min': 0.02,
                'dsev_max': 0.35
            }
        }
    
    # Try with climate_ prefix first, then without
    climate_file = config_dir / f"climate_{climate_name}.yaml"
    
    if not climate_file.exists():
        climate_file = config_dir / f"{climate_name}.yaml"
    
    if not climate_file.exists():
        raise FileNotFoundError(f"Climate config file not found. Tried: climate_{climate_name}.yaml and {climate_name}.yaml")
    
    with open(climate_file, 'r') as f:
        return yaml.safe_load(f)


def apply_climate_adjustments_to_config(baseline_config: Dict[str, Any], 
                                      climate_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply climate adjustments to a baseline site configuration.
    
    Args:
        baseline_config: Original site configuration
        climate_config: Climate adjustment configuration
        
    Returns:
        Modified configuration with climate adjustments applied
    """
    # Create a deep copy to avoid modifying the original
    adjusted_config = copy.deepcopy(baseline_config)
    
    # Extract climate parameters (support both naming conventions)
    temp_increase = climate_config.get('temperature_increase', climate_config.get('temperature_change', 0.0))
    rainfall_reduction = climate_config.get('rainfall_reduction_mm', 0.0)
    
    # Get climate sensitivity parameters if available
    sensitivity = climate_config.get('climate_sensitivity', {})
    
    # Adjust FPI ratios for each scenario
    if 'fpi_ratios' in adjusted_config:
        fpi_ratios = adjusted_config['fpi_ratios']
        
        # Apply climate adjustments to each FPI ratio
        for scenario in ['baseline', 'management', 'reforestation']:
            if scenario in fpi_ratios:
                baseline_fpi = fpi_ratios[scenario]
                
                # Handle different FPI structures
                if isinstance(baseline_fpi, dict):
                    # Complex structure with default and year-specific values
                    if 'default' in baseline_fpi:
                        baseline_value = baseline_fpi['default']
                        
                        # Skip if baseline value is None
                        if baseline_value is not None:
                            # Use climate sensitivity parameters if available
                            if sensitivity:
                                # Get scenario-specific sensitivity parameters
                                temp_sensitivity = sensitivity.get('fpi_temperature_factor_per_degree', {})
                                rain_sensitivity = sensitivity.get('fpi_rainfall_factor_per_100mm', {})
                                
                                # Use scenario-specific values if available, otherwise fall back to defaults
                                if isinstance(temp_sensitivity, dict):
                                    temp_factor = 1.0 + (temp_increase * temp_sensitivity.get(scenario, -0.10))
                                else:
                                    temp_factor = 1.0 + (temp_increase * temp_sensitivity)
                                    
                                if isinstance(rain_sensitivity, dict):
                                    rain_factor = 1.0 + (rainfall_reduction / 100 * rain_sensitivity.get(scenario, -0.08))
                                else:
                                    rain_factor = 1.0 + (rainfall_reduction / 100 * rain_sensitivity)
                                
                                adjusted_fpi = baseline_value * temp_factor * rain_factor
                                
                                # Apply bounds
                                fpi_min = climate_config.get('parameter_bounds', {}).get('fpi_min', 0.40)
                                fpi_max = climate_config.get('parameter_bounds', {}).get('fpi_max', 1.20)
                                adjusted_fpi = max(fpi_min, min(fpi_max, adjusted_fpi))
                            else:
                                # Use default adjustment function
                                adjusted_fpi = adjust_climate_parameters(
                                    {'fpi': baseline_value}, temp_increase, rainfall_reduction
                                )['fpi']
                            
                            # Update the default value
                            fpi_ratios[scenario]['default'] = adjusted_fpi
                else:
                    # Simple numeric value
                    baseline_value = baseline_fpi
                    
                    # Use climate sensitivity parameters if available
                    if sensitivity:
                        # Get scenario-specific sensitivity parameters
                        temp_sensitivity = sensitivity.get('fpi_temperature_factor_per_degree', {})
                        rain_sensitivity = sensitivity.get('fpi_rainfall_factor_per_100mm', {})
                        
                        # Use scenario-specific values if available, otherwise fall back to defaults
                        if isinstance(temp_sensitivity, dict):
                            temp_factor = 1.0 + (temp_increase * temp_sensitivity.get(scenario, -0.10))
                        else:
                            temp_factor = 1.0 + (temp_increase * temp_sensitivity)
                            
                        if isinstance(rain_sensitivity, dict):
                            rain_factor = 1.0 + (rainfall_reduction / 100 * rain_sensitivity.get(scenario, -0.08))
                        else:
                            rain_factor = 1.0 + (rainfall_reduction / 100 * rain_sensitivity)
                        
                        adjusted_fpi = baseline_value * temp_factor * rain_factor
                        
                        # Apply bounds
                        fpi_min = climate_config.get('parameter_bounds', {}).get('fpi_min', 0.40)
                        fpi_max = climate_config.get('parameter_bounds', {}).get('fpi_max', 1.20)
                        adjusted_fpi = max(fpi_min, min(fpi_max, adjusted_fpi))
                    else:
                        # No sensitivity parameters available, use default values
                        temp_factor = 1.0 + (temp_increase * -0.10)
                        rain_factor = 1.0 + (rainfall_reduction / 100 * -0.08)
                        adjusted_fpi = baseline_value * temp_factor * rain_factor
                    
                    fpi_ratios[scenario] = adjusted_fpi
    
    # Adjust mortality rates with scenario-specific sensitivity
    mortality_scenarios = ['degraded', 'managed', 'reforestation']
    
    for scenario in mortality_scenarios:
        key = f'm_{scenario}'
        if key in adjusted_config:
            baseline_mortality = adjusted_config[key]
            
            if sensitivity:
                # Get scenario-specific sensitivity parameters
                temp_sensitivity = sensitivity.get('mortality_temperature_per_degree', {})
                rain_sensitivity = sensitivity.get('mortality_rainfall_per_100mm', {})
                
                # Use scenario-specific values if available, otherwise fall back to defaults
                if isinstance(temp_sensitivity, dict):
                    temp_factor = temp_sensitivity.get(scenario, 0.012)
                else:
                    temp_factor = temp_sensitivity
                    
                if isinstance(rain_sensitivity, dict):
                    rain_factor = rain_sensitivity.get(scenario, 0.008)
                else:
                    rain_factor = rain_sensitivity
                
                temp_effect = temp_increase * temp_factor
                rain_effect = rainfall_reduction / 100 * rain_factor
                adjusted_mortality = baseline_mortality + temp_effect + rain_effect
                
                # Apply bounds
                mort_min = climate_config.get('parameter_bounds', {}).get('mortality_min', 0.005)
                mort_max = climate_config.get('parameter_bounds', {}).get('mortality_max', 0.080)
                adjusted_mortality = max(mort_min, min(mort_max, adjusted_mortality))
            else:
                adjusted_mortality = adjust_climate_parameters(
                    {'mortality': baseline_mortality}, temp_increase, rainfall_reduction
                )['mortality']
            
            adjusted_config[key] = adjusted_mortality
    
    # Adjust disturbance probabilities with scenario-specific sensitivity
    for scenario in mortality_scenarios:
        key = f'pdist_{scenario}'
        if key in adjusted_config:
            baseline_pdist = adjusted_config[key]
            
            if sensitivity:
                # Get scenario-specific sensitivity parameters
                temp_sensitivity = sensitivity.get('pdist_temperature_per_degree', {})
                rain_sensitivity = sensitivity.get('pdist_rainfall_per_100mm', {})
                
                # Use scenario-specific values if available, otherwise fall back to defaults
                if isinstance(temp_sensitivity, dict):
                    temp_factor = temp_sensitivity.get(scenario, 0.03)
                else:
                    temp_factor = temp_sensitivity
                    
                if isinstance(rain_sensitivity, dict):
                    rain_factor = rain_sensitivity.get(scenario, 0.02)
                else:
                    rain_factor = rain_sensitivity
                
                temp_effect = temp_increase * temp_factor
                rain_effect = rainfall_reduction / 100 * rain_factor
                adjusted_pdist = baseline_pdist + temp_effect + rain_effect
                
                # Apply bounds
                pdist_min = climate_config.get('parameter_bounds', {}).get('pdist_min', 0.01)
                pdist_max = climate_config.get('parameter_bounds', {}).get('pdist_max', 0.25)
                adjusted_pdist = max(pdist_min, min(pdist_max, adjusted_pdist))
            else:
                adjusted_pdist = adjust_climate_parameters(
                    {'pdist': baseline_pdist}, temp_increase, rainfall_reduction
                )['pdist']
            
            adjusted_config[key] = adjusted_pdist
    
    # Adjust disturbance severities with scenario-specific sensitivity
    for scenario in mortality_scenarios:
        key = f'dsev_{scenario}'
        if key in adjusted_config:
            baseline_dsev = adjusted_config[key]
            
            if sensitivity:
                # Get scenario-specific sensitivity parameters
                temp_sensitivity = sensitivity.get('dsev_temperature_per_degree', {})
                rain_sensitivity = sensitivity.get('dsev_rainfall_per_100mm', {})
                
                # Use scenario-specific values if available, otherwise fall back to defaults
                if isinstance(temp_sensitivity, dict):
                    temp_factor = temp_sensitivity.get(scenario, 0.02)
                else:
                    temp_factor = temp_sensitivity
                    
                if isinstance(rain_sensitivity, dict):
                    rain_factor = rain_sensitivity.get(scenario, 0.012)
                else:
                    rain_factor = rain_sensitivity
                
                temp_effect = temp_increase * temp_factor
                rain_effect = rainfall_reduction / 100 * rain_factor
                adjusted_dsev = baseline_dsev + temp_effect + rain_effect
                
                # Apply bounds
                dsev_min = climate_config.get('parameter_bounds', {}).get('dsev_min', 0.02)
                dsev_max = climate_config.get('parameter_bounds', {}).get('dsev_max', 0.35)
                adjusted_dsev = max(dsev_min, min(dsev_max, adjusted_dsev))
            else:
                adjusted_dsev = adjust_climate_parameters(
                    {'dsev': baseline_dsev}, temp_increase, rainfall_reduction
                )['dsev']
            
            adjusted_config[key] = adjusted_dsev
    
    return adjusted_config


class ClimateAdjustments:
    """Climate adjustment utility class."""
    
    def __init__(self, config_dir: Path = Path("configs/base")):
        """Initialize with configuration directory."""
        self.config_dir = config_dir
    
    def load_climate_config(self, climate_name: str) -> Dict[str, Any]:
        """Load climate configuration."""
        return load_climate_config(self.config_dir, climate_name)
    
    def apply_adjustments(self, baseline_config: Dict[str, Any], climate_name: str) -> Dict[str, Any]:
        """Apply climate adjustments to baseline configuration."""
        climate_config = self.load_climate_config(climate_name)
        return apply_climate_adjustments_to_config(baseline_config, climate_config)