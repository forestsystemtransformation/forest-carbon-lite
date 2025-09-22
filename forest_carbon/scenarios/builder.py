#!/usr/bin/env python3
"""
Scenario Builder System for Forest Carbon Modeling

This system creates dynamic scenario configurations by combining:
- Forest types (ETOF, EOF, AFW)
- Climate scenarios (normal, paris2050, hot_dry)
- Management levels (light, moderate, adaptive)
- Time periods (2025-2050, etc.)

Config files are named: {forest}_{climate}_{years}_{management}.yaml
Example: ETOF_paris2050_2025-2050_adaptive.yaml
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import itertools
import pandas as pd
from datetime import datetime

class ForestType:
    """Forest type discovery and management."""
    
    @classmethod
    def discover_forest_types(cls, base_dir: Path = Path("configs/base")) -> List[str]:
        """Discover available forest types from site_*.yaml files."""
        if not base_dir.exists():
            return []
        
        forest_types = []
        for config_file in base_dir.glob("site_*.yaml"):
            forest_type = config_file.stem.replace("site_", "")
            forest_types.append(forest_type)
        
        return sorted(forest_types)
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available forest types."""
        return cls.discover_forest_types()

class ClimateScenario:
    """Climate scenario discovery and management."""
    
    @classmethod
    def discover_climate_scenarios(cls, base_dir: Path = Path("configs/base")) -> List[str]:
        """Discover available climate scenarios from climate_*.yaml files."""
        if not base_dir.exists():
            return []
        
        climate_scenarios = []
        for config_file in base_dir.glob("climate_*.yaml"):
            climate_scenario = config_file.stem.replace("climate_", "")
            climate_scenarios.append(climate_scenario)
        
        return sorted(climate_scenarios)
    
    @classmethod
    def get_available_scenarios(cls) -> List[str]:
        """Get list of available climate scenarios."""
        return cls.discover_climate_scenarios()

class ManagementLevel:
    """Management level discovery and management."""
    
    @classmethod
    def discover_management_levels(cls, base_dir: Path = Path("configs/base")) -> List[str]:
        """Discover available management levels from management_*.yaml files."""
        if not base_dir.exists():
            return []
        
        management_levels = []
        for config_file in base_dir.glob("management_*.yaml"):
            management_level = config_file.stem.replace("management_", "")
            management_levels.append(management_level)
        
        return sorted(management_levels)
    
    @classmethod
    def get_available_levels(cls) -> List[str]:
        """Get list of available management levels."""
        return cls.discover_management_levels()


@dataclass
class ScenarioConfig:
    """Container for a complete scenario configuration."""
    name: str
    forest_type: ForestType
    climate: ClimateScenario
    management: ManagementLevel
    time_period: str
    years: int
    config: Dict[str, Any]
    
    def save(self, output_dir: Path = Path("configs/generated")) -> Path:
        """Save configuration to YAML file with detailed documentation."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        filename = f"{self.name}.yaml"
        filepath = output_dir / filename
        
        # Add comprehensive documentation
        documented_config = {
            'scenario_metadata': {
                'name': self.name,
                'forest_type': self.forest_type,
                'climate_scenario': self.climate,
                'management_level': self.management,
                'time_period': self.time_period,
                'simulation_years': self.years,
                'created': datetime.now().isoformat(),
                'description': f"Forest Carbon Scenario: {self.forest_type} forest under {self.climate} climate with {self.management} management for {self.years} years"
            },
            'scenario_components': {
                'forest_type': {
                    'name': self.forest_type,
                    'description': f"Forest type configuration from site_{self.forest_type}.yaml",
                    'config_file': f"configs/base/site_{self.forest_type}.yaml"
                },
                'climate_scenario': {
                    'name': self.climate,
                    'description': f"Climate scenario configuration from climate_{self.climate}.yaml",
                    'config_file': f"configs/base/climate_{self.climate}.yaml"
                },
                'management_level': {
                    'name': self.management,
                    'description': f"Management level configuration from management_{self.management}.yaml",
                    'config_file': f"configs/base/management_{self.management}.yaml"
                }
            },
            'simulation_parameters': {
                'duration_years': self.years,
                'time_period': self.time_period,
                'project_area_ha': 1000.0,  # Default area
                'uncertainty_analysis': True,
                'generate_plots': True
            }
        }
        
        # Merge with actual configuration
        documented_config.update(self.config)
        
        # Save to file
        with open(filepath, 'w') as f:
            yaml.dump(documented_config, f, default_flow_style=False, sort_keys=False)
        
        return filepath


class ScenarioBuilder:
    """Builds scenario configurations by combining base configurations."""
    
    def __init__(self, base_dir: Path = Path("configs/base")):
        """
        Initialize scenario builder.
        
        Args:
            base_dir: Directory containing base configuration files
        """
        self.base_dir = base_dir
        self.config_loader = None  # Will be initialized when needed
    
    def _load_base_config(self, config_type: str, config_name: str) -> Dict[str, Any]:
        """Load a base configuration file."""
        config_file = self.base_dir / f"{config_type}_{config_name}.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _merge_configs(self, site_config: Dict[str, Any], 
                      climate_config: Dict[str, Any], 
                      management_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge base configurations into a complete scenario configuration."""
        # Start with site configuration as base
        merged_config = site_config.copy()
        
        # Create scenario-specific mortality fields from m_baseline if it exists
        if 'm_baseline' in merged_config:
            baseline_mortality = merged_config['m_baseline']
            merged_config['m_degraded'] = baseline_mortality
            merged_config['m_managed'] = baseline_mortality
            merged_config['m_reforestation'] = baseline_mortality
            # Remove the original m_baseline field
            del merged_config['m_baseline']
        
        # Apply climate adjustments if present
        if 'effects' in climate_config:
            climate_effects = climate_config['effects']
            # Apply climate adjustments to FPI ratios
            if 'fpi_ratios' in merged_config:
                fpi_adjustment = climate_effects.get('fpi_adjustment', 1.0)
                for scenario in ['baseline', 'management', 'reforestation']:
                    if scenario in merged_config['fpi_ratios']:
                        merged_config['fpi_ratios'][scenario] *= fpi_adjustment
        
        # Apply management adjustments if present
        if 'effects' in management_config:
            effects = management_config['effects']
            
            # Load baseline management effects for comparison
            baseline_mgmt_path = self.base_dir / "management_baseline.yaml"
            baseline_effects = {}
            if baseline_mgmt_path.exists():
                with open(baseline_mgmt_path, 'r') as f:
                    baseline_mgmt_config = yaml.safe_load(f)
                    baseline_effects = baseline_mgmt_config.get('effects', {})
            
            # DESIGN DECISION: Reforestation scenarios represent natural forest restoration
            # Management effects are NOT applied to reforestation by default.
            # This can be extended in the future to support "managed reforestation" scenarios
            # if explicitly specified in the management configuration.
            
            # Apply management effects to TYF calibrations - SCENARIO SPECIFIC
            if 'tyf_calibrations' in merged_config:
                # BASELINE: Apply baseline management effects (degradation)
                if 'baseline' in merged_config['tyf_calibrations']:
                    baseline_tyf = merged_config['tyf_calibrations']['baseline']
                    if 'y_multiplier' in baseline_effects:
                        baseline_tyf['y'] *= baseline_effects['y_multiplier']  # Apply baseline degradation
                
                # MANAGEMENT: Apply specified management effects (improvement)
                if 'management' in merged_config['tyf_calibrations']:
                    mgmt_tyf = merged_config['tyf_calibrations']['management']
                    if 'y_multiplier' in effects:
                        mgmt_tyf['y'] *= effects['y_multiplier']  # Apply management improvement
                
                # REFORESTATION: Apply management effects if explicitly enabled
                if 'reforestation' in merged_config['tyf_calibrations']:
                    reforestation_tyf = merged_config['tyf_calibrations']['reforestation']
                    # Check if managed reforestation is enabled in management config
                    if effects.get('apply_to_reforestation', False):
                        if 'y_multiplier' in effects:
                            reforestation_tyf['y'] *= effects['y_multiplier']  # Apply management to reforestation
                    # Otherwise, keep natural reforestation (y=1.0)
            
            # Apply mortality factors - SCENARIO SPECIFIC
            # BASELINE: Apply baseline management effects
            if 'mortality_factor' in baseline_effects and 'm_degraded' in merged_config:
                merged_config['m_degraded'] *= baseline_effects['mortality_factor']
            # MANAGEMENT: Apply specified management effects
            if 'mortality_factor' in effects and 'm_managed' in merged_config:
                merged_config['m_managed'] *= effects['mortality_factor']
                # REFORESTATION: Apply management effects if explicitly enabled
                if effects.get('apply_to_reforestation', False) and 'm_reforestation' in merged_config:
                    merged_config['m_reforestation'] *= effects['mortality_factor']
            
            # Apply disturbance factors - SCENARIO SPECIFIC
            # BASELINE: Apply baseline management effects
            if 'disturbance_factor' in baseline_effects:
                if 'pdist_degraded' in merged_config:
                    merged_config['pdist_degraded'] *= baseline_effects['disturbance_factor']
                if 'dsev_degraded' in merged_config:
                    merged_config['dsev_degraded'] *= baseline_effects['disturbance_factor']
            # MANAGEMENT: Apply specified management effects
            if 'disturbance_factor' in effects:
                if 'pdist_managed' in merged_config:
                    merged_config['pdist_managed'] *= effects['disturbance_factor']
                if 'dsev_managed' in merged_config:
                    merged_config['dsev_managed'] *= effects['disturbance_factor']
                # REFORESTATION: Apply management effects if explicitly enabled
                if effects.get('apply_to_reforestation', False):
                    if 'pdist_reforestation' in merged_config:
                        merged_config['pdist_reforestation'] *= effects['disturbance_factor']
                    if 'dsev_reforestation' in merged_config:
                        merged_config['dsev_reforestation'] *= effects['disturbance_factor']
        
        return merged_config
    
    def build_scenario(self, forest_type: str, climate: str, management: str, 
                      time_period: str, years: int) -> ScenarioConfig:
        """
        Build a single scenario configuration.
        
        Args:
            forest_type: Forest type identifier
            climate: Climate scenario identifier
            management: Management level identifier
            time_period: Time period string (e.g., "2025-2050")
            years: Simulation duration in years
            
        Returns:
            Complete scenario configuration
        """
        # Load base configurations
        site_config = self._load_base_config("site", forest_type)
        climate_config = self._load_base_config("climate", climate)
        management_config = self._load_base_config("management", management)
        
        # Merge configurations
        merged_config = self._merge_configs(site_config, climate_config, management_config)
        
        # Create scenario name (clean name without years)
        scenario_name = f"{forest_type}_{climate}_{management}"
        
        return ScenarioConfig(
            name=scenario_name,
            forest_type=forest_type,
            climate=climate,
            management=management,
            time_period=time_period,
            years=years,
            config=merged_config
        )


class ScenarioGenerator:
    """Generates multiple scenario configurations."""
    
    def __init__(self, output_dir: Path = Path("configs/generated")):
        """
        Initialize scenario generator.
        
        Args:
            output_dir: Directory for generated scenario configurations
        """
        self.output_dir = output_dir
        self.builder = ScenarioBuilder()
    
    def generate_matrix(self, forest_types: List[str], climates: List[str], 
                       managements: List[str], time_periods: List[str]) -> List[ScenarioConfig]:
        """
        Generate all combinations of scenarios.
        
        Args:
            forest_types: List of forest type identifiers
            climates: List of climate scenario identifiers
            managements: List of management level identifiers
            time_periods: List of time period strings
            
        Returns:
            List of scenario configurations
        """
        scenarios = []
        
        for forest_type in forest_types:
            for climate in climates:
                for management in managements:
                    for time_period in time_periods:
                        # Calculate years from time period
                        years = self._calculate_years(time_period)
                        
                        try:
                            scenario = self.builder.build_scenario(
                                forest_type, climate, management, time_period, years
                            )
                            scenarios.append(scenario)
                        except FileNotFoundError as e:
                            print(f"Warning: Skipping scenario due to missing config: {e}")
                            continue
        
        return scenarios
    
    def _calculate_years(self, time_period: str) -> int:
        """Calculate simulation years from time period string."""
        if '-' in time_period:
            # Format: "2025-2050"
            start_year, end_year = map(int, time_period.split('-'))
            return end_year - start_year
        else:
            # Assume it's just a number
            return int(time_period)
    
    def save_all_scenarios(self, scenarios: List[ScenarioConfig]) -> pd.DataFrame:
        """
        Save all scenarios and return manifest.
        
        Args:
            scenarios: List of scenario configurations
            
        Returns:
            DataFrame with scenario manifest
        """
        manifest_data = []
        
        for scenario in scenarios:
            filepath = scenario.save(self.output_dir)
            
            manifest_data.append({
                'name': scenario.name,
                'forest_type': scenario.forest_type,
            'climate': scenario.climate,
            'management': scenario.management,
                'time_period': scenario.time_period,
                'years': scenario.years,
                'file': str(filepath)
            })
        
        # Create manifest DataFrame
        manifest_df = pd.DataFrame(manifest_data)
        
        # Save manifest
        manifest_path = self.output_dir / "scenario_manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)
        
        return manifest_df


def main():
    """Example usage of scenario builder."""
    
    print("🌲 Forest Carbon Scenario Builder")
    print("=" * 50)
    
    # Get available options
    forest_types = ForestType.get_available_types()
    climates = ClimateScenario.get_available_scenarios()
    managements = ManagementLevel.get_available_levels()
    
    print(f"Available Forest Types: {forest_types}")
    print(f"Available Climate Scenarios: {climates}")
    print(f"Available Management Levels: {managements}")
    
    if not forest_types or not climates or not managements:
        print("Error: Missing configuration files!")
        print("Ensure you have:")
        print("  - configs/base/site_*.yaml files")
        print("  - configs/base/climate_*.yaml files")
        print("  - configs/base/management_*.yaml files")
        return
    
    # Generate scenarios
    generator = ScenarioGenerator()
    
    scenarios = generator.generate_matrix(
        forest_types=forest_types,
        climates=climates,
        managements=managements,
        time_periods=["2025-2050"]  # 25 years
    )
    
    print(f"\nGenerated {len(scenarios)} scenarios")
    
    # Save scenarios
    manifest = generator.save_all_scenarios(scenarios)
    
    print(f"Saved scenarios to: configs/generated/")
    print(f"Manifest saved to: configs/generated/scenario_manifest.csv")
    
    # Show sample scenarios
    print(f"\nSample scenarios:")
    for i, row in manifest.head(5).iterrows():
        print(f"  {row['name']} ({row['years']} years)")
    
    if len(scenarios) > 5:
        print(f"  ... and {len(scenarios) - 5} more")

if __name__ == "__main__":
    main()