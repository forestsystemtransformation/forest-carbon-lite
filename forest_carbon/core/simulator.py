"""Main simulation orchestrator for forest carbon modeling."""

import pandas as pd
import numpy as np
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .models.tyf_engine import TYFEngine
from .models.carbon_pools import CarbonPoolManager
from .models.disturbance import DisturbanceModel
from .models.economics import EconomicsModel
from ..utils.config_loader import ConfigLoader
from ..utils.guardrails import validate_all_parameters, check_parameter_realism
from ..visualization.plotter import Plotter

@dataclass
class SimulationResults:
    """Container for simulation results."""
    sequestration_curves: pd.DataFrame
    carbon_pools: Dict[str, List[Dict]]
    economics: Dict[str, pd.DataFrame]
    summary: pd.DataFrame
    final_pools: Dict[str, Dict]

class ForestCarbonSimulator:
    """Main simulator coordinating all components."""
    
    def __init__(self, forest_type: str, years: int = 50, 
                 area_ha: float = 1.0, output_dir: str = "output", 
                 config_file: Optional[str] = None, climate_config: Optional[str] = None,
                 validate_config: bool = True, enable_uncertainty: bool = True, 
                 seed: Optional[int] = None):
        """
        Initialize simulator.
        
        Args:
            forest_type: Forest type (EOF or ETOF)
            years: Simulation duration
            area_ha: Project area in hectares
            output_dir: Output directory name
            config_file: Optional custom config file name or path
            climate_config: Optional climate configuration name (without .yaml extension)
            validate_config: Whether to validate configuration (default: True)
            enable_uncertainty: Whether to run uncertainty analysis (default: False)
            seed: Random seed for reproducibility (default: None)
        """
        # Validate input parameters
        if years <= 0:
            raise ValueError("Simulation years must be positive")
        if area_ha <= 0:
            raise ValueError("Project area must be positive")
        
        self.forest_type = forest_type
        self.years = years
        self.area_ha = area_ha
        self.output_dir = Path(output_dir)
        self.config_file = config_file
        self.climate_config = climate_config
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration with validation
        self.config_loader = ConfigLoader(validate=validate_config)
        self.config = self.config_loader.load_site_config(forest_type, config_file, climate_config)
        
        # Apply baseline degradation to make baseline scenario actually degrade
        self._apply_baseline_degradation()
        
        # Initialize economics - handle both dict and validated config
        if hasattr(self.config, 'economics') and self.config.economics:
            # Validated config
            economics_dict = self.config.economics.model_dump()
        else:
            # Dict config - load from config_loader if not present
            economics_dict = self.config.get('economics', {})
            if not economics_dict:
                economics_dict = self.config_loader._load_economics_defaults()
        
        # Assert that we have valid economics configuration
        assert economics_dict, "Economics configuration must not be empty"
        assert 'carbon' in economics_dict, "Economics configuration must contain 'carbon' section"
        assert 'costs' in economics_dict, "Economics configuration must contain 'costs' section"
        
        self.economics_model = EconomicsModel(economics_dict)
        
        # Store results
        self.results = {}
        
        # Store config file paths for transparency
        self.config_file_used = config_file
        self.climate_config_used = climate_config
        
        # Store uncertainty analysis flag and seed
        self.enable_uncertainty = enable_uncertainty
        self.seed = seed
    
    def _apply_baseline_degradation(self):
        """Apply baseline degradation to make baseline scenario actually degrade."""
        # Apply 10% degradation to baseline FPI ratio
        if hasattr(self.config, 'fpi_ratios'):
            # Validated config
            if hasattr(self.config.fpi_ratios, 'baseline'):
                self.config.fpi_ratios.baseline *= 0.90  # 10% degradation
        else:
            # Dict config
            if 'fpi_ratios' in self.config and 'baseline' in self.config['fpi_ratios']:
                self.config['fpi_ratios']['baseline'] *= 0.90  # 10% degradation
        
        # Apply degradation to TYF calibrations
        if hasattr(self.config, 'tyf_calibrations'):
            # Validated config
            if hasattr(self.config.tyf_calibrations, 'baseline'):
                baseline_tyf = self.config.tyf_calibrations.baseline
                if hasattr(baseline_tyf, 'y'):
                    baseline_tyf.y *= 0.90  # 10% growth reduction
        else:
            # Dict config
            if 'tyf_calibrations' in self.config and 'baseline' in self.config['tyf_calibrations']:
                baseline_tyf = self.config['tyf_calibrations']['baseline']
                if 'y' in baseline_tyf:
                    baseline_tyf['y'] *= 0.90  # 10% growth reduction
    
    def _create_tyf_engine(self, scenario: str) -> TYFEngine:
        """Create TYF engine for specific scenario."""
        # Validate scenario name
        valid_scenarios = {'baseline', 'management', 'reforestation'}
        assert scenario in valid_scenarios, f"Invalid scenario: {scenario}. Must be one of {valid_scenarios}"
        
        # Handle both dict and validated config
        if hasattr(self.config, 'tyf_calibrations'):
            tyf_calibrations = self.config.tyf_calibrations
            # Check if it's a dictionary (validated config) or has attributes (dict config)
            if isinstance(tyf_calibrations, dict):
                if scenario not in tyf_calibrations:
                    raise ValueError(f"TYF calibration missing for scenario: {scenario}. Available scenarios: {list(tyf_calibrations.keys())}")
                calibration = tyf_calibrations[scenario]
            else:
                # Handle dict config case
                if not hasattr(tyf_calibrations, scenario):
                    raise ValueError(f"TYF calibration missing for scenario: {scenario}. Available scenarios: {list(tyf_calibrations.keys())}")
                calibration = tyf_calibrations[scenario]
            
            # Get FPI ratio for this scenario
            fpi_ratio = 1.0  # Default
            if hasattr(self.config, 'fpi_ratios') and self.config.fpi_ratios:
                fpi_config = self.config.fpi_ratios
                if scenario == 'baseline':
                    fpi_ratio = fpi_config.baseline
                elif scenario == 'management':
                    fpi_ratio = fpi_config.management
                elif scenario == 'reforestation':
                    fpi_ratio = fpi_config.reforestation
            
            # Validate parameters using scientifically defensible guardrails
            validated_params = validate_all_parameters(
                fpi=fpi_ratio,
                y=calibration.y,
                m=0.01,  # Not used in TYF engine
                pdist=0.05,  # Not used in TYF engine
                dsev=0.2,  # Not used in TYF engine
                forest_type=self.forest_type,
                scenario=scenario
            )
            
            # Check for parameter realism warnings
            warnings = check_parameter_realism({
                'fpi': validated_params['fpi'],
                'y': validated_params['y']
            }, self.forest_type)
            
            if warnings:
                print(f"Parameter warnings for {scenario} scenario:")
                for param, warning in warnings.items():
                    print(f"  {param}: {warning}")
            
            return TYFEngine(
                M=calibration.M,
                G=calibration.G,
                y=validated_params['y'],
                FPI_ratio=validated_params['fpi']
            )
        else:
            assert 'tyf_calibrations' in self.config, "TYF calibrations missing from config"
            assert scenario in self.config['tyf_calibrations'], f"TYF calibration missing for scenario: {scenario}"
            calibration = self.config['tyf_calibrations'][scenario]
            
            # Get FPI ratio for this scenario
            fpi_ratio = 1.0  # Default
            if 'fpi_ratios' in self.config and self.config['fpi_ratios']:
                fpi_config = self.config['fpi_ratios']
                if scenario == 'baseline':
                    fpi_ratio = fpi_config.get('baseline', 1.0)
                elif scenario == 'management':
                    fpi_ratio = fpi_config.get('management', 1.0)
                elif scenario == 'reforestation':
                    fpi_ratio = fpi_config.get('reforestation', 1.0)
            
            # Validate parameters using scientifically defensible guardrails
            validated_params = validate_all_parameters(
                fpi=fpi_ratio,
                y=calibration['y'],
                m=0.01,  # Not used in TYF engine
                pdist=0.05,  # Not used in TYF engine
                dsev=0.2,  # Not used in TYF engine
                forest_type=self.forest_type,
                scenario=scenario
            )
            
            # Check for parameter realism warnings
            warnings = check_parameter_realism({
                'fpi': validated_params['fpi'],
                'y': validated_params['y']
            }, self.forest_type)
            
            if warnings:
                print(f"Parameter warnings for {scenario} scenario:")
                for param, warning in warnings.items():
                    print(f"  {param}: {warning}")
            
            return TYFEngine(
                M=calibration['M'],
                G=calibration['G'],
                y=validated_params['y'],
                FPI_ratio=validated_params['fpi']
            )
    
    def _create_disturbance_model(self, scenario: str) -> DisturbanceModel:
        """Create disturbance model for scenario."""
        # Validate scenario name
        valid_scenarios = {'baseline', 'management', 'reforestation'}
        assert scenario in valid_scenarios, f"Invalid scenario: {scenario}. Must be one of {valid_scenarios}"
        
        # Map scenario names to config keys
        scenario_map = {
            'baseline': 'degraded',
            'management': 'managed',
            'reforestation': 'reforestation'
        }
        
        suffix = scenario_map[scenario]
        
        # Handle both dict and validated config
        if hasattr(self.config, f'm_{suffix}'):
            # Validated config - use attribute access
            chronic_mortality = getattr(self.config, f'm_{suffix}')
            disturbance_probability = getattr(self.config, f'pdist_{suffix}')
            disturbance_severity = getattr(self.config, f'dsev_{suffix}')
        else:
            # Dict config - use get method
            chronic_mortality = self.config.get(f'm_{suffix}', 0.01)
            disturbance_probability = self.config.get(f'pdist_{suffix}', 0.05)
            disturbance_severity = self.config.get(f'dsev_{suffix}', 0.2)
        
        # Validate parameters using scientifically defensible guardrails
        validated_params = validate_all_parameters(
            fpi=1.0,  # Not used in disturbance model
            y=1.0,    # Not used in disturbance model
            m=chronic_mortality,
            pdist=disturbance_probability,
            dsev=disturbance_severity,
            forest_type=self.forest_type,
            scenario=scenario
        )
        
        # Update parameters with validated values
        chronic_mortality = validated_params['m']
        disturbance_probability = validated_params['pdist']
        disturbance_severity = validated_params['dsev']
        
        # Check for parameter realism warnings
        warnings = check_parameter_realism({
            'm': chronic_mortality,
            'pdist': disturbance_probability,
            'dsev': disturbance_severity
        }, self.forest_type)
        
        if warnings:
            print(f"Parameter warnings for {scenario} scenario:")
            for param, warning in warnings.items():
                print(f"  {param}: {warning}")
        
        return DisturbanceModel(
            chronic_mortality=chronic_mortality,
            disturbance_probability=disturbance_probability,
            disturbance_severity=disturbance_severity,
            seed=42  # For reproducibility
        )
    
    def _get_initial_age(self, scenario: str) -> float:
        """Get initial age for scenario."""
        # Validate scenario name
        valid_scenarios = {'baseline', 'management', 'reforestation'}
        assert scenario in valid_scenarios, f"Invalid scenario: {scenario}. Must be one of {valid_scenarios}"
        
        # Handle both dict and validated config
        if hasattr(self.config, 'age_degraded'):
            # Validated config - use attribute access
            if scenario == 'baseline':
                age = self.config.age_degraded
            elif scenario == 'management':
                age = self.config.age_managed
            else:  # reforestation
                age = self.config.age_reforestation
        else:
            # Dict config - use get method
            if scenario == 'baseline':
                age = self.config.get('age_degraded', 15)
            elif scenario == 'management':
                age = self.config.get('age_managed', 15)
            else:  # reforestation
                age = self.config.get('age_reforestation', 0)
        
        # Validate age is non-negative
        assert age >= 0, f"Initial age must be non-negative, got {age} for scenario {scenario}"
        return age
    
    def _get_initial_biomass(self, tyf_engine: TYFEngine, age: float, scenario: str) -> float:
        """Calculate initial biomass for given age."""
        # Validate inputs
        assert age >= 0, f"Age must be non-negative, got {age}"
        assert scenario in {'baseline', 'management', 'reforestation'}, f"Invalid scenario: {scenario}"
        
        if age <= 0:
            return 0.0
        
        # Check if fixed initial biomass is specified
        fixed_biomass_key = f'initial_biomass_{scenario}'
        
        # Handle both dict and validated config
        if hasattr(self.config, fixed_biomass_key):
            # Validated config - use attribute access
            biomass = getattr(self.config, fixed_biomass_key)
        elif fixed_biomass_key in self.config:
            # Dict config - use key access
            biomass = self.config[fixed_biomass_key]
        else:
            # Fallback to TYF calculation
            biomass = tyf_engine.calculate_total_agb(age)
        
        # Validate biomass is non-negative
        assert biomass >= 0, f"Initial biomass must be non-negative, got {biomass} for scenario {scenario}"
        return biomass
    
    def simulate_scenario(self, scenario: str) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Simulate a single scenario.
        
        Args:
            scenario: Scenario name
            
        Returns:
            Tuple of (time series DataFrame, carbon pools history)
        """
        # Validate scenario name
        valid_scenarios = {'baseline', 'management', 'reforestation'}
        assert scenario in valid_scenarios, f"Invalid scenario: {scenario}. Must be one of {valid_scenarios}"
        
        # Initialize components
        tyf = self._create_tyf_engine(scenario)
        # Handle both dict and validated config for root_shoot
        if hasattr(self.config, 'root_shoot'):
            root_shoot = self.config.root_shoot
        else:
            root_shoot = self.config.get('root_shoot', 0.25)
        
        # Validate root_shoot ratio
        assert 0 < root_shoot <= 1, f"Root-shoot ratio must be between 0 and 1, got {root_shoot}"
        
        carbon_pools = CarbonPoolManager(root_shoot)
        disturbance = self._create_disturbance_model(scenario)
        
        # Get initial conditions
        initial_age = self._get_initial_age(scenario)
        initial_biomass = self._get_initial_biomass(tyf, initial_age, scenario)
        
        # Initialize carbon pools
        if initial_biomass > 0:
            # Handle both dict and validated config
            if hasattr(self.config, 'S0_deg_frac'):
                # Validated config - use attribute access
                if scenario == 'baseline':
                    soil_frac = self.config.S0_deg_frac
                elif scenario == 'management':
                    soil_frac = self.config.S0_man_frac
                else:
                    soil_frac = self.config.S0_new_frac
            else:
                # Dict config - use get method
                if scenario == 'baseline':
                    soil_frac = self.config.get('S0_deg_frac', 0.3)
                elif scenario == 'management':
                    soil_frac = self.config.get('S0_man_frac', 0.3)
                else:
                    soil_frac = self.config.get('S0_new_frac', 0.01)
            
            # Validate soil fraction
            assert 0 <= soil_frac <= 1, f"Soil fraction must be between 0 and 1, got {soil_frac} for scenario {scenario}"
            
            carbon_pools.initialize_from_standing_forest(initial_biomass, soil_frac)
        
        # Record initial state before any growth calculations (year 0)
        results = []
        pools_history = []
        
        # Record the true initial state (before any simulation steps)
        initial_pools = carbon_pools.get_pools()
        results.append({
            'year': 0,
            'age': initial_age,
            'agb_growth': 0.0,  # No growth in initial state
            'total_agb': initial_pools.agb / 0.47,  # Convert back to biomass
            'total_carbon': initial_pools.get_total_carbon(),
            'total_co2e': initial_pools.get_total_co2e(),
            'disturbance': False  # No disturbance in initial state
        })
        
        # Store detailed initial pools
        initial_pools_dict = initial_pools.to_dict()
        initial_pools_dict['year'] = 0
        pools_history.append(initial_pools_dict)
        
        # Run simulation starting from year 1 (year 0 already recorded)
        for year in range(1, self.years):
            current_age = initial_age + year
            next_age = current_age + 1
            
            # Calculate biomass growth with year-specific FPI ratio
            delta_agb = tyf.calculate_delta_agb(current_age, next_age, year)
            
            # Update carbon pools
            carbon_pools.update_biomass(delta_agb)
            
            # Apply disturbance
            dist_info = disturbance.simulate_year()
            carbon_pools.apply_mortality(dist_info['chronic_mortality'])
            
            if dist_info['disturbance_occurred']:
                carbon_pools.apply_disturbance(dist_info['disturbance_severity'])
            
            # Decay dead pools
            carbon_pools.decay_pools()
            
            # Store results
            pools = carbon_pools.get_pools()
            results.append({
                'year': year,
                'age': current_age,
                'agb_growth': delta_agb,
                'total_agb': pools.agb / 0.47,  # Convert back to biomass
                'total_carbon': pools.get_total_carbon(),
                'total_co2e': pools.get_total_co2e(),
                'disturbance': dist_info['disturbance_occurred']
            })
            
            # Store detailed pools
            pools_dict = pools.to_dict()
            pools_dict['year'] = year
            pools_history.append(pools_dict)
        
        return pd.DataFrame(results), pools_history
    
    def run_all_scenarios(self) -> Dict[str, Tuple[pd.DataFrame, List[Dict]]]:
        """Run all three scenarios."""
        scenarios = ['baseline', 'management', 'reforestation']
        results = {}
        
        for scenario in scenarios:
            print(f"Running {scenario} scenario...")
            df, pools = self.simulate_scenario(scenario)
            
            # Validate results are not empty
            assert not df.empty, f"Simulation results are empty for scenario {scenario}"
            assert len(pools) > 0, f"Carbon pools history is empty for scenario {scenario}"
            assert len(df) == self.years, f"Expected {self.years} years of data, got {len(df)} for scenario {scenario}"
            
            results[scenario] = (df, pools)
        
        # Validate all scenarios completed successfully
        assert len(results) == 3, f"Expected 3 scenarios, got {len(results)}"
        assert all(scenario in results for scenario in scenarios), "Missing scenario results"
        
        return results
    
    def _copy_config_to_output(self, output_dir=None):
        """Copy the configuration file used for this simulation to the output directory for transparency."""
        if output_dir is None:
            output_dir = self.output_dir
            
        if self.config_file_used:
            # Determine the source config file path
            if Path(self.config_file_used).is_absolute():
                source_config = Path(self.config_file_used)
            else:
                source_config = Path("config") / self.config_file_used
            
            # Create destination path in output directory
            dest_config = output_dir / f"config_used_{self.forest_type}.yaml"
            
            # Copy the file
            if source_config.exists():
                shutil.copy2(source_config, dest_config)
                print(f"[SUCCESS] Configuration file copied to: {dest_config}")
            else:
                print(f"⚠ Warning: Could not find config file: {source_config}")
        else:
            # Use default config file
            default_config = Path("config") / f"site_{self.forest_type}.yaml"
            dest_config = self.output_dir / f"config_used_{self.forest_type}.yaml"
            
            if default_config.exists():
                shutil.copy2(default_config, dest_config)
                print(f"[SUCCESS] Default configuration file copied to: {dest_config}")
            else:
                print(f"⚠ Warning: Could not find default config file: {default_config}")
    
    def _create_metadata_file(self, output_dir=None):
        """Create a metadata file with simulation information for transparency."""
        if output_dir is None:
            output_dir = self.output_dir
            
        metadata = {
            "simulation_info": {
                "timestamp": datetime.now().isoformat(),
                "forest_type": self.forest_type,
                "simulation_years": self.years,
                "project_area_ha": self.area_ha,
                "config_file_used": self.config_file_used or f"site_{self.forest_type}.yaml",
                "output_directory": str(output_dir)
            },
            "model_parameters": self._get_model_parameters_for_metadata(),
            "software_info": {
                "model": "Forest Carbon Lite",
                "version": "1.0",
                "description": "FullCAM-lite forest carbon sequestration simulator"
            }
        }
        
        # Save metadata to JSON file
        metadata_file = output_dir / "simulation_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"[SUCCESS] Simulation metadata saved to: {metadata_file}")
    
    def _get_model_parameters_for_metadata(self):
        """Get model parameters for metadata, handling both dict and validated config."""
        if hasattr(self.config, 'tyf_calibrations'):
            # Validated config - convert to dict
            tyf_calibrations = {k: v.dict() for k, v in self.config.tyf_calibrations.items()}
            initial_ages = {
                "degraded": getattr(self.config, 'age_degraded', 'N/A'),
                "managed": getattr(self.config, 'age_managed', 'N/A'),
                "reforestation": getattr(self.config, 'age_reforestation', 'N/A')
            }
            mortality_rates = {
                "degraded": getattr(self.config, 'm_degraded', 'N/A'),
                "managed": getattr(self.config, 'm_managed', 'N/A'),
                "reforestation": getattr(self.config, 'm_reforestation', 'N/A')
            }
            disturbance_parameters = {
                "degraded": {
                    "probability": getattr(self.config, 'pdist_degraded', 'N/A'),
                    "severity": getattr(self.config, 'dsev_degraded', 'N/A')
                },
                "managed": {
                    "probability": getattr(self.config, 'pdist_managed', 'N/A'),
                    "severity": getattr(self.config, 'dsev_managed', 'N/A')
                },
                "reforestation": {
                    "probability": getattr(self.config, 'pdist_reforestation', 'N/A'),
                    "severity": getattr(self.config, 'dsev_reforestation', 'N/A')
                }
            }
        else:
            # Dict config - use get method
            tyf_calibrations = self.config.get('tyf_calibrations', {})
            initial_ages = {
                "degraded": self.config.get('age_degraded', 'N/A'),
                "managed": self.config.get('age_managed', 'N/A'),
                "reforestation": self.config.get('age_reforestation', 'N/A')
            }
            mortality_rates = {
                "degraded": self.config.get('m_degraded', 'N/A'),
                "managed": self.config.get('m_managed', 'N/A'),
                "reforestation": self.config.get('m_reforestation', 'N/A')
            }
            disturbance_parameters = {
                "degraded": {
                    "probability": self.config.get('pdist_degraded', 'N/A'),
                    "severity": self.config.get('dsev_degraded', 'N/A')
                },
                "managed": {
                    "probability": self.config.get('pdist_managed', 'N/A'),
                    "severity": self.config.get('dsev_managed', 'N/A')
                },
                "reforestation": {
                    "probability": self.config.get('pdist_reforestation', 'N/A'),
                    "severity": self.config.get('dsev_reforestation', 'N/A')
                }
            }
        
        return {
            "tyf_calibrations": tyf_calibrations,
            "initial_ages": initial_ages,
            "mortality_rates": mortality_rates,
            "disturbance_parameters": disturbance_parameters
        }
    
    def calculate_abatement(self, results: Dict) -> pd.DataFrame:
        """Calculate carbon abatement for managed scenarios using Project Level Additionality."""
        # Validate input
        assert 'baseline' in results, "Baseline scenario results are required for abatement calculation"
        assert len(results['baseline']) == 2, "Baseline results must contain (dataframe, pools) tuple"
        
        baseline_df = results['baseline'][0]
        assert not baseline_df.empty, "Baseline results cannot be empty"
        assert 'total_co2e' in baseline_df.columns, "Baseline results must contain 'total_co2e' column"
        
        abatement_data = {'year': baseline_df['year'].values}
        
        for scenario in ['management', 'reforestation']:
            if scenario in results:
                assert len(results[scenario]) == 2, f"{scenario} results must contain (dataframe, pools) tuple"
                scenario_df = results[scenario][0]
                assert not scenario_df.empty, f"{scenario} results cannot be empty"
                assert 'total_co2e' in scenario_df.columns, f"{scenario} results must contain 'total_co2e' column"
                
                if scenario == 'management':
                    # Management: Project Level Additionality = Management - Baseline
                    abatement = scenario_df['total_co2e'] - baseline_df['total_co2e']
                elif scenario == 'reforestation':
                    # Reforestation: Project Level Additionality = Reforestation - Zero
                    abatement = scenario_df['total_co2e'] - 0.0
                else:
                    raise ValueError(f"Unexpected scenario: {scenario}")
                
                abatement_data[f'{scenario}_abatement'] = abatement.values
        
        abatement_df = pd.DataFrame(abatement_data)
        assert not abatement_df.empty, "Abatement calculation resulted in empty dataframe"
        return abatement_df
    
    def compile_results(self, scenario_results: Dict) -> SimulationResults:
        """Compile all results into structured format."""
        # Validate input
        assert scenario_results, "Scenario results cannot be empty"
        required_scenarios = {'baseline', 'management', 'reforestation'}
        assert all(scenario in scenario_results for scenario in required_scenarios), f"Missing required scenarios. Expected {required_scenarios}, got {set(scenario_results.keys())}"
        
        # Combine sequestration curves
        sequestration = pd.DataFrame({'year': range(self.years)})
        carbon_pools_history = {}
        
        for scenario, (df, pools) in scenario_results.items():
            assert not df.empty, f"DataFrame for scenario {scenario} cannot be empty"
            assert 'total_co2e' in df.columns, f"DataFrame for scenario {scenario} must contain 'total_co2e' column"
            assert 'total_agb' in df.columns, f"DataFrame for scenario {scenario} must contain 'total_agb' column"
            assert len(pools) > 0, f"Carbon pools history for scenario {scenario} cannot be empty"
            
            sequestration[f'{scenario}_co2e'] = df['total_co2e']
            sequestration[f'{scenario}_agb'] = df['total_agb']
            carbon_pools_history[scenario] = pools
        
        # Calculate abatement
        abatement_df = self.calculate_abatement(scenario_results)
        for col in abatement_df.columns:
            if col != 'year':
                sequestration[col] = abatement_df[col]
        
        # Economic analysis
        economics = {}
        for scenario in ['management', 'reforestation']:
            if f'{scenario}_abatement' in sequestration.columns:
                abatement_series = sequestration[f'{scenario}_abatement'].values
                cashflow = self.economics_model.calculate_cashflow(
                    abatement_series.tolist(), 
                    scenario, 
                    self.area_ha
                )
                economics[scenario] = cashflow
        
        # Summary metrics
        summary_data = []
        for scenario, (df, pools) in scenario_results.items():
            final_pools = pools[-1] if pools else {}
            
            summary_data.append({
                'scenario': scenario,
                'forest_type': self.forest_type,
                'final_co2e_stock': df['total_co2e'].iloc[-1],
                'mean_annual_increment': df['total_co2e'].iloc[-1] / self.years,
                'total_agb': df['total_agb'].iloc[-1],
                'disturbance_events': df['disturbance'].sum()
            })
        
        # Add economic metrics
        for scenario, cashflow in economics.items():
            econ_summary = self.economics_model.generate_summary(cashflow)
            for row in summary_data:
                if row['scenario'] == scenario:
                    row.update(econ_summary)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Get final pools for comparison
        final_pools = {}
        for scenario, (_, pools) in scenario_results.items():
            if pools:
                final_pools[scenario] = pools[-1]
        
        # Validate final results
        assert not sequestration.empty, "Sequestration curves cannot be empty"
        assert carbon_pools_history, "Carbon pools history cannot be empty"
        assert not summary_df.empty, "Summary dataframe cannot be empty"
        assert final_pools, "Final pools cannot be empty"
        
        return SimulationResults(
            sequestration_curves=sequestration,
            carbon_pools=carbon_pools_history,
            economics=economics,
            summary=summary_df,
            final_pools=final_pools
        )
    
    def save_results(self, results: SimulationResults):
        """Save all results to CSV files."""
        # Validate input
        assert results is not None, "Results cannot be None"
        assert hasattr(results, 'sequestration_curves'), "Results must have sequestration_curves attribute"
        assert hasattr(results, 'summary'), "Results must have summary attribute"
        assert not results.sequestration_curves.empty, "Sequestration curves cannot be empty"
        assert not results.summary.empty, "Summary cannot be empty"
        
        # Create other subdirectory for data files
        other_dir = self.output_dir / "other"
        other_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy configuration file for transparency
        self._copy_config_to_output(other_dir)
        
        # Create metadata file for transparency
        self._create_metadata_file(other_dir)
        
        # Save sequestration curves
        results.sequestration_curves.to_csv(
            other_dir / 'sequestration_curves.csv', index=False
        )
        
        # Save summary
        results.summary.to_csv(
            other_dir / 'results_summary.csv', index=False
        )
        
        # Save economic results
        if results.economics:
            # Combine economic results
            combined_econ = []
            for scenario, df in results.economics.items():
                df['scenario'] = scenario
                combined_econ.append(df)
            
            if combined_econ:
                pd.concat(combined_econ).to_csv(
                    other_dir / 'cashflow_breakdown.csv', index=False
                )
                
                # Save financial summary
                finance_summary = []
                for scenario, df in results.economics.items():
                    summary = self.economics_model.generate_summary(df)
                    summary['scenario'] = scenario
                    finance_summary.append(summary)
                
                pd.DataFrame(finance_summary).to_csv(
                    other_dir / 'finance_results.csv', index=False
                )
        
        print(f"Results saved to {other_dir}/")
        
        # Generate summary JSON
        self.generate_summary_json(results, other_dir)
    
    def generate_summary_json(self, results: SimulationResults, output_dir=None):
        """Generate a comprehensive summary JSON file with results and insights."""
        if output_dir is None:
            output_dir = self.output_dir
        summary_data = {
            "simulation_info": {
                "forest_type": self.forest_type,
                "duration_years": self.years,
                "area_hectares": self.area_ha,
                "timestamp": datetime.now().isoformat(),
                "config_file": str(self.config_file) if self.config_file else f"site_{self.forest_type}.yaml"
            },
            "summary_results": {
                "scenarios": {}
            },
            "key_insights": {
                "carbon_sequestration": {},
                "economic_performance": {},
                "disturbance_analysis": {},
                "scenario_rankings": {}
            },
            "detailed_metrics": {}
        }
        
        # Extract summary results for each scenario
        for _, row in results.summary.iterrows():
            scenario = row['scenario']
            summary_data["summary_results"]["scenarios"][scenario] = {
                "final_co2e_stock": float(row['final_co2e_stock']) if pd.notna(row['final_co2e_stock']) else None,
                "mean_annual_increment": float(row['mean_annual_increment']) if pd.notna(row['mean_annual_increment']) else None,
                "total_agb": float(row['total_agb']) if pd.notna(row['total_agb']) else None,
                "disturbance_events": int(row['disturbance_events']) if pd.notna(row['disturbance_events']) else 0,
                "npv": float(row['npv']) if pd.notna(row['npv']) else None,
                "irr": float(row['irr']) if pd.notna(row['irr']) else None,
                "payback_period": float(row['payback_period']) if pd.notna(row['payback_period']) else None,
                "total_revenue": float(row['total_revenue']) if pd.notna(row['total_revenue']) else None,
                "total_costs": float(row['total_costs']) if pd.notna(row['total_costs']) else None,
                "total_credits": float(row['total_credits']) if pd.notna(row['total_credits']) else None,
                "avg_carbon_price": float(row['avg_carbon_price']) if pd.notna(row['avg_carbon_price']) else None
            }
        
        # Generate key insights
        insights = self._generate_insights(results.summary)
        summary_data["key_insights"] = insights
        
        # Add detailed metrics
        summary_data["detailed_metrics"] = {
            "carbon_sequestration_ranking": self._rank_scenarios_by_metric(results.summary, 'final_co2e_stock', 'desc'),
            "economic_performance_ranking": self._rank_scenarios_by_metric(results.summary, 'npv', 'desc'),
            "disturbance_resilience_ranking": self._rank_scenarios_by_metric(results.summary, 'disturbance_events', 'asc'),
            "credit_generation_ranking": self._rank_scenarios_by_metric(results.summary, 'total_credits', 'desc')
        }
        
        # Save JSON file
        json_path = output_dir / 'simulation_summary.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"[SUCCESS] Summary JSON saved to: {json_path}")
    
    def _generate_insights(self, summary_df: pd.DataFrame) -> Dict:
        """Generate key insights from simulation results."""
        insights = {
            "carbon_sequestration": {},
            "economic_performance": {},
            "disturbance_analysis": {},
            "scenario_rankings": {}
        }
        
        # Carbon sequestration insights
        max_carbon_scenario = summary_df.loc[summary_df['final_co2e_stock'].idxmax(), 'scenario']
        max_carbon_value = summary_df['final_co2e_stock'].max()
        min_carbon_scenario = summary_df.loc[summary_df['final_co2e_stock'].idxmin(), 'scenario']
        min_carbon_value = summary_df['final_co2e_stock'].min()
        
        insights["carbon_sequestration"] = {
            "best_performing_scenario": max_carbon_scenario,
            "highest_carbon_stock": float(max_carbon_value),
            "lowest_carbon_stock": float(min_carbon_value),
            "carbon_improvement": float(max_carbon_value - min_carbon_value),
            "improvement_percentage": float(((max_carbon_value - min_carbon_value) / min_carbon_value) * 100) if min_carbon_value > 0 else 0
        }
        
        # Economic performance insights
        economic_scenarios = summary_df[summary_df['npv'].notna()]
        if not economic_scenarios.empty:
            best_economic = economic_scenarios.loc[economic_scenarios['npv'].idxmax(), 'scenario']
            worst_economic = economic_scenarios.loc[economic_scenarios['npv'].idxmin(), 'scenario']
            
            insights["economic_performance"] = {
                "best_economic_scenario": best_economic,
                "highest_npv": float(economic_scenarios['npv'].max()),
                "lowest_npv": float(economic_scenarios['npv'].min()),
                "profitable_scenarios": int((economic_scenarios['npv'] > 0).sum()),
                "total_scenarios": len(economic_scenarios)
            }
        else:
            insights["economic_performance"] = {
                "note": "No economic scenarios with valid NPV data"
            }
        
        # Disturbance analysis
        total_disturbances = summary_df['disturbance_events'].sum()
        avg_disturbances = summary_df['disturbance_events'].mean()
        most_resilient = summary_df.loc[summary_df['disturbance_events'].idxmin(), 'scenario']
        least_resilient = summary_df.loc[summary_df['disturbance_events'].idxmax(), 'scenario']
        
        insights["disturbance_analysis"] = {
            "total_disturbance_events": int(total_disturbances),
            "average_disturbances_per_scenario": float(avg_disturbances),
            "most_resilient_scenario": most_resilient,
            "least_resilient_scenario": least_resilient,
            "disturbance_range": int(summary_df['disturbance_events'].max() - summary_df['disturbance_events'].min())
        }
        
        # Scenario rankings
        insights["scenario_rankings"] = {
            "by_carbon_sequestration": self._rank_scenarios_by_metric(summary_df, 'final_co2e_stock', 'desc'),
            "by_economic_performance": self._rank_scenarios_by_metric(summary_df, 'npv', 'desc'),
            "by_disturbance_resilience": self._rank_scenarios_by_metric(summary_df, 'disturbance_events', 'asc'),
            "by_credit_generation": self._rank_scenarios_by_metric(summary_df, 'total_credits', 'desc')
        }
        
        return insights
    
    def _rank_scenarios_by_metric(self, summary_df: pd.DataFrame, metric: str, order: str = 'desc') -> List[Dict]:
        """Rank scenarios by a specific metric."""
        # Filter out NaN values
        valid_data = summary_df[summary_df[metric].notna()].copy()
        
        if valid_data.empty:
            return [{"scenario": "N/A", "value": None, "rank": 1, "note": f"No valid data for {metric}"}]
        
        # Sort by metric
        if order == 'desc':
            valid_data = valid_data.sort_values(metric, ascending=False)
        else:
            valid_data = valid_data.sort_values(metric, ascending=True)
        
        # Create ranking
        ranking = []
        for rank, (_, row) in enumerate(valid_data.iterrows(), 1):
            ranking.append({
                "scenario": row['scenario'],
                "value": float(row[metric]) if pd.notna(row[metric]) else None,
                "rank": rank
            })
        
        return ranking
    
    def generate_plots(self, results: SimulationResults):
        """Generate all visualization plots."""
        # Use the output directory directly - Plotter will create plots subdirectory
        plotter = Plotter(self.output_dir)
        
        # Prepare data for plotting
        plot_results = {self.forest_type: results.sequestration_curves}
        
        # Generate plots
        plotter.plot_biomass_comparison(plot_results, [self.forest_type])
        plotter.plot_total_carbon_stocks_all_scenarios(plot_results, [self.forest_type])
        # Removed net_carbon_abatement plot - total carbon stocks plot is sufficient
        plotter.plot_reforestation_minus_losses(plot_results, [self.forest_type])  # Reforestation minus losses
        plotter.plot_management_minus_reforestation(plot_results, [self.forest_type])  # Management minus reforestation
        plotter.plot_project_level_additionality(plot_results, [self.forest_type])  # Now saves as additionality.png
        
        # Plot carbon pools for each scenario
        for scenario in ['baseline', 'management', 'reforestation']:
            if scenario in results.carbon_pools:
                plotter.plot_carbon_pools_breakdown(results.carbon_pools, scenario)
        
        # Plot final pools comparison
        if results.final_pools:
            plotter.plot_carbon_pools_comparison(results.final_pools)
        
        # Plot economics
        for scenario, cashflow in results.economics.items():
            plotter.plot_economics(cashflow, scenario)
        
        print(f"Plots saved to {self.output_dir}/plots/")
    
    def run(self, generate_plots: bool = False):
        """
        Run complete simulation.
        
        Args:
            generate_plots: Whether to generate visualization plots
        """
        # Validate simulation parameters
        assert self.years > 0, f"Simulation years must be positive, got {self.years}"
        assert self.area_ha > 0, f"Project area must be positive, got {self.area_ha}"
        assert self.forest_type, "Forest type must be specified"
        
        print(f"Starting Forest Carbon Simulation")
        print(f"Forest Type: {self.forest_type}")
        print(f"Duration: {self.years} years")
        print(f"Area: {self.area_ha} hectares")
        print("-" * 50)
        
        # Run scenarios
        scenario_results = self.run_all_scenarios()
        assert scenario_results, "Scenario results cannot be empty"
        
        # Compile results
        results = self.compile_results(scenario_results)
        assert results is not None, "Compiled results cannot be None"
        
        # Save results
        self.save_results(results)
        
        # Generate plots if requested
        if generate_plots:
            print("Generating plots...")
            self.generate_plots(results)
        
        # Run uncertainty analysis if enabled
        if self.enable_uncertainty:
            print("Running uncertainty analysis...")
            self.run_uncertainty_analysis(results)
        
        print("-" * 50)
        print("Simulation complete!")
        
        # Print summary
        print("\nSummary Results:")
        print(results.summary.to_string(index=False))
        
        return results
    
    def run_uncertainty_analysis(self, results):
        """Run uncertainty analysis for the climate scenario."""
        try:
            from .uncertainty_analysis import GrowthCarbonUncertainty
            
            # Determine the climate scenario name from the config file
            if self.config_file_used:
                # Extract scenario name from config file path
                config_path = Path(self.config_file_used)
                scenario_name = config_path.stem  # Get filename without extension
            else:
                # Use default scenario name
                scenario_name = f"{self.forest_type}_default"
            
            print(f"Running uncertainty analysis for climate scenario: {scenario_name}")
            
            # Use baseline scenario data as the base for parameter distributions
            baseline_data = results.summary[results.summary['scenario'] == 'baseline'].iloc[0]
            
            # Create uncertainty subdirectory within the years folder
            uncertainty_dir = self.output_dir / "uncertainty"
            uncertainty_analyzer = GrowthCarbonUncertainty(uncertainty_dir, scenario_name, seed=self.seed)
            
            # Run uncertainty analysis for this climate scenario
            uncertainty_results = uncertainty_analyzer.run_uncertainty_analysis(
                baseline_data.to_dict(),
                n_runs=500
            )
            
            print(f"Uncertainty analysis completed for climate scenario: {scenario_name}")
            print(f"Uncertainty plots saved to: {uncertainty_dir}/")
            
        except ImportError as e:
            print(f"Warning: Could not import uncertainty analysis module: {e}")
        except Exception as e:
            print(f"Warning: Uncertainty analysis failed: {e}")
