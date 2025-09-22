"""Configuration validation schemas using Pydantic."""

from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
from typing import Dict, Optional, Literal, Union
import logging
from .guardrails import (
    validate_fpi, validate_y_multiplier, validate_mortality_rate,
    validate_disturbance_probability, validate_disturbance_severity,
    validate_compound_effects, get_forest_limits
)

logger = logging.getLogger(__name__)

class ForestType(str):
    """Custom forest type that validates against available site configurations."""
    
    def __new__(cls, value):
        return str.__new__(cls, value)
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v, field=None):
        if not isinstance(v, str):
            raise TypeError('Forest type must be a string')
        
        # Check if site config file exists
        from pathlib import Path
        config_dir = Path("configs/base")
        site_config_path = config_dir / f"site_{v}.yaml"
        
        if not site_config_path.exists():
            # List available forest types for better error message
            available_configs = []
            for config_file in config_dir.glob("site_*.yaml"):
                forest_type = config_file.stem.replace("site_", "")
                available_configs.append(forest_type)
            
            available_str = ", ".join(sorted(available_configs))
            raise ValueError(
                f"Forest type '{v}' not found. Available forest types: {available_str}. "
                f"Create a site_{v}.yaml file in configs/base/ to add a new forest type."
            )
        
        return cls(v)
    
    @property
    def value(self):
        """Return the string value for compatibility with Pydantic."""
        return str(self)

class ScenarioType(str, Enum):
    """Supported scenario types."""
    BASELINE = "baseline"
    MANAGEMENT = "management"
    REFORESTATION = "reforestation"

class FPIBaseConfig(BaseModel):
    """Base FPI configuration with default value."""
    default: float = Field(
        ge=0.1,
        le=2.0,
        description="Default FPI ratio"
    )

class FPITimeVaryingConfig(FPIBaseConfig):
    """FPI configuration with time-varying values."""
    # Dynamic year-specific values (e.g., year_10: 0.83)
    # We'll use a flexible approach to handle any year_X pattern
    
    class Config:
        extra = "allow"  # Allow extra fields for year_X attributes
    
    @model_validator(mode='after')
    def validate_year_specific_values(self):
        """Validate year-specific FPI values."""
        # Get all attributes that match year_X pattern
        year_attrs = {k: v for k, v in self.__dict__.items() if k.startswith('year_')}
        
        for attr_name, value in year_attrs.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"FPI value for {attr_name} must be numeric")
            if not 0.1 <= value <= 2.0:
                raise ValueError(f"FPI value for {attr_name} ({value}) must be between 0.1 and 2.0")
        
        return self

class FPIRatiosConfig(BaseModel):
    """FPI ratios configuration for all scenarios."""
    baseline: Union[float, FPITimeVaryingConfig] = Field(
        description="FPI ratio for baseline scenario (can be constant or time-varying)"
    )
    management: float = Field(
        ge=0.1,
        le=2.0,
        description="FPI ratio for management scenario"
    )
    reforestation: float = Field(
        ge=0.1,
        le=2.0,
        description="FPI ratio for reforestation scenario"
    )
    
    @field_validator('baseline', mode='before')
    @classmethod
    def validate_baseline_fpi(cls, v):
        """Validate baseline FPI configuration using scientifically defensible guardrails."""
        if isinstance(v, (int, float)):
            # Use guardrails to validate and potentially adjust the value
            validated_fpi, was_adjusted = validate_fpi(float(v))
            if was_adjusted:
                logger.warning(f"Baseline FPI adjusted from {v} to {validated_fpi}")
            return validated_fpi
        elif isinstance(v, dict):
            # Convert dict to FPITimeVaryingConfig
            return FPITimeVaryingConfig(**v)
        else:
            raise ValueError("Baseline FPI must be a number or time-varying configuration")
    
    @field_validator('management', 'reforestation')
    @classmethod
    def validate_scenario_fpi(cls, v):
        """Validate scenario FPI ratios using scientifically defensible guardrails."""
        # Use guardrails to validate and potentially adjust the value
        validated_fpi, was_adjusted = validate_fpi(float(v))
        if was_adjusted:
            logger.warning(f"Scenario FPI adjusted from {v} to {validated_fpi}")
        return validated_fpi

class TYFCalibration(BaseModel):
    """TYF (Tree Yield Formula) calibration parameters."""
    M: float = Field(
        gt=0, 
        le=1000, 
        description="Maximum potential biomass (tonnes/ha)"
    )
    G: float = Field(
        gt=0, 
        le=100, 
        description="Age of maximum growth rate (years)"
    )
    y: float = Field(
        gt=0, 
        le=5, 
        description="Growth multiplier factor"
    )
    
    @field_validator('M')
    @classmethod
    def validate_M_realistic(cls, v):
        """Ensure M is within realistic biomass range."""
        if v < 10:
            raise ValueError("Maximum biomass (M) should be at least 10 tonnes/ha")
        if v > 500:
            logger.warning(f"Maximum biomass (M={v}) seems unusually high for most forest types")
        return v
    
    @field_validator('G')
    @classmethod
    def validate_G_realistic(cls, v):
        """Ensure G is within realistic age range."""
        if v < 1:
            raise ValueError("Age of maximum growth (G) should be at least 1 year")
        if v > 50:
            logger.warning(f"Age of maximum growth (G={v}) seems unusually high")
        return v
    
    @field_validator('y')
    @classmethod
    def validate_y_realistic(cls, v):
        """Validate Y multiplier using scientifically defensible guardrails."""
        # Use guardrails to validate and potentially adjust the value
        validated_y, was_adjusted = validate_y_multiplier(float(v))
        if was_adjusted:
            logger.warning(f"Y multiplier adjusted from {v} to {validated_y}")
        return validated_y

class EconomicsConfig(BaseModel):
    """Economic configuration parameters."""
    carbon: Dict[str, float] = Field(
        description="Carbon pricing and crediting parameters"
    )
    costs: Dict[str, Dict[str, float]] = Field(
        description="Cost parameters for different scenarios"
    )
    
    @field_validator('carbon')
    @classmethod
    def validate_carbon_params(cls, v):
        """Validate carbon pricing parameters."""
        required_keys = {'price_start', 'price_growth', 'buffer', 'crediting_years', 'discount_rate'}
        missing_keys = required_keys - set(v.keys())
        if missing_keys:
            raise ValueError(f"Missing required carbon parameters: {missing_keys}")
        
        # Validate individual parameters
        if v['price_start'] < 0:
            raise ValueError("Carbon price must be non-negative")
        if not 0 <= v['price_growth'] <= 1:
            raise ValueError("Price growth rate should be between 0 and 1")
        if not 0 <= v['buffer'] <= 1:
            raise ValueError("Buffer should be between 0 and 1")
        if v['crediting_years'] < 1:
            raise ValueError("Crediting years must be at least 1")
        if not 0 <= v['discount_rate'] <= 1:
            raise ValueError("Discount rate should be between 0 and 1")
        
        return v
    
    @field_validator('costs')
    @classmethod
    def validate_costs(cls, v):
        """Validate cost parameters."""
        required_scenarios = {'management', 'reforestation'}
        missing_scenarios = required_scenarios - set(v.keys())
        if missing_scenarios:
            raise ValueError(f"Missing required cost scenarios: {missing_scenarios}")
        
        for scenario, costs in v.items():
            required_cost_types = {'capex', 'opex', 'mrv'}
            missing_cost_types = required_cost_types - set(costs.keys())
            if missing_cost_types:
                raise ValueError(f"Missing cost types for {scenario}: {missing_cost_types}")
            
            for cost_type, value in costs.items():
                if value < 0:
                    raise ValueError(f"{cost_type} for {scenario} must be non-negative")
        
        return v

class SiteConfig(BaseModel):
    """Site configuration with comprehensive validation."""
    
    # Forest type and basic parameters
    forest_type: Optional[ForestType] = Field(
        description="Forest type identifier"
    )
    K_AGB: float = Field(
        gt=0, 
        le=1000, 
        description="Maximum AGB potential (tonnes/ha)"
    )
    G: float = Field(
        gt=0, 
        le=100, 
        description="Age of maximum growth rate (years)"
    )
    root_shoot: float = Field(
        ge=0.05, 
        le=1.0, 
        description="Root-to-shoot ratio"
    )
    
    # Initial soil carbon fractions
    S0_deg_frac: float = Field(
        ge=0, 
        le=1, 
        description="Initial soil carbon fraction for degraded forest"
    )
    S0_man_frac: float = Field(
        ge=0, 
        le=1, 
        description="Initial soil carbon fraction for managed forest"
    )
    S0_new_frac: float = Field(
        ge=0, 
        le=1, 
        description="Initial soil carbon fraction for reforestation"
    )
    
    # Initial ages
    age_degraded: int = Field(
        ge=0, 
        le=500, 
        description="Initial age of degraded forest (years)"
    )
    age_managed: int = Field(
        ge=0, 
        le=500, 
        description="Initial age of managed forest (years)"
    )
    age_reforestation: int = Field(
        ge=0, 
        le=100, 
        description="Initial age of reforestation (years)"
    )
    
    # Initial biomass values
    initial_biomass_baseline: float = Field(
        ge=0, 
        description="Initial biomass for baseline scenario (tonnes/ha)"
    )
    initial_biomass_management: float = Field(
        ge=0, 
        description="Initial biomass for management scenario (tonnes/ha)"
    )
    initial_biomass_reforestation: float = Field(
        ge=0, 
        description="Initial biomass for reforestation scenario (tonnes/ha)"
    )
    
    # Mortality rates (annual)
    m_degraded: float = Field(
        ge=0, 
        le=0.5, 
        description="Annual mortality rate for degraded forest"
    )
    m_managed: float = Field(
        ge=0, 
        le=0.5, 
        description="Annual mortality rate for managed forest"
    )
    m_reforestation: float = Field(
        ge=0, 
        le=0.5, 
        description="Annual mortality rate for reforestation"
    )
    
    # Disturbance probabilities (annual)
    pdist_degraded: float = Field(
        ge=0, 
        le=1, 
        description="Annual disturbance probability for degraded forest"
    )
    pdist_managed: float = Field(
        ge=0, 
        le=1, 
        description="Annual disturbance probability for managed forest"
    )
    pdist_reforestation: float = Field(
        ge=0, 
        le=1, 
        description="Annual disturbance probability for reforestation"
    )
    
    # Disturbance severity
    dsev_degraded: float = Field(
        ge=0, 
        le=1, 
        description="Disturbance severity for degraded forest"
    )
    dsev_managed: float = Field(
        ge=0, 
        le=1, 
        description="Disturbance severity for managed forest"
    )
    dsev_reforestation: float = Field(
        ge=0, 
        le=1, 
        description="Disturbance severity for reforestation"
    )
    
    # TYF calibrations
    tyf_calibrations: Dict[str, TYFCalibration] = Field(
        description="TYF calibration parameters for each scenario"
    )
    
    # FPI ratios for climate-aware projections
    fpi_ratios: Optional[FPIRatiosConfig] = Field(
        default=None,
        description="FPI ratios for climate-aware projections"
    )
    
    # Economics configuration
    economics: Optional[EconomicsConfig] = Field(
        default=None,
        description="Economic parameters"
    )
    
    # Optional FullCAM parameters
    fullcam_parameters: Optional[Dict] = Field(
        default=None,
        description="FullCAM-specific parameters"
    )
    
    # Optional model performance metrics
    model_performance: Optional[Dict] = Field(
        default=None,
        description="Model performance metrics from FullCAM calibration"
    )
    
    # Optional disturbance coverage data
    disturbance_coverage: Optional[Dict] = Field(
        default=None,
        description="Disturbance history coverage from calibration dataset"
    )
    
    # Optional typical carbon stocks
    typical_carbon_stocks: Optional[Dict] = Field(
        default=None,
        description="Typical carbon stocks from FullCAM calibration dataset"
    )
    
    @field_validator('tyf_calibrations')
    @classmethod
    def validate_tyf_scenarios(cls, v):
        """Validate that all required TYF scenarios are present."""
        required_scenarios = {ScenarioType.BASELINE, ScenarioType.MANAGEMENT, ScenarioType.REFORESTATION}
        present_scenarios = set(v.keys())
        missing_scenarios = required_scenarios - present_scenarios
        
        if missing_scenarios:
            raise ValueError(f"Missing required TYF scenarios: {missing_scenarios}")
        
        return v
    
    @field_validator('root_shoot')
    @classmethod
    def validate_root_shoot_realistic(cls, v):
        """Validate root-shoot ratio is realistic."""
        if v < 0.1:
            logger.warning(f"Root-shoot ratio ({v}) seems unusually low for most forest types")
        if v > 0.6:
            logger.warning(f"Root-shoot ratio ({v}) seems unusually high for most forest types")
        return v
    
    @field_validator('m_degraded', 'm_managed', 'm_reforestation')
    @classmethod
    def validate_mortality_rates(cls, v):
        """Validate mortality rates using scientifically defensible guardrails."""
        # Use guardrails to validate and potentially adjust the value
        validated_m, was_adjusted = validate_mortality_rate(float(v))
        if was_adjusted:
            logger.warning(f"Mortality rate adjusted from {v} to {validated_m}")
        return validated_m
    
    @field_validator('pdist_degraded', 'pdist_managed', 'pdist_reforestation')
    @classmethod
    def validate_disturbance_probabilities(cls, v):
        """Validate disturbance probabilities using scientifically defensible guardrails."""
        # Use guardrails to validate and potentially adjust the value
        validated_pdist, was_adjusted = validate_disturbance_probability(float(v))
        if was_adjusted:
            logger.warning(f"Disturbance probability adjusted from {v} to {validated_pdist}")
        return validated_pdist
    
    @field_validator('dsev_degraded', 'dsev_managed', 'dsev_reforestation')
    @classmethod
    def validate_disturbance_severities(cls, v):
        """Validate disturbance severities using scientifically defensible guardrails."""
        # Use guardrails to validate and potentially adjust the value
        validated_dsev, was_adjusted = validate_disturbance_severity(float(v))
        if was_adjusted:
            logger.warning(f"Disturbance severity adjusted from {v} to {validated_dsev}")
        return validated_dsev
    
    @model_validator(mode='after')
    def validate_biomass_consistency(self):
        """Validate biomass values are consistent with K_AGB."""
        if self.initial_biomass_baseline > self.K_AGB:
            raise ValueError(f"Initial baseline biomass ({self.initial_biomass_baseline}) exceeds maximum potential ({self.K_AGB})")
        
        if self.initial_biomass_management > self.K_AGB:
            raise ValueError(f"Initial management biomass ({self.initial_biomass_management}) exceeds maximum potential ({self.K_AGB})")
        
        return self
    
    @model_validator(mode='after')
    def validate_age_consistency(self):
        """Validate age values are reasonable."""
        # Reforestation should typically start at age 0
        if self.age_reforestation > 5:
            logger.warning(f"Reforestation age ({self.age_reforestation}) is unusually high - typically starts at 0")
        
        return self
    
    @model_validator(mode='after')
    def validate_compound_effects(self):
        """Validate compound effects to prevent unrealistic growth modifiers."""
        forest_type = self.forest_type.value if self.forest_type else None
        
        # Validate compound effects for each scenario
        for scenario_name, tyf_calibration in self.tyf_calibrations.items():
            # Get FPI ratio for this scenario
            if self.fpi_ratios:
                if scenario_name == 'baseline':
                    fpi = self.fpi_ratios.baseline if isinstance(self.fpi_ratios.baseline, (int, float)) else 1.0
                elif scenario_name == 'management':
                    fpi = self.fpi_ratios.management
                elif scenario_name == 'reforestation':
                    fpi = self.fpi_ratios.reforestation
                else:
                    fpi = 1.0
            else:
                fpi = 1.0
            
            # Validate compound effects
            validated_fpi, validated_y, was_adjusted = validate_compound_effects(
                fpi, tyf_calibration.y, forest_type
            )
            
            if was_adjusted:
                logger.warning(
                    f"Compound effects adjusted for {scenario_name} scenario: "
                    f"FPI {fpi:.3f} -> {validated_fpi:.3f}, Y {tyf_calibration.y:.3f} -> {validated_y:.3f}"
                )
                # Update the values
                tyf_calibration.y = validated_y
        
        return self

class ValidatedConfigLoader:
    """Configuration loader with validation."""
    
    def __init__(self, config_dir: str = "configs/base"):
        """Initialize with configuration directory."""
        self.config_dir = config_dir
    
    def load_and_validate_site_config(self, forest_type: str, config_file: Optional[str] = None) -> SiteConfig:
        """
        Load and validate site configuration.
        
        Args:
            forest_type: Forest type identifier
            config_file: Optional custom config file path
            
        Returns:
            Validated SiteConfig object
            
        Raises:
            ValidationError: If configuration validation fails
            FileNotFoundError: If configuration file not found
        """
        import yaml
        from pathlib import Path
        
        # Determine config file path
        if config_file:
            if Path(config_file).is_absolute():
                config_path = Path(config_file)
            elif "/" in config_file or "\\" in config_file:
                # Config file already contains a path, use it as-is
                config_path = Path(config_file)
            else:
                # Simple filename, prepend config_dir
                config_path = Path(self.config_dir) / config_file
        else:
            config_path = Path(self.config_dir) / f"site_{forest_type}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load YAML
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Handle generated configuration files (they have scenario_metadata and scenario_components)
        if 'scenario_metadata' in config_dict:
            # This is a generated config file, extract the actual configuration
            # The configuration parameters are at the top level after the metadata sections
            actual_config = {}
            for key, value in config_dict.items():
                if key not in ['scenario_metadata', 'scenario_components']:
                    actual_config[key] = value
            
            # Add forest_type from metadata if not present
            if 'forest_type' not in actual_config:
                actual_config['forest_type'] = config_dict['scenario_metadata'].get('forest_type', forest_type)
            
            config_dict = actual_config
        else:
            # This is a base site configuration file
            # Add forest_type if not present
            if 'forest_type' not in config_dict:
                config_dict['forest_type'] = forest_type
        
        # Validate and return
        return SiteConfig(**config_dict)
    
    def load_and_validate_economics_config(self, config_file: str = "economics_default.yaml") -> EconomicsConfig:
        """
        Load and validate economics configuration.
        
        Args:
            config_file: Economics config file name
            
        Returns:
            Validated EconomicsConfig object
        """
        import yaml
        from pathlib import Path
        
        config_path = Path("configs/base") / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Economics configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return EconomicsConfig(**config_dict)

# Convenience function for validation
def validate_config(config_dict: Dict) -> SiteConfig:
    """Validate a configuration dictionary."""
    return SiteConfig(**config_dict)