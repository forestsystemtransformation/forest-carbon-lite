# Configuration Validation Guide

This guide explains the new configuration validation system that prevents "silent foot-guns" by catching invalid parameters early.

## Overview

The validation system uses **Pydantic** to provide comprehensive type checking, bounds validation, and custom business logic validation for all configuration parameters.

## Key Benefits

✅ **Prevents Silent Failures**: Catches invalid parameters before simulation starts  
✅ **Clear Error Messages**: Provides specific field-level error details  
✅ **Type Safety**: Ensures parameters are correct types  
✅ **Bounds Checking**: Validates realistic parameter ranges  
✅ **Enum Validation**: Restricts forest types to valid options  
✅ **Backward Compatibility**: Can be disabled for legacy behavior  

## Quick Start

### Basic Usage

```python
from simulator import ForestCarbonSimulator

# With validation (default)
simulator = ForestCarbonSimulator("EOF", years=25, validate_config=True)

# Without validation (legacy)
simulator = ForestCarbonSimulator("EOF", years=25, validate_config=False)
```

### Custom Configuration

```python
from utils.validation import SiteConfig, EconomicsConfig

# Create validated configuration
site_config = SiteConfig(
    forest_type="EOF",
    K_AGB=180.0,
    G=12.0,
    root_shoot=0.30,
    # ... other parameters
)

# Validate economics
economics_config = EconomicsConfig(
    carbon={
        'price_start': 45.0,
        'price_growth': 0.04,
        'buffer': 0.15,
        'crediting_years': 35,
        'discount_rate': 0.06
    },
    costs={
        'management': {'capex': 2500.0, 'opex': 180.0, 'mrv': 20.0},
        'reforestation': {'capex': 6000.0, 'opex': 250.0, 'mrv': 40.0}
    }
)
```

## Validation Rules

### Forest Types
- **Valid**: `EOF`, `ETOF`, `AFW`, `Rainforest`, `Mallee`, `Shrubland`, `Other_FW`, `EW_OW`
- **Invalid**: Any other string

### Biomass Parameters
- **K_AGB**: `> 0, ≤ 1000` tonnes/ha
- **M (TYF)**: `> 0, ≤ 1000` tonnes/ha
- **G (TYF)**: `> 0, ≤ 100` years
- **y (TYF)**: `> 0, ≤ 5`

### Probabilities and Rates
- **Mortality rates** (`m_*`): `0 ≤ rate ≤ 0.5`
- **Disturbance probabilities** (`pdist_*`): `0 ≤ prob ≤ 1`
- **Disturbance severity** (`dsev_*`): `0 ≤ severity ≤ 1`
- **Root-shoot ratio**: `0.05 ≤ ratio ≤ 1.0`

### Ages
- **Initial ages**: `0 ≤ age ≤ 500` years
- **Reforestation age**: `0 ≤ age ≤ 100` years (typically 0)

### Economics
- **Carbon price**: `≥ 0` $/tCO2e
- **Price growth**: `0 ≤ growth ≤ 1`
- **Buffer**: `0 ≤ buffer ≤ 1`
- **Discount rate**: `0 ≤ rate ≤ 1`
- **Crediting years**: `≥ 1`
- **All costs**: `≥ 0`

## Error Examples

### Invalid Forest Type
```python
# This will raise ValidationError
config = SiteConfig(forest_type="INVALID_TYPE", ...)
# Error: forest_type
#   Input should be 'EOF', 'ETOF', 'AFW', 'Rainforest', 'Mallee', 'Shrubland', 'Other_FW' or 'EW_OW'
```

### Negative Biomass
```python
# This will raise ValidationError
config = SiteConfig(K_AGB=-50.0, ...)
# Error: K_AGB
#   Input should be greater than 0
```

### Invalid Probability
```python
# This will raise ValidationError
config = SiteConfig(m_degraded=1.5, ...)
# Error: m_degraded
#   Input should be less than or equal to 0.5
```

### Missing Required Scenarios
```python
# This will raise ValidationError
config = SiteConfig(
    tyf_calibrations={
        'baseline': {'M': 160.0, 'G': 10.0, 'y': 1.0}
        # Missing 'management' and 'reforestation'
    },
    ...
)
# Error: tyf_calibrations
#   Missing required scenarios: {'management', 'reforestation'}
```

## Advanced Features

### Custom Validators

The system includes custom validators for business logic:

```python
@validator('M')
def validate_M_realistic(cls, v):
    """Ensure M is within realistic biomass range."""
    if v < 10:
        raise ValueError("Maximum biomass (M) should be at least 10 tonnes/ha")
    if v > 500:
        logger.warning(f"Maximum biomass (M={v}) seems unusually high")
    return v
```

### Cross-Field Validation

```python
@root_validator
def validate_biomass_consistency(cls, values):
    """Validate biomass values are consistent with K_AGB."""
    K_AGB = values.get('K_AGB')
    initial_baseline = values.get('initial_biomass_baseline')
    
    if K_AGB and initial_baseline and initial_baseline > K_AGB:
        raise ValueError(f"Initial baseline biomass ({initial_baseline}) exceeds maximum potential ({K_AGB})")
    
    return values
```

### Warnings for Unusual Values

The system provides warnings (not errors) for unusual but potentially valid values:

```python
@validator('root_shoot')
def validate_root_shoot_realistic(cls, v):
    if v < 0.1:
        logger.warning(f"Root-shoot ratio ({v}) seems unusually low")
    if v > 0.6:
        logger.warning(f"Root-shoot ratio ({v}) seems unusually high")
    return v
```

## File-Based Configuration

### Loading with Validation

```python
from utils.config_loader import ConfigLoader

# With validation (default)
loader = ConfigLoader(validate=True)
config = loader.load_site_config("EOF")

# Without validation (legacy)
loader = ConfigLoader(validate=False)
config = loader.load_site_config("EOF")
```

### Custom Config Files

```python
# Load custom config file with validation
config = loader.load_site_config("EOF", "custom_config.yaml")
```

## Testing

Run the validation tests:

```bash
python test_validation.py
```

Run the validation example:

```bash
python validation_example.py
```

## Migration Guide

### For Existing Code

1. **No changes required** - validation is enabled by default but backward compatible
2. **To disable validation** (if needed): `validate_config=False`
3. **To enable validation** (explicit): `validate_config=True`

### For New Code

1. **Always use validation** for new configurations
2. **Create configs programmatically** using Pydantic models
3. **Validate custom configs** before saving to files

## Performance

- **Validation overhead**: ~1-2ms per configuration load
- **Memory overhead**: Minimal (Pydantic models are lightweight)
- **Startup time**: Negligible impact

## Troubleshooting

### Common Issues

1. **ImportError**: Install pydantic: `pip install pydantic>=2.0.0`
2. **ValidationError**: Check parameter bounds and types
3. **Missing scenarios**: Ensure all required TYF scenarios are present

### Debugging

Enable detailed error messages:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Disabling Validation

If you encounter issues, you can temporarily disable validation:

```python
simulator = ForestCarbonSimulator("EOF", validate_config=False)
```

## Future Enhancements

- **JSON Schema export** for configuration documentation
- **Configuration templates** for common forest types
- **Interactive validation** in CLI tools
- **Configuration diffing** for change tracking
