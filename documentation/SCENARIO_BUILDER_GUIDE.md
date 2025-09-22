# Forest Carbon Scenario Builder System

## Overview

The Scenario Builder System solves the problem of managing multiple static configuration files by providing a dynamic, maintainable approach to creating forest carbon scenarios.

## Key Features

✅ **Dynamic Configuration Generation**: No more managing hundreds of static config files  
✅ **Comprehensive Documentation**: Each generated config includes detailed explanations  
✅ **Transparent Parameter Modifications**: Clear tracking of how parameters are modified  
✅ **Parallel Processing**: Run multiple scenarios efficiently  
✅ **Integrated Analysis**: Automatic comparison and visualization  
✅ **Research-Based Parameters**: Climate and management effects calibrated to literature findings  
✅ **Climate Threshold Warnings**: Alerts about critical ecosystem limits  
✅ **Your Preferred Output**: Uses `output_years` directory as specified  
✅ **Fixed Validation Issues**: All scenarios now run successfully without configuration errors  

## Quick Start

### 1. Run Complete Analysis (Recommended)

```bash
# Run all scenarios with comprehensive analysis
python run_scenario_analysis.py
```

This will:
- Generate 27 scenarios (3 forest types × 3 climates × 3 management levels)
- Run all simulations in parallel
- Generate comprehensive analysis and plots
- Save results to `output_years/` directory

### 2. Run Specific Scenarios

```bash
# Single forest type with all climates and managements
python run_scenario_analysis.py --forest-types ETOF

# Specific climate scenarios
python run_scenario_analysis.py --climates normal,paris2050

# Custom combination
python run_scenario_analysis.py --forest-types ETOF,EOF --climates paris2050 --managements moderate,intensive
```

### 3. View Results

```bash
# Check generated outputs
ls output_years/

# View analysis results
ls output_years/analysis/
```

## Recent Fixes and Improvements

### Validation Error Fix (Latest)
**Issue**: Scenarios were failing with validation errors:
```
1 validation error for SiteConfig
fpi_ratios.baseline
  Field required [type=missing, input_value={'baseline degraded': 0.9...}]
```

**Root Cause**: The scenario builder was incorrectly using `"baseline degraded"` as the key for FPI ratios instead of `"baseline"`.

**Fix Applied**: Updated the scenario builder to use the correct key name:
```python
# Before (incorrect):
'baseline degraded': base_fpi * 0.9,

# After (correct):
'baseline': base_fpi * 0.9,
```

**Result**: All scenarios now run successfully without validation errors.

### Configuration Loading
- **External Config Priority**: The scenario builder loads from external site config files (e.g., `config/site_ETOF.yaml`) when available
- **Fallback to Built-ins**: Uses internal defaults only when external files are missing
- **Validation Schema**: All generated configurations are validated against Pydantic schemas to prevent runtime errors

## Research-Based Parameter Updates

The scenario builder now incorporates findings from forest model calibration literature:

### Climate Effects
- **Paris2050**: 12% FPI reduction (research: 10-15% for 1.5°C warming)
- **Hot_Dry**: 30% FPI reduction (research: 25-35% for 2.5-3°C warming)
- **Mortality**: Already increased 67% under current warming trends
- **Critical Thresholds**: Warnings for 1.5°C (carbon sink to source) and 2.5°C+ (ecosystem collapse)

### Forest Parameters
- **AFW**: Corrected M parameter to 49.0 t DM/ha (research: 48.5-49.5 t DM/ha)
- **ETOF**: M parameter 300 t DM/ha (research: 234-351 t DM/ha range)
- **EOF**: M parameter 160 t DM/ha (research: 115-225 t DM/ha range)

### Management Effects
- **Prescribed Burning**: 9% biomass consumption, 62-72% fire severity reduction
- **Thinning**: 50% carbon removal but significant fire risk reduction
- **No Double Counting**: FPI enhancements removed to avoid parameter inflation

### Mortality Parameter Structure (No Double-Counting)
The system now uses a clean separation of mortality effects to prevent double-counting:

**Base Config Files:**
```yaml
# Each forest type has a single baseline mortality rate
m_baseline: 0.015  # ETOF: Lower mortality (mature forest)
m_baseline: 0.018  # EOF: Medium mortality  
m_baseline: 0.022  # AFW: Higher mortality (arid conditions)
```

**Scenario Builder Effects:**
```yaml
# All effects applied consistently:
m_degraded: m_baseline × climate_adjustment
m_managed: m_baseline × climate_adjustment × management_factor  
m_reforestation: m_baseline × climate_adjustment × 1.1
```

**Effect Separation:**
- **Climate Effects**: mortality_adjustment (1.0 normal, 1.15 Paris2050, 1.67 Hot_Dry)
- **Management Effects**: mortality_factor (1.0 baseline, 0.92 light, 0.75 moderate, 0.50 intensive)
- **Young Tree Stress**: 1.1× multiplier for reforestation establishment stress

## Configuration Naming Convention

Config files are named: `{forest}_{climate}_{years}_{management}.yaml`

Examples:
- `ETOF_paris2050_2025-2050_intensive.yaml`
- `EOF_hot_dry_2025-2050_moderate.yaml`
- `AFW_normal_2025-2050_baseline.yaml`

## Scenario Components

### Forest Types
- **ETOF**: Eucalypt Tall Open Forest (high biomass potential)
- **EOF**: Eucalypt Open Forest (medium biomass potential)  
- **AFW**: Acacia Forest & Woodland (moderate biomass potential)

### Climate Scenarios
- **normal**: Default climate with no adjustments
- **paris2050**: Moderate warming scenario (5% productivity reduction)
- **hot_dry**: Significant warming and drying (20% productivity reduction)

### Management Levels
- **baseline**: No management (degraded forest)
- **light**: Minimal management (10% growth improvement)
- **moderate**: Regular management (25% growth improvement)
- **intensive**: Heavy management (40% growth improvement)

### Time Periods
- **2025-2030**: 5 years
- **2025-2050**: 25 years (default)
- **2025-2100**: 75 years

## Generated Config Structure

Each generated config includes:

```yaml
scenario_metadata:
  name: ETOF_paris2050_2025-2050_intensive
  description: ETOF forest under paris2050 climate with intensive management
  forest_type: ETOF
  climate_scenario: paris2050
  management_level: intensive
  time_period: 2025-2050
  simulation_years: 25
  created: '2025-09-18T16:31:50.006846'
  created_by: scenario_builder.py

scenario_interpretation:
  forest_type_meaning: Eucalypt Tall Open Forest - High density eucalypt forest with high biomass potential
  climate_meaning: Paris 2050 climate - Moderate warming scenario with some productivity impacts
  management_meaning: Intensive management - Heavy interventions with maximum improvements
  parameter_modifications:
    # Detailed explanations of all parameter modifications

# Actual configuration parameters
K_AGB: 350.0
fpi_ratios:
  baseline: 0.855
  management: 1.092
  reforestation: 0.969
# ... etc
```

## Parameter Modifications

### Climate Effects
- **FPI Adjustment**: Multiplies forest productivity
- **Mortality Adjustment**: Increases mortality rates
- **Disturbance Adjustment**: Increases disturbance probability and severity

### Management Effects  
- **Growth Multiplier**: Enhances TYF y parameter
- **Mortality Reduction**: Reduces mortality rates
- **Disturbance Reduction**: Reduces disturbance probability and severity
- **FPI Enhancement**: Improves forest productivity

## Usage Examples

### Single Scenario
```python
from scenario_builder import ScenarioBuilder, ForestType, ClimateScenario, ManagementLevel

scenario = ScenarioBuilder()\
    .set_forest_type(ForestType.ETOF)\
    .set_climate(ClimateScenario.PARIS2050)\
    .set_management(ManagementLevel.INTENSIVE)\
    .set_time_period("2025-2050")\
    .build()

config_path = scenario.save()
```

### Batch Generation
```python
from scenario_builder import ScenarioGenerator

generator = ScenarioGenerator()
scenarios = generator.generate_matrix(
    forest_types=[ForestType.ETOF, ForestType.EOF],
    climates=[ClimateScenario.NORMAL, ClimateScenario.HOT_DRY],
    managements=[ManagementLevel.BASELINE, ManagementLevel.INTENSIVE],
    time_periods=["2025-2050"]
)

manifest = generator.save_all_scenarios(scenarios)
```

### Batch Execution
```python
from batch_runner import BatchRunner

runner = BatchRunner(n_workers=4)
results = runner.run_from_manifest("configs/scenarios/scenario_manifest.csv")
```

## Output Structure

```
output_years/
├── batch_results.csv              # Compiled results from all scenarios
├── analysis/
│   ├── comprehensive_analysis.png # Complete visualization suite
│   ├── scenario_analysis_report.md # Detailed analysis report
│   └── statistical_analysis.json  # Statistical test results
└── [scenario_name]/               # Individual scenario outputs
    ├── results_summary.csv
    ├── simulation_summary.json
    └── plots/
```

## Key Benefits

1. **No File Proliferation**: Generate scenarios dynamically instead of managing hundreds of files
2. **Transparent Modifications**: Every parameter change is documented and explained
3. **Consistent Naming**: Clear, systematic naming convention
4. **Reproducible**: All scenarios are generated deterministically
5. **Scalable**: Easy to add new forest types, climates, or management levels
6. **Integrated**: Works seamlessly with existing CLI and simulator
7. **Your Preferences**: Uses `output_years` directory and your preferred structure

## Troubleshooting

### Config File Not Found
```bash
# Copy generated config to config/ directory
copy "configs\scenarios\[config_name].yaml" "config\[config_name].yaml"
```

### Missing Dependencies
```bash
pip install pandas matplotlib seaborn tqdm
```

### Memory Issues with Large Batches
```bash
# Reduce number of workers
python run_scenario_analysis.py --workers 2
```

## Next Steps

1. **Run your first batch**: `python run_scenario_analysis.py`
2. **Customize scenarios**: Modify the builder to add your specific requirements
3. **Add new forest types**: Extend the ForestType enum and base configs
4. **Create custom analyses**: Use the results for your specific research questions

The scenario builder system gives you a clean, maintainable, and scalable approach to forest carbon scenario analysis!
