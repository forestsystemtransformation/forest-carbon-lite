# Forest Carbon Lite V.1.0

A comprehensive Python implementation of a "FullCAM-lite" forest carbon accounting model, featuring the Tree Yield Formula (TYF) growth engine with dynamic scenario generation and climate change integration.

```
forest-carbon-lite/
├── forest_carbon/                    # Main package
│   ├── __init__.py                  # Package initialization
│   ├── core/                        # Core simulation engine
│   │   ├── __init__.py
│   │   ├── simulator.py            # Main simulator
│   │   ├── uncertainty_analysis.py # Uncertainty analysis
│   │   └── models/                 # All models
│   │       ├── __init__.py
│   │       ├── tyf_engine.py       # Tree Yield Formula engine
│   │       ├── carbon_pools.py     # Carbon pool management
│   │       ├── disturbance.py      # Disturbance modeling
│   │       └── economics.py        # Economic modeling
│   ├── scenarios/                   # Scenario management
│   │   ├── __init__.py
│   │   ├── builder.py              # Scenario configuration builder
│   │   ├── runner.py               # Batch simulation runner
│   │   ├── analyzer.py             # Scenario analysis
│   │   └── manager.py              # Main scenario orchestrator
│   ├── analysis/                    # Analysis and reporting
│   │   ├── __init__.py
│   │   └── comprehensive.py        # Comprehensive analysis system
│   ├── utils/                       # Utilities and helpers
│   │   ├── __init__.py
│   │   ├── config_loader.py        # Configuration management
│   │   ├── validation.py           # Input validation
│   │   ├── colors.py               # Color management
│   │   ├── constants.py            # Physical constants
│   │   └── climate_adjustments.py  # Climate adjustment functions
│   └── visualization/               # Plotting and visualization
│       ├── __init__.py
│       └── plotter.py              # Main plotting system
├── main.py                         # Unified entry point
├── custom_afm_simulator.py         # AFM vs Degrading analysis tool
├── plot_matrix_generator.py        # Plot comparison matrix generator
├── data_matrix_generator.py        # Data-driven matrix generator
├── configs/                        # Configuration files
│   ├── base/                       # Base configurations
│   │   ├── management_*.yaml       # Management configs (i, ir, m, mr, l, afm_m)
│   │   ├── climate_*.yaml          # Climate configs (current, paris, plus2, plus3)
│   │   └── site_*.yaml             # Site configs (EOF, EOFD, ETOF, ETOFD)
│   └── generated/                  # Generated scenario configs
├── output/                         # Default output directory
├── output_matrix/                  # Matrix comparison outputs
└── requirements.txt                # Dependencies
```

## 📋 Simplified Configuration Names

For ease of use, configuration files now use shortened, intuitive names:

### **Management Configs** (`management_*.yaml`)
- **`i`** - Intensive AFM for Management
- **`ir`** - Intensive AFM for Management and Reforestation  
- **`m`** - Moderate AFM for Management
- **`mr`** - Moderate AFM for Management and Reforestation
- **`l`** - Light AFM for Management
- **`lr`** - Light AFM for Management and Reforestation
- **`afm_m`** - Moderate AFM Only (no reforestation)

### **Climate Configs** (`climate_*.yaml`)
- **`current`** - No climate change (baseline)
- **`paris`** - Paris target plus 1.5°C warming
- **`plus2`** - Plus 2°C warming
- **`plus3`** - Plus 3°C warming

### **Site Configs** (`site_*.yaml`)
- **`EOF`** - Eucalypt Open Forest
- **`EOFD`** - Eucalypt Open Forest Degraded
- **`ETOF`** - Eucalypt Tall Open Forest
- **`ETOFD`** - Eucalypt Tall Open Forest Degraded

## 📋 Available Configurations

### Forest Types
- **AFW**: Acacia Forest Woodland
- **EOF**: Eucalypt Open Forest  
- **EOFD**: Eucalypt Open Forest Degraded
- **ETOF**: Eucalypt Tall Open Forest
- **ETOFD**: Eucalypt Tall Open Forest Degraded

### Climate Scenarios
- **current**: Current climate conditions
- **paris**: Paris Agreement pathway (1.5°C)
- **plus2**: +2°C warming scenario
- **plus3**: +3°C warming scenario

### Management Levels
- **l**: Low management (minimal intervention)
- **m**: Moderate management (standard practices)
- **i**: Intensive management (high intervention)
- **ir**: Intensive management with reforestation
- **mr**: Moderate management with reforestation
- **afm_m**: Adaptive Forest Management (moderate)

## 🚀 Quick Start

### 1. Single Simulation

```bash
# Run a single simulation
python main.py simulate --forest ETOF --years 25 --plot

# With uncertainty analysis and reproducibility
python main.py simulate --forest ETOF --years 25 --plot --uncertainty --seed 42

# Reproducible results - same seed = identical results
python main.py simulate --forest ETOF --years 25 --seed 123
python main.py simulate --forest ETOF --years 25 --seed 123  # Identical results

# With climate configuration
python main.py simulate --forest EOF --years 30 --climate paris --seed 456
```

### 2. Scenario Analysis

```bash
# Run comprehensive scenario analysis with reproducibility
python main.py analyze --forest-types ETOF,EOFD --climates current,paris --years 25 --seed 42

# With custom parameters and uncertainty analysis
python main.py analyze --forest-types ETOF --climates current,plus2 --managements l,m,i --workers 8 --plots --uncertainty --seed 123

# With managed reforestation and reproducibility
python main.py analyze --forest-types ETOF --climates current --managements ir --years 25 --seed 456

# Compare natural vs managed reforestation (reproducible)
python main.py analyze --forest-types ETOF --climates current --managements i,ir --years 25 --seed 789
```

### 3. AFM vs Degrading Analysis

```bash
# Run AFM vs Degrading analysis (no reforestation scenario)
python custom_afm_simulator.py --forest-type ETOF --management i --years 52

# Different forest types and management levels
python custom_afm_simulator.py --forest-type AFW --management m --years 25
python custom_afm_simulator.py --forest-type EOF --management l --years 30

# Custom output directory
python custom_afm_simulator.py --forest-type ETOF --management i --years 52 --output-dir my_afm_analysis

# Skip plot generation for faster execution
python custom_afm_simulator.py --forest-type ETOF --management i --years 52 --no-plots
```

**What this does:**
- Runs only **baseline** (degrading forest) and **management** (AFM) scenarios
- **Excludes reforestation** completely for clean comparison
- Shows dramatic difference between degrading vs managed forest
- Generates focused plots with only relevant scenarios

### 4. Plot Matrix Comparison

```bash
# List available scenarios and plot types
python plot_matrix_generator.py --list

# Create comparison matrices (legends automatically cropped)
python plot_matrix_generator.py --plot-type total_carbon_stocks_all_scenarios
python plot_matrix_generator.py --scenario ETOF_degraded_paris_target_intensive

# Custom matrix comparison
python plot_matrix_generator.py --scenarios ETOF_degraded_paris_target_intensive ETOF_degraded_current_intensive --plot-types total_carbon_stocks_all_scenarios additionality
```

### 5. Data Matrix Generation

```bash
# List available scenarios and data types
python data_matrix_generator.py --list

# Create matrices from CSV data (more flexible)
python data_matrix_generator.py --scenarios ETOF_degraded_paris_target_intensive ETOF_degraded_current_intensive --matrix-type carbon_stocks
python data_matrix_generator.py --scenarios ETOF_degraded_paris_target_intensive ETOF_degraded_current_intensive --matrix-type additionality
python data_matrix_generator.py --scenarios ETOF_degraded_paris_target_intensive ETOF_degraded_current_intensive --matrix-type economics

# Combined matrix showing both carbon stocks and additionality
python data_matrix_generator.py --scenarios ETOF_degraded_paris_target_intensive ETOF_degraded_current_intensive --matrix-type combined

# Multi-row matrices (2x3, 3x3, etc.)
python data_matrix_generator.py --scenarios SCENARIO1 SCENARIO2 SCENARIO3 SCENARIO4 SCENARIO5 SCENARIO6 --matrix-type carbon_stocks --max-per-row 3
```

### 6. Comprehensive Analysis

```bash
# Run comprehensive analysis on existing results
python main.py comprehensive --results-path output/batch_results.csv --output-dir output/analysis
```

## 📊 Key Features

### Core Simulation Engine
- **Tree Yield Formula (TYF)**: Dynamic forest growth modeling
- **Carbon Pool Management**: Above-ground biomass, soil carbon, litter, etc.
- **Disturbance Modeling**: Fire, drought, and other disturbances
- **Economic Analysis**: NPV, IRR, and carbon credit calculations
- **AFM vs Degrading Analysis**: Focused comparison without reforestation scenarios
- **Uncertainty Analysis**: Monte Carlo simulations for parameter uncertainty

### Scenario Management
- **Dynamic Configuration**: Automatic scenario generation from base configs
- **Batch Processing**: Parallel execution of multiple scenarios
- **Climate Integration**: Paris Agreement scenarios, extreme climate, etc.
- **Management Levels**: Baseline, adaptive, intensive management
- **Managed Reforestation**: Option to apply management effects to reforestation scenarios

### Analysis & Visualization
- **Comprehensive Analysis**: 12+ individual plots and statistical analysis
- **Scenario Comparison**: Cross-scenario performance analysis
- **Economic Assessment**: Cost-effectiveness and financial viability
- **Statistical Testing**: ANOVA and other statistical analyses
- **Plot Matrix Generator**: Create comparison matrices from existing plots with automatic legend cropping
- **Data Matrix Generator**: Create comparison matrices by generating new plots from CSV data

## 🔬 Reproducibility & Scientific Rigor

Forest Carbon Lite V.8 implements comprehensive reproducibility features for scientific research:

### Seeded Random Number Generation
- **Isolated RNG Instances**: Each component uses its own `np.random.default_rng(seed)` instance
- **No Global State**: Eliminates coupling between different modules
- **Thread-Safe**: Parallel processing maintains reproducibility
- **End-to-End Control**: Seed propagates through all stochastic components

### Reproducible Components
- **Disturbance Events**: Fire, drought, and mortality events are reproducible
- **Uncertainty Analysis**: Monte Carlo simulations produce identical results with same seed
- **Parameter Sampling**: All parameter distributions use seeded sampling
- **Batch Processing**: Multi-scenario analysis maintains reproducibility

### Usage Examples
```bash
# Identical results across runs
python main.py simulate --forest ETOF --years 25 --seed 42
python main.py simulate --forest ETOF --years 25 --seed 42  # Same results

# Different results with different seeds
python main.py simulate --forest ETOF --years 25 --seed 123  # Different results

# Reproducible scenario analysis
python main.py analyze --forest-types ETOF --climates current --managements i --seed 456
```

### Scientific Benefits
- **Experiment Replication**: Other researchers can reproduce your exact results
- **Parameter Sensitivity**: Test how different random seeds affect outcomes
- **Publication Ready**: Results are fully documented and reproducible
- **Quality Assurance**: Verify that code changes don't affect core results

📖 **Detailed Guide**: See [REPRODUCIBILITY_GUIDE.md](REPRODUCIBILITY_GUIDE.md) for comprehensive reproducibility documentation.

## 🌲 Managed Reforestation

The system supports two types of reforestation scenarios:

### Natural Reforestation (Default)
- **Management Level**: `intensive` (or any standard management level)
- **Characteristics**: Passive forest restoration without ongoing management
- **Parameters**: Natural growth rates, mortality, and disturbance patterns
- **Use Case**: Protected areas, wilderness restoration, passive carbon projects

### Managed Reforestation (New)
- **Management Level**: `intensive_managed_reforestation`
- **Characteristics**: Active forest restoration with ongoing management
- **Parameters**: Enhanced growth rates, reduced mortality, lower disturbance
- **Use Case**: Active reforestation projects, managed carbon projects

### Management Levels Available
- `baseline` - No management effects (degraded baseline)
- `moderate` - Moderate management effects
- `adaptive` - Adaptive management effects  
- `intensive` - Intensive management effects (natural reforestation)
- `intensive_managed_reforestation` - Intensive management effects (managed reforestation)

## 🔧 Advanced Usage

### Using the Package Directly

```python
from forest_carbon import ForestCarbonSimulator, ScenarioManager, ComprehensiveAnalyzer

# Single simulation
simulator = ForestCarbonSimulator(
    forest_type='ETOF',
    years=25,
    area_ha=1000,
    output_dir='output'
)
results = simulator.run(generate_plots=True)

# Scenario analysis
manager = ScenarioManager()
results = manager.run_analysis(
    forest_types=['ETOF', 'EOF'],
    climates=['current', 'paris_target'],
    managements=['baseline', 'adaptive'],
    years=25
)

# Comprehensive analysis
analyzer = ComprehensiveAnalyzer(results_path='output/batch_results.csv')
analyzer.run_complete_analysis()
```

### Configuration Management

```python
from forest_carbon.utils import ConfigLoader
from forest_carbon.scenarios import ForestType, ClimateScenario, ManagementLevel

# Discover available configurations
forest_types = ForestType.get_available_types()
climates = ClimateScenario.get_available_scenarios()
managements = ManagementLevel.get_available_levels()

print(f"Available forest types: {forest_types}")
print(f"Available climates: {climates}")
print(f"Available managements: {managements}")
```

## 📁 Output Structure

### Single Simulation Output
```
output/
├── [scenario_name]/
│   ├── plots/                      # Individual scenario plots
│   ├── uncertainty_analysis/       # Uncertainty analysis (if enabled)
│   ├── results_summary.csv         # Summary results
│   ├── finance_results.csv         # Economic results
│   └── simulation_metadata.json    # Simulation metadata
```

### Scenario Analysis Output
```
output/
├── batch_results.csv               # Batch results summary
├── analysis/                       # Comprehensive analysis
│   ├── comprehensive_analysis.png  # Main analysis plot
│   ├── 01_climate_impact.png       # Individual plots
│   ├── 02_management_effectiveness.png
│   ├── ...                         # 12 total individual plots
│   ├── scenario_summary.csv        # Data tables
│   ├── forest_type_performance.csv
│   ├── ...                         # 6 total data tables
│   ├── statistical_analysis.json   # Statistical results
│   └── comprehensive_report.md     # Analysis report
└── [individual_scenario_outputs]/  # Individual scenario results
```

## 🎨 Visualization

The system includes a comprehensive color management system:

```python
from forest_carbon.utils.colors import color_manager

# Get scenario colors
baseline_color = color_manager.get_scenario_color('baseline')
management_color = color_manager.get_scenario_color('management')

# Get forest type colors
etof_color = color_manager.get_forest_type_color('ETOF')

# Get complete styling
style = color_manager.get_scenario_style('management')
# Returns: {'color': '#3498db', 'linestyle': '-', 'linewidth': 2.5, 'alpha': 0.8}
```

## 🔬 Scientific Background

### Tree Yield Formula (TYF)
The TYF is a dynamic forest growth model that simulates:
- **Mortality (M)**: Tree death rates under different scenarios
- **Growth (G)**: Biomass accumulation rates
- **Yield (y)**: Carbon sequestration efficiency

### Carbon Pools
- **Above-Ground Biomass (AGB)**: Living tree biomass
- **Below-Ground Biomass (BGB)**: Root systems
- **Litter**: Dead organic matter
- **Soil Carbon**: Organic carbon in soil
- **Harvested Wood Products (HWP)**: Long-term carbon storage

### Climate Integration
- **Current Climate**: Baseline conditions
- **Paris Target**: 1.5°C warming scenario
- **Paris Overshoot**: 2.0°C warming scenario
- **Extreme Climate**: High temperature, low rainfall

## 🌲 AFM vs Degrading Analysis

The custom AFM simulator (`custom_afm_simulator.py`) provides focused analysis comparing Active Forest Management (AFM) against degrading baseline scenarios, without the complexity of reforestation scenarios.

### Key Benefits
- **Clean Comparison**: Only baseline (degrading) vs management (AFM) scenarios
- **No Reforestation Clutter**: Excludes reforestation scenarios for focused analysis
- **Dramatic Results**: Shows stark contrast between degrading and managed forests
- **Realistic Parameters**: Uses proper intensive management configurations

### Example Results
```
Key Results:
  Degrading Baseline: 475.1 t CO2e/ha
  AFM Management: 1048.0 t CO2e/ha
  Carbon Additionality: 572.9 t CO2e/ha
  AFM NPV: $98,180,607/ha
```

### Management Levels
- **baseline**: No management effects (degraded forest)
- **moderate**: Moderate management effects (y_multiplier: 1.2)
- **adaptive**: Adaptive management effects (y_multiplier: 1.35)
- **intensive**: Intensive management effects (y_multiplier: 1.5)

### Output Files
```
output/ETOF_current_intensive_afm_only/
├── afm_analysis_summary.csv          # Summary results
├── baseline_results/                 # Detailed baseline data
├── management_results/               # Detailed AFM data
└── plots/                           # Comparison visualizations
    ├── total_carbon_stocks_all_scenarios.png  # Baseline vs AFM trajectories
    └── additionality.png                      # AFM additionality (difference)
```

## 📊 Plot Matrix Generator

The Plot Matrix Generator (`plot_matrix_generator.py`) creates comparison matrices by pulling existing plots from different scenarios into one PNG file. **No new computations needed** - it simply arranges existing plot images in a grid format with automatic legend cropping for clean comparison.

### Key Features
- **🎯 Legend Cropping**: Automatically crops repetitive legends from individual plots (default)
- **📐 Multiple Matrix Types**: Plot-type matrices, scenario matrices, and custom matrices
- **🎨 Flexible Layout**: Control grid size and arrangement
- **📈 High Quality**: 300 DPI output suitable for publications

### Quick Start

```bash
# List available scenarios and plot types
python plot_matrix_generator.py --list

# Compare one plot type across all scenarios (legends cropped)
python plot_matrix_generator.py --plot-type total_carbon_stocks_all_scenarios

# Show all plots for one scenario (legends cropped)
python plot_matrix_generator.py --scenario ETOF_burnt_paris_overshoot_intensive

# Custom matrix with specific scenarios and plot types
python plot_matrix_generator.py --scenarios ETOF_burnt_paris_overshoot_intensive ETOF_burnt_paris_target_intensive --plot-types total_carbon_stocks_all_scenarios additionality

# Keep legends when needed
python plot_matrix_generator.py --plot-type additionality --keep-legends
```

### Available Plot Types
- `additionality` - Carbon Additionality
- `biomass_all_scenarios` - Biomass All Scenarios  
- `carbon_pools_comparison` - Carbon Pools Comparison
- `economics_management` - Economics - Management
- `economics_reforestation` - Economics - Reforestation
- `total_carbon_stocks_all_scenarios` - Total Carbon Stocks
- And 5 more plot types...

### Example Commands

```bash
# Compare carbon stocks across climate scenarios
python plot_matrix_generator.py --scenarios ETOF_burnt_paris_overshoot_intensive ETOF_burnt_paris_target_intensive ETOF_paris_overshoot_intensive --plot-types total_carbon_stocks_all_scenarios

# Compare management intensity
python plot_matrix_generator.py --scenarios ETOF_burnt_paris_overshoot_intensive ETOF_burnt_paris_overshoot_moderate --plot-types additionality carbon_pools_comparison economics_management

# Control grid layout
python plot_matrix_generator.py --plot-type additionality --max-per-row 6
```

### Output Files
All matrices are saved to the `output_matrix` folder with descriptive names:
- `total_carbon_stocks_all_scenarios_matrix_comparison.png` - All scenarios for one plot type
- `ETOF_burnt_paris_overshoot_intensive_all_plots_matrix.png` - All plots for one scenario  
- `custom_matrix_3x2.png` - Custom 3 scenarios × 2 plot types matrix

📖 **Detailed Guide**: See [documentation/PLOT_MATRIX_GUIDE.md](documentation/PLOT_MATRIX_GUIDE.md) for comprehensive usage documentation.

## 📊 Data Matrix Generator

The Data Matrix Generator (`data_matrix_generator.py`) creates comparison matrices by generating **new plots from CSV data** rather than pulling existing plot images. This provides more flexibility for custom data-driven comparisons and allows you to create matrices even when individual plot images don't exist.

### Key Features
- **📈 Data-Driven**: Creates plots directly from CSV data files
- **🎯 Multiple Matrix Types**: Carbon stocks, additionality, economics, and combined matrices
- **🎨 Flexible Visualization**: Custom plot generation with consistent styling
- **📊 Multi-Row Layouts**: Support for 2x3, 3x3, and other grid arrangements
- **📊 No Dependencies on Existing Plots**: Works even if individual plot images are missing

### Available Matrix Types

#### Carbon Stocks Matrix
```bash
python data_matrix_generator.py --scenarios ETOF_degraded_paris_target_intensive ETOF_degraded_current_intensive --matrix-type carbon_stocks
```
- Shows all three scenarios (baseline, management, reforestation) for each scenario
- Data source: `sequestration_curves.csv`

#### Additionality Matrix
```bash
python data_matrix_generator.py --scenarios ETOF_degraded_paris_target_intensive ETOF_degraded_current_intensive --matrix-type additionality
```
- Shows carbon additionality (difference from baseline or vs 0 for reforestation)
- Data source: `sequestration_curves.csv`

#### Economics Matrix
```bash
python data_matrix_generator.py --scenarios ETOF_degraded_paris_target_intensive ETOF_degraded_current_intensive --matrix-type economics
```
- Shows NPV and other economic metrics as bar charts
- Data source: `results_summary.csv`

#### Combined Matrix
```bash
python data_matrix_generator.py --scenarios ETOF_degraded_paris_target_intensive ETOF_degraded_current_intensive --matrix-type combined
```
- Shows both carbon stocks AND additionality side-by-side for each scenario
- Data source: `sequestration_curves.csv`

### Data Matrix vs Plot Matrix

| Feature | Data Matrix Generator | Plot Matrix Generator |
|---------|----------------------|----------------------|
| **Data Source** | CSV files | Existing PNG images |
| **Flexibility** | High - custom plots | Medium - existing plots only |
| **Speed** | Slower - generates plots | Faster - arranges images |
| **Customization** | Full control over plot style | Limited to existing plot style |
| **Use Case** | Custom analysis, new visualizations | Quick comparison of existing plots |

📖 **Detailed Guide**: See [documentation/DATA_MATRIX_GUIDE.md](documentation/DATA_MATRIX_GUIDE.md) for comprehensive usage documentation.

### Configuration Reference
When using matrix generators, scenario names follow the pattern: `{SITE}_{CLIMATE}_{MANAGEMENT}`

**Examples with new simplified names:**
```bash
# ETOF (Eucalypt Tall Open Forest) scenarios
ETOF_current_i      # ETOF + current climate + intensive management
ETOF_paris_ir       # ETOF + Paris climate + intensive management + reforestation  
ETOF_plus2_m        # ETOF + plus2°C + moderate management
ETOF_plus3_lr       # ETOF + plus3°C + light management + reforestation
ETOF_current_afm_m  # ETOF + current climate + moderate AFM only
ETOF_paris_afm_m    # ETOF + Paris climate + moderate AFM only

# EOFD (Eucalypt Open Forest Degraded) scenarios  
EOFD_current_m      # EOFD + current climate + moderate management
EOFD_paris_ir       # EOFD + Paris climate + intensive management + reforestation
EOFD_current_afm_m  # EOFD + current climate + moderate AFM only
EOFD_plus2_afm_m    # EOFD + plus2°C + moderate AFM only
```

## 📈 Performance

- **Parallel Processing**: Multi-core batch execution
- **Memory Efficient**: Optimized data structures
- **Fast Visualization**: Matplotlib with optimized rendering
- **Scalable**: Handles 100+ scenarios efficiently

## 🛠️ Development

### Running Tests
```bash
# Test single simulation
python main.py simulate --forest ETOF --years 5 --plot

# Test scenario analysis
python main.py analyze --forest-types ETOF --climates current --years 5

# Test comprehensive analysis
python main.py comprehensive --results-path output/batch_results.csv

# Test AFM vs Degrading analysis
python custom_afm_simulator.py --forest-type ETOF --management i --years 5
```

### Adding New Forest Types
1. Create `configs/base/site_[FOREST_TYPE].yaml`
2. Add TYF calibrations for baseline, management, reforestation
3. Update forest type discovery in `ForestType` class

### Adding New Climate Scenarios
1. Create `configs/base/climate_[SCENARIO].yaml`
2. Define temperature and rainfall adjustments
3. Update climate discovery in `ClimateScenario` class

## 📚 Documentation

- **API Documentation**: Available in docstrings
- **Configuration Guide**: See `configs/base/` examples
- **Scenario Builder Guide**: See `scenario_system/` documentation
- **Validation Guide**: See `validation.py` for input requirements

## 🤝 Contributing

1. Follow the unified package structure
2. Add comprehensive docstrings
3. Include type hints
4. Test with multiple scenarios
5. Update documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Healthy Forests Foundation**
- **FullCAM**: Australian Government forest carbon model
- **Tree Yield Formula**: Scientific foundation
- **Climate Science**: IPCC scenarios and data

---

**Forest Carbon Lite V.1.0** - Unified, scalable, and comprehensive forest carbon modeling.
