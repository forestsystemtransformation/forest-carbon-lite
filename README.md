# Forest Carbon Lite v0.1 - Professional Review Draft

A transparent Python implementation of forest carbon accounting for restoration and management decision support under climate change. Built on the validated FullCAM Tree Yield Formula (TYF) with integrated climate scenarios and economic analysis.

**Version:** 0.1 (Professional Review Draft)  
**Status:** Seeking validation partnerships and peer review feedback  
**Institution:** Healthy Forests Foundation

---

## ⚠️ VALIDATION STATUS

**Forest Carbon Lite v0.1 is a Professional Review Draft**

This implementation uses validated components (FullCAM TYF validated against 9,300+ plots by Forrester et al., 2025) but **the integrated system lacks independent field validation**.

### ✅ **Appropriate Use Cases:**
- Preliminary site screening and prioritization
- Scenario comparison (relative performance analysis)
- Training and education on forest carbon dynamics
- Identifying knowledge gaps and research priorities
- Exploring climate-management trade-offs

### ❌ **NOT Yet Appropriate For:**
- Sole basis for investment decisions
- Carbon credit project MRV (Monitoring, Reporting, Verification)
- Regulatory compliance reporting
- Definitive carbon forecasts without field verification

### 🔬 **Validation Priority:**
We are actively seeking validation partnerships to test FCL against 5-10 year monitoring data from existing restoration projects. This will establish confidence bounds and identify systematic biases requiring correction.

**Critical Limitation:** Without field validation, FCL should be used for preliminary screening and scenario exploration only. Independent verification is essential before operational deployment for project planning and investment decisions.

**Reference:** See Section 4.2 of the technical paper for complete validation status and requirements.

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Scientific Foundation](#-scientific-foundation)
- [Model Parameters](#-model-parameters)
- [Quick Start](#-quick-start)
- [Project Structure](#️-project-structure)
- [Configuration System](#-configuration-system)
- [Usage Examples](#-usage-examples)
- [Output Structure](#-output-structure)
- [Reproducibility](#-reproducibility)
- [Advanced Features](#-advanced-features)
- [Known Limitations](#-known-limitations)
- [Contributing](#-contributing)

---

## 🎯 Key Features

### Core Capabilities
- **Validated Growth Engine**: FullCAM Tree Yield Formula (Forrester et al., 2025)
- **Climate-Management Separation**: Orthogonal parameterization prevents double-counting
- **Uncertainty Quantification**: Monte Carlo analysis with reproducible seeding
- **Economic Analysis**: NPV, IRR, carbon credit revenue projections
- **Visual Decision Support**: 12+ plot types for scenario comparison

### Scientific Rigor
- **Published Parameter Sets**: All defaults from peer-reviewed literature
- **Transparent Assumptions**: Every parameter documented with source
- **Reproducible Results**: Seeded RNG for experiment replication
- **Comprehensive Documentation**: Technical paper with mathematical derivations

### Model Innovation (Section 3 of technical paper)
- **Orthogonal Climate-Management Effects**: Prevents double-counting (Algorithm 1)
- **Active Forest Management (AFM) Framework**: Based on Bennett et al. (2024)
- **Integrated Economic Analysis**: Aligned with carbon credit markets

---

## 🔬 Scientific Foundation

### The FullCAM Tree Yield Formula

FCL implements the FullCAM Tree Yield Formula (TYF) as validated by Forrester et al. (2025) against 9,300+ Australian forest plots.

**Annual biomass increment:**

```
ΔAGBₜ = y × M × (e^(-k/t) - e^(-k/(t-1))) × FPIᵣₐₜᵢₒ
```

Where:
- **ΔAGBₜ** = Annual increment in aboveground biomass (t/ha/yr)
- **t** = Forest age (years)
- **M** = Maximum aboveground biomass potential (t/ha) - *site-specific from Roxburgh et al. (2019)*
- **G** = Age of maximum growth rate (years)
- **k** = 2G - 1.25 (growth curve shape parameter, derived not calibrated)
- **y** = Management multiplier (1.0 = baseline, >1.0 = enhanced)
- **FPIᵣₐₜᵢₒ** = Climate productivity adjustment (FPI/FPIₐᵥₑ)

**Key References:**
- Forrester et al. (2025) - FullCAM calibration methodology
- Paul & Roxburgh (2020) - Regeneration parameterization  
- Paul & Roxburgh (2025) - Validation and domain of application
- Waterworth et al. (2007) - Original TYF formulation
- Roxburgh et al. (2019) - Maximum biomass spatial layer

### Climate-Management Separation (Section 3.2-3.3)

**Critical Innovation:** FCL implements strict orthogonal parameterization to avoid double-counting:

```
Growth = Base × (FPIₜ/FPIᵦₐₛₑₗᵢₙₑ) × y
                 └──────┬──────┘   └┬┘
                  Climate only    Mgmt only
```

This ensures:
- Climate effects modify FPI ratios exclusively
- Management effects modify y-multiplier exclusively  
- Combined effects are multiplicative, not compounded

**Reference:** Algorithm 1 in Section 3.3 of technical paper

### Active Forest Management (AFM) - Section 3.4

FCL's management scenarios can represent Active Forest Management (AFM) as defined by Bennett et al. (2024): temporary restoration interventions in degraded forests that work toward obsolescence.

**AFM Principles:**
- Management as ecosystem medicine, not perpetual manipulation
- Interventions decrease as forests regain autonomous function
- Focus on restoration, biodiversity, and carbon sequestration
- Not commercial forestry operations

**Current Limitation (v0.1):** Model applies constant management intensity throughout simulations, overestimating long-term costs. Temporal de-escalation (intensive → targeted → minimal over 25+ years) planned for v0.2.

**Reference:** Bennett, L.T. et al. (2024) "Active Management: A Definition and Considerations for Implementation in Forests of Temperate Australia." Australian Forestry 87(3): 125-147.

---

## 📊 Model Parameters

### Table 1: Forest Type Parameters (from Technical Paper Table 1)

Default parameters for FCL v0.1 from published FullCAM calibrations:

| Parameter | ETOF | EOF | AFW | Source |
|-----------|------|-----|-----|--------|
| **FullCAM TYF Growth Parameters** |
| M - Max biomass (t/ha) | 290 | 170 | 49 | Roxburgh et al. (2019)* |
| G - Age max growth (yr) | 12.53 | 12.53 | 12.53 | Paul & Roxburgh (2020, 2025) |
| y - Management multiplier | 1.0-1.35 | 1.0-1.35 | 1.0-1.35 | See Table 2 |
| k - Shape parameter | 23.81 | 23.81 | 23.81 | Derived: k = 2G - 1.25 |
| **Ecosystem Process Parameters** |
| Root:shoot ratio | 0.25 | 0.30 | 0.40 | Mokany et al. (2006) |
| Mortality (%/yr) | 0.75 | 1.15 | 1.75 | FullCAM defaults |
| Fire return interval (yr) | 35 | 20 | 15 | Historical records |
| Carbon fraction | 0.47 | 0.47 | 0.47 | IPCC (2023) |

**Forest Type Abbreviations:**
- **ETOF** = Eucalypt Tall Open Forest
- **EOF** = Eucalypt Open Forest  
- **AFW** = Acacia Forest Woodland

**Important Notes:**
- *M values are site-specific - practitioners should obtain values from Roxburgh et al. (2019) spatial layer for their location
- G = 12.53 years is calibrated for natural regeneration (Paul & Roxburgh 2020)
- k is not independently calibrated but derived from G
- Management multipliers (y) vary by intervention intensity (see Table 2)

**Parameter Source Details:**
- **M (Maximum biomass)**: ETOF and EOF from Roxburgh et al. (2019) spatial layer (national coverage); AFW from Paul & Roxburgh (2025) mulga calibration (2,438 plots)
- **G (Age at max growth)**: Calibrated for natural regeneration across all forest types (Paul & Roxburgh 2020, 2025)
- **y (Management multiplier)**: Baseline = 1.0 (no intervention). Enhanced values based on intensity levels in Table 2
- **Root:shoot ratios**: From Mokany et al. (2006); ratios decrease with increasing precipitation
- **Mortality rates**: FullCAM default values for forest types
- **Fire return intervals**: Based on historical fire frequency records
- **Carbon fraction**: IPCC default (IPCC 2023). Note: Australian eucalypts may range 0.45-0.49 (Ximenes & Wright 2006), introducing ±4% uncertainty

### Table 2: Management Levels (from Technical Paper Table 2)

| Code | Name | Applied to | y Multiplier | Description |
|------|------|-----------|--------------|-------------|
| `l` | Low Management | AFM in degraded forest | 1.10 | Minimal intervention |
| `m` | Moderate Management | AFM in degraded forest | 1.20 | Standard practices |
| `i` | Intensive Management | AFM in degraded forest | 1.35 | High intervention |
| `mr` | Moderate Reforestation | Reforestation and AFM | 1.20 | Moderate management + planting |
| `ir` | Intensive Reforestation | Reforestation and AFM | 1.35 | Intensive management + planting |

**Management Effectiveness Sources:**
- Cost ranges: Austin et al. (2020), Busch et al. (2024), Evans (2018)
- Carbon uplift estimates: Paul et al. (2018), Paul & Roxburgh (2020)
- Intervention cycles: Australian forest management guidelines

### Table 3: Climate Scenarios (from Technical Paper Table 5)

| Scenario | ΔT (°C) | ΔP (%) | FPI ratio | Fire interval | Mortality increase |
|----------|---------|--------|-----------|---------------|-------------------|
| `current` | 0 | 0 | 1.00 | 20 yr | 1.0× |
| `paris` | +1.5 | -5 | 0.85 | 15 yr | 1.3× |
| `plus2` | +2.0 | -10 | 0.80 | 12 yr | 1.5× |
| `plus3` | +3.0 | -15 | 0.70 | 8 yr | 2.0× |

**Climate Scenario Sources:**
- Temperature scenarios: IPCC (2023) projections
- FPI adjustments: Kesteven and Landsberg (2004) process-based model
- Fire frequency changes: Boer et al. (2021), Furlaud et al. (2021), McColl-Gausden et al. (2022)
- Mortality relationships: Wardlaw (2021) climate-fire analysis

### Table 4: Economic Parameters (from Technical Paper Table 3)

| Parameter | FCL Range (2024 AUD) | Literature Range | Source |
|-----------|---------------------|------------------|--------|
| Carbon price | $35-70/tCO₂e | $20-100/tCO₂e | Market data |
| Establishment cost | $1,500-3,000/ha | $2,000-6,000/ha | Jonson & Freudenberger (2011), Austin et al. (2020), Pacheco et al. (2024) |
| Management cost | $25-100/ha/yr | $150-250/ha/yr* | Conservative range; operational projects higher |
| MRV cost | Not included | $15-40/ha/yr | Jonson & Freudenberger (2011), Verra VM0047 |
| Discount rate | 5-7% real | 3-12% | Project finance standards |
| Project period | 25-100 years | 25-100 years | Policy constraints |
| Buffer pool | 5-25% | 10-30% | Conservative crediting |

*FCL uses conservative lower bound for management costs

**Economic Validation Sources:**
- Austin et al. (2020) - Global forest mitigation costs
- Busch et al. (2024) - Cost-effectiveness of natural regeneration
- Evans (2018) - Australian carbon farming policy effectiveness
- Jonson & Freudenberger (2011) - Australian woodland restoration costs
- Pacheco et al. (2024) - Economics of carbon sequestration

**MRV Cost Note:** Monitoring, Reporting, and Verification costs are NOT included in FCL economic projections. Users should budget $15-40/ha/yr depending on methodology:
- $15/ha/yr: Low-intensity field monitoring (Jonson & Freudenberger 2011)
- $30-40/ha/yr: Commercial carbon projects (Verra VM0047, Clean Energy Regulator standards)

### Validation Benchmarks (Section 4.2.1)

FCL's modeled sequestration rates align with published Australian data:
- **8.5 tCO₂e/ha/yr**: Eucalypt woodland restoration (Jonson & Freudenberger 2011)
- **12 tCO₂e/ha/yr**: Intensive mixed-species reforestation (Paul et al. 2018)
- **4.5 tCO₂e/ha/yr**: Natural regeneration (Cook-Patton et al. 2020)

These provide confidence bounds for FCL predictions across the spectrum from passive natural regeneration to intensive restoration management.

---

## 🗂️ Project Structure

```
forest-carbon-lite/
├── forest_carbon/                    # Main package
│   ├── __init__.py
│   ├── core/                        # Core simulation engine
│   │   ├── simulator.py            # Main ForestCarbonSimulator
│   │   ├── uncertainty_analysis.py # Monte Carlo analysis
│   │   └── models/                 # Physical models
│   │       ├── tyf_engine.py       # Tree Yield Formula
│   │       ├── carbon_pools.py     # Carbon pool dynamics
│   │       ├── disturbance.py      # Fire, drought, mortality
│   │       └── economics.py        # NPV, IRR, carbon credits
│   ├── scenarios/                   # Scenario management
│   │   ├── builder.py              # Scenario configuration
│   │   ├── runner.py               # Batch execution
│   │   ├── analyzer.py             # Scenario analysis
│   │   └── manager.py              # Orchestrator
│   ├── analysis/                    # Analysis & reporting
│   │   └── comprehensive.py        # Comprehensive analysis
│   ├── utils/                       # Utilities
│   │   ├── config_loader.py        # YAML configuration
│   │   ├── validation.py           # Input validation
│   │   ├── colors.py               # Plotting colors
│   │   ├── constants.py            # Physical constants
│   │   └── climate_adjustments.py  # FPI calculations
│   └── visualization/               # Plotting
│       └── plotter.py              # Matplotlib visualization
├── main.py                         # Unified CLI entry point
├── custom_afm_simulator.py         # AFM vs Degrading tool
├── plot_matrix_generator.py        # Plot comparison matrices
├── data_matrix_generator.py        # Data-driven matrices
├── configs/                        # Configuration files
│   ├── base/                       # Base configurations
│   │   ├── management_*.yaml       # Management configs
│   │   ├── climate_*.yaml          # Climate configs
│   │   └── site_*.yaml             # Site/forest configs
│   └── generated/                  # Auto-generated scenarios
├── output/                         # Simulation outputs
├── output_matrix/                  # Matrix comparisons
└── requirements.txt                # Python dependencies
```

---

## ⚙️ Configuration System

### Base Configuration Files

FCL uses YAML configuration files organized by category:

#### Site/Forest Configurations (`site_*.yaml`)
- `site_ETOF.yaml` - Eucalypt Tall Open Forest
- `site_EOFD.yaml` - Eucalypt Open Forest Degraded
- `site_EOF.yaml` - Eucalypt Open Forest
- `site_ETOFD.yaml` - Eucalypt Tall Open Forest Degraded
- `site_AFW.yaml` - Acacia Forest Woodland

#### Climate Configurations (`climate_*.yaml`)
- `climate_current.yaml` - No climate change (baseline)
- `climate_paris.yaml` - Paris Agreement (+1.5°C)
- `climate_plus2.yaml` - +2°C warming
- `climate_plus3.yaml` - +3°C warming

#### Management Configurations (`management_*.yaml`)
- `management_l.yaml` - Low management (minimal intervention)
- `management_m.yaml` - Moderate management (standard practices)
- `management_i.yaml` - Intensive management (high intervention)
- `management_ir.yaml` - Intensive management + reforestation
- `management_mr.yaml` - Moderate management + reforestation

### Scenario Naming Convention

Scenarios are automatically named: `{SITE}_{CLIMATE}_{MANAGEMENT}`

**Examples:**
```
ETOF_current_i      # ETOF + current climate + intensive management
ETOF_paris_ir       # ETOF + Paris climate + intensive reforestation
EOF_plus2_m         # EOF + +2°C warming + moderate management
EOFD_current_l      # EOFD (degraded) + current + low management
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/forest-carbon-lite.git
cd forest-carbon-lite

# Install dependencies
pip install -r requirements.txt
```

### 1. Single Simulation

```bash
# Run basic simulation
python main.py simulate --forest ETOF --years 25 --plot

# With uncertainty analysis and reproducibility
python main.py simulate --forest ETOF --years 25 --plot --uncertainty --seed 42

# Reproducible results - same seed = identical results
python main.py simulate --forest ETOF --years 25 --seed 123
python main.py simulate --forest ETOF --years 25 --seed 123  # Identical output

# With climate scenario
python main.py simulate --forest EOF --years 30 --climate paris --seed 456
```

### 2. Scenario Analysis (Batch Mode)

```bash
# Run comprehensive scenario analysis
python main.py analyze --forest-type ETOF,EOFD --climate current,paris --years 25 --seed 42

# Custom parameter combinations
python main.py analyze --forest-type ETOF --climate current,plus2 --management l,m,i --workers 8 --plot --uncertainty --seed 123

# With managed reforestation
python main.py analyze --forest-type ETOF --climate current --management ir --years 25 --seed 456

# Compare natural vs managed reforestation
python main.py analyze --forest-type ETOF --climate current --management i,ir --years 25 --seed 789
```

### 3. AFM vs Degrading Analysis

Focused comparison of Active Forest Management (AFM) against degrading baseline, excluding reforestation scenarios for clarity:

```bash
# Run AFM vs Degrading analysis
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
- Compares only **baseline** (degrading forest) vs **management** (AFM)
- **Excludes reforestation** for clean comparison
- Shows dramatic difference between degrading vs managed forest
- Generates focused plots with only relevant scenarios

### 4. Plot Matrix Comparison

Create comparison matrices by arranging existing plots:

```bash
# List available scenarios and plot types
python plot_matrix_generator.py --list

# Create comparison matrices (legends automatically cropped)
python plot_matrix_generator.py --plot-type total_carbon_stocks_all_scenarios
python plot_matrix_generator.py --scenario ETOF_paris_i

# Custom matrix comparison
python plot_matrix_generator.py --scenarios ETOF_paris_i ETOF_current_i --plot-types total_carbon_stocks_all_scenarios additionality
```

### 5. Data Matrix Generation

Create comparison matrices from CSV data:

```bash
# List available scenarios and data types
python data_matrix_generator.py --list

# Create matrices from CSV data
python data_matrix_generator.py --scenarios EOFD_paris_i EOFD_current_i --matrix-type carbon_stocks
python data_matrix_generator.py --scenarios ETOF_paris_i ETOF_current_i --matrix-type additionality
python data_matrix_generator.py --scenarios ETOF_paris_i ETOF_current_i --matrix-type economics

# Combined matrix (carbon stocks + additionality)
python data_matrix_generator.py --scenarios ETOF_paris_i ETOF_current_i --matrix-type combined

# Multi-row matrices (2x3, 3x3, etc.)
python data_matrix_generator.py --scenarios SCENARIO1 SCENARIO2 SCENARIO3 --matrix-type carbon_stocks --max-per-row 3
```

### 6. Comprehensive Analysis

```bash
# Run comprehensive analysis on batch results
python main.py comprehensive --results-path output/batch_results.csv --output-dir output/analysis
```

---

## 📁 Output Structure

### Single Simulation Output

```
output/
└── [scenario_name]/
    ├── plots/                      # Visualization outputs
    ├── uncertainty_analysis/       # Monte Carlo results (if enabled)
    ├── results_summary.csv         # Summary statistics
    ├── finance_results.csv         # Economic analysis
    ├── sequestration_curves.csv    # Time series data
    └── simulation_metadata.json    # Configuration and metadata
```

### Scenario Analysis Output

```
output/
├── batch_results.csv               # All scenarios summary
├── analysis/                       # Comprehensive analysis
│   ├── comprehensive_analysis.png  # Main overview plot
│   ├── 01_climate_impact.png       # Individual plots (12 total)
│   ├── 02_management_effectiveness.png
│   ├── ...
│   ├── scenario_summary.csv        # Data tables (6 total)
│   ├── forest_type_performance.csv
│   ├── ...
│   ├── statistical_analysis.json   # ANOVA and stats
│   └── comprehensive_report.md     # Analysis report
└── [individual_scenarios]/         # Individual scenario outputs
```

---

## 🔬 Reproducibility & Scientific Rigor

FCL v0.1 implements comprehensive reproducibility features for research:

### Seeded Random Number Generation

- **Isolated RNG Instances**: Each component uses its own `np.random.default_rng(seed)`
- **No Global State**: Eliminates coupling between modules
- **Thread-Safe**: Parallel processing maintains reproducibility
- **End-to-End Control**: Seed propagates through all stochastic components

### Reproducible Components

- **Disturbance Events**: Fire, drought, mortality
- **Uncertainty Analysis**: Monte Carlo simulations
- **Parameter Sampling**: All distributions seeded
- **Batch Processing**: Multi-scenario analysis reproducible

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

- **Experiment Replication**: Researchers can reproduce exact results
- **Parameter Sensitivity**: Test how different random seeds affect outcomes
- **Publication Ready**: Results fully documented and reproducible
- **Quality Assurance**: Verify code changes don't affect core results

---

## 🔧 Configuration Examples

### Site Configuration Example

**File:** `configs/base/site_ETOF.yaml` - Eucalypt Tall Open Forest

```yaml
# Eucalypt Tall Open Forest (ETOF) - FullCAM Calibrated Parameters
# Based on Paul KI and Roxburgh SH (2025) - 9,300+ field plots validation

# Forest type parameters
K_AGB: 300.0  # Maximum potential biomass (tonnes/ha)
G: 11.254      # Age of maximum growth rate (from FullCAM tyf_G)
root_shoot: 0.25  # Root-to-shoot ratio

# Initial biomass values (tonnes/ha)
initial_biomass_baseline: 284.0    # Typical ETOF AGB from FullCAM
initial_biomass_management: 284.0  # Same starting biomass for management
initial_biomass_reforestation: 10.0  # Reforestation starts from zero

# Mortality rates (annual)
m_degraded: 0.015        # ETOF degraded mortality
m_managed: 0.015         # ETOF managed mortality
m_reforestation: 0.015   # ETOF reforestation mortality

# TYF calibrations (Tree Yield Formula parameters)
tyf_calibrations:
  baseline:
    M: 300.0       # Maximum potential biomass 
    G: 11.254      # Age of maximum growth
    y: 1           # Growth multiplier (baseline)
  management:
    M: 300.0       # Same maximum potential
    G: 11.254      # Same growth timing
    y: 1           # Growth multiplier (management effects applied by scenario builder)
  reforestation:
    M: 300.0       # Same maximum potential
    G: 11.254      # Same growth timing
    y: 1.0         # Growth multiplier (management effects applied by scenario builder)
```

### Climate Configuration Example

**File:** `configs/base/climate_plus2.yaml` - +2°C Warming Scenario

```yaml
# Paris Agreement 2.0°C Climate Scenario
# Exceeds critical thresholds - approaching ecosystem collapse

name: "plus2"
description: "Exceeds 2°C - approaching ecosystem collapse thresholds"
display_name: "Paris 2.0°C Overshoot"

# Climate effects on forest parameters
effects:
  fpi_adjustment: 0.78         # 22% productivity reduction
  temp_change: 2.0             # Additional 2.0°C warming
  rainfall_change: -0.20       # 20% rainfall reduction
  mortality_adjustment: 1.65   # 65% mortality increase
  disturbance_adjustment: 1.7  # 70% disturbance increase

# Climate metadata
metadata:
  temperature_target: "2.0°C above pre-industrial"
  rainfall_change: "20% reduction"
  fire_risk: "70% increase"
  drought_frequency: "Frequent and severe"
  ecosystem_threshold: "Approaching collapse"
```

### Management Configuration Example

**File:** `configs/base/management_i.yaml` - Intensive Management

```yaml
# Intensive Management - Maximum intervention with intensive forest management

name: "i"
description: "Intensive management - Maximum intervention with intensive forest management"
display_name: "Intensive Management"

# Management effects on forest parameters
effects:
  y_multiplier: 1.5            # 50% growth improvement
  mortality_factor: 0.5        # 50% mortality reduction
  disturbance_factor: 0.3      # 70% disturbance reduction
  prescribed_burn_effect: 0.2  # 20% biomass consumption from frequent burns

# Management practices
practices:
  prescribed_burning: "Very frequent (3 year intervals)"
  thinning: "Intensive for maximum growth"
  species_mixing: "Climate-adapted species with genetic diversity"
  pest_control: "Integrated pest management with monitoring"
  fire_management: "Advanced fuel management and fire breaks"
  climate_adaptation: "Species selection and assisted migration"
  carbon_monitoring: "Continuous carbon stock monitoring"
```

---

## 💻 Usage Examples

### Basic Simulation

```python
from forest_carbon import ForestCarbonSimulator

# Initialize simulator with forest type
sim = ForestCarbonSimulator(
    forest_type='ETOF',  # Eucalypt Tall Open Forest
    years=25,
    area_ha=1000.0
    # Note: climate_config parameter has known issues in v0.1
)

# Run simulation with specific management level
results = sim.run(
    scenario='management',  # or 'baseline', 'reforestation'
    management_level='i',   # Intensive management
    generate_plots=True,
    seed=42  # For reproducibility
)

# Get key outputs from results
carbon_data = results.sequestration_curves
economic_data = results.economics
summary_stats = results.summary

# Extract specific metrics
final_carbon = carbon_data['management_co2e'].iloc[-1]
npv_per_ha = economic_data['management']['npv_per_ha']
total_npv = npv_per_ha * sim.area_ha

print(f"Final carbon stock: {final_carbon:.1f} tCO2e/ha")
print(f"NPV per hectare: ${npv_per_ha:,.0f}/ha")
print(f"Total project NPV: ${total_npv:,.0f}")
```

### Batch Analysis

```python
from forest_carbon import ScenarioManager

# Initialize scenario manager
manager = ScenarioManager()

# Run comprehensive analysis across multiple dimensions
results = manager.run_analysis(
    forest_types=['ETOF', 'EOF', 'AFW'],  # Multiple forest types
    climates=['current', 'paris', 'plus2'],  # Climate scenarios
    managements=['l', 'm', 'i'],  # Management levels
    years=25,
    workers=4,  # Parallel processing
    generate_plots=True,
    seed=123  # Reproducible results
)

# Access comprehensive results
summary = results['summary']
batch_results = results['batch_results']  # CSV with all scenarios
plots_dir = results['plots_directory']
analysis_dir = results['analysis_path']

# Individual scenario results
for scenario_name, scenario_data in results['scenario_results'].items():
    print(f"Scenario: {scenario_name}")
    print(f"  Final carbon: {scenario_data['final_co2e_stock']:.1f} tCO2e/ha")
    print(f"  NPV: ${scenario_data['npv_per_ha']:,.0f}/ha")
    print(f"  IRR: {scenario_data['irr']:.1f}%")
```

### Uncertainty Analysis

```python
from forest_carbon import ForestCarbonSimulator

# Initialize simulator with uncertainty enabled
sim = ForestCarbonSimulator(
    forest_type='ETOF',
    years=25,
    area_ha=1000.0,
    enable_uncertainty=True,
    seed=456  # Reproducible uncertainty analysis
)

# Run uncertainty analysis
uncertainty_results = sim.run_uncertainty_analysis(
    n_iterations=1000,
    parameter_distributions={
        'M_max_biomass': {'type': 'normal', 'cv': 0.15},
        'G_age_max_growth': {'type': 'normal', 'cv': 0.10},
        'y_growth_multiplier': {'type': 'normal', 'cv': 0.20},
        'fpi_multiplier': {'type': 'normal', 'cv': 0.25},
        'mortality_rate': {'type': 'beta', 'alpha': 2, 'beta': 5}
    }
)

# Access uncertainty statistics
uncertainty_stats = uncertainty_results['statistics']
confidence_intervals = uncertainty_results['confidence_intervals']

# Print key uncertainty metrics
print("Uncertainty Analysis Results:")
print(f"Mean final carbon: {uncertainty_stats['mean_final_carbon']:.1f} tCO2e/ha")
print(f"90% CI: {confidence_intervals['final_carbon']['5th']:.1f} - {confidence_intervals['final_carbon']['95th']:.1f} tCO2e/ha")
print(f"Mean NPV: ${uncertainty_stats['mean_npv']:,.0f}/ha")
```

---

## 🎯 Advanced Features

### Uncertainty Analysis

Monte Carlo analysis for parameter uncertainty quantification:

```python
from forest_carbon import ForestCarbonSimulator

simulator = ForestCarbonSimulator(
    forest_type='ETOF',
    years=25,
    uncertainty_analysis=True,
    n_iterations=1000,
    seed=42  # Reproducible uncertainty analysis
)
results = simulator.run()
```

**Uncertainty parameters sampled:**
- Maximum biomass (M): ±15% CV
- Growth multiplier (y): ±20% CV
- FPI multiplier: ±25% CV
- Mortality rates: Beta distribution
- Disturbance parameters: Gamma distribution

**Output:** 50% and 90% confidence intervals, parameter correlation matrices, sensitivity rankings

### Economic Analysis

NPV and IRR calculations aligned with carbon markets:

```python
# Net Present Value
NPV = -C₀ + Σ[(Rₜ - Cₜ)/(1+r)ᵗ] + VT/(1+r)ᵀ

# Carbon Credits (within crediting period)
Creditsₜ = max(0, ΔCₜ) × (1-b) × P_CO2(t)
```

Where:
- C₀ = Initial establishment cost
- Rₜ = Carbon credit revenue (year t)
- Cₜ = Management costs (year t)
- VT = Terminal value
- r = Real discount rate
- b = Buffer withholding (default 0.20)
- P_CO2(t) = Time-varying carbon price

**Economic Parameters:** See Table 4 above

### Custom Analysis

```python
from forest_carbon import ForestCarbonSimulator, ScenarioManager, ComprehensiveAnalyzer

# Single simulation
simulator = ForestCarbonSimulator(
    forest_type='ETOF',
    years=25,
    area_ha=1000,
    output_dir='output',
    seed=42
)
results = simulator.run(generate_plots=True)

# Scenario analysis
manager = ScenarioManager()
results = manager.run_analysis(
    forest_types=['ETOF', 'EOF'],
    climates=['current', 'paris'],
    managements=['l', 'm', 'i'],
    years=25,
    seed=42
)

# Comprehensive analysis
analyzer = ComprehensiveAnalyzer(results_path='output/batch_results.csv')
analyzer.run_complete_analysis()
```

---

## ⚠️ Known Limitations (Section 6 of Technical Paper)

### Model Limitations

1. **Spatial Simplification**: No landscape connectivity or edge effects
2. **Static Parameters**: Climate and management effects don't vary with forest age
3. **Linear Climate Response**: May underestimate threshold effects
4. **Simplified Mortality**: Doesn't capture pest/disease dynamics
5. **Carbon Pool Transfers**: Fixed rates don't respond to environmental conditions
6. **Constant Management Intensity**: v0.1 doesn't model temporal de-escalation of AFM

### Key Assumptions Requiring Review

**Climate Sensitivity:**
- 10% productivity loss per degree warming based on temperate forest studies (Wood et al. 2015, Wardlaw 2021)
- May be conservative for some forest types, optimistic for others

**Management Effectiveness:**
- 20-30% growth improvement assumes moderate intervention (Paul et al. 2018, Paul & Roxburgh 2025)
- Intensive management might achieve 40-50%, low management only 10-15%

**Disturbance Return Intervals:**
- Fire frequencies based on historical data may not reflect future climate-driven changes
- See Table 3 for current climate adjustment factors

### Validation Status (Section 4.2)

**Validated Components:**
- Tree Yield Formula implementation (matches FullCAM specifications)
- IPCC carbon accounting framework
- Economic calculation logic

**Requires Validation:**
- Integrated carbon pool transfers over time
- Climate sensitivity responses in practice
- Management effect magnitudes in real projects
- Multi-year projection accuracy
- Economic projections against actual project outcomes

**Critical Limitation:** Without field validation, FCL should be used for preliminary screening and scenario comparison only, not as the sole basis for project investment decisions or MRV reporting.

---

## 🤝 Contributing

We welcome contributions in several forms:

### 1. Validation Partnerships
- Share multi-year carbon monitoring data from restoration sites
- ACCU issuance records from completed projects
- Chronosequence data across climate gradients
- Realized project economics and carbon credit revenues

### 2. Code Contributions
1. Follow the unified package structure
2. Add comprehensive docstrings
3. Include type hints
4. Test with multiple scenarios
5. Update documentation

### 3. Scientific Review
- Review mathematical formulations (Section 2 of paper)
- Challenge parameter assumptions (Section 6 of paper)
- Suggest additional validation approaches (Section 4.2.3 of paper)

### 4. Feature Requests
Priority areas for v0.2 development:
- Dynamic parameter adjustment with forest age
- Non-linear climate response options
- Spatial connectivity effects
- Enhanced fire modeling
- Biodiversity co-benefits tracking
- Temporal de-escalation of AFM interventions

---

## 📚 References

### Core Model Development

**FullCAM & Tree Yield Formula:**
- Forrester et al. (2025) "Calibration of the FullCAM model for Australian native vegetation." Ecological Modelling 508: 111204
- Paul & Roxburgh (2020) "Predicting carbon sequestration of woody biomass following land restoration." Forest Ecology and Management 460: 117838
- Paul & Roxburgh (2025) "Carbon sequestration in woody biomass of mulga (Acacia aneura) woodlands: confidence in prediction using the carbon accounting model FullCAM." The Rangeland Journal 47(3)
- Waterworth et al. (2007) "A generalised hybrid process-empirical model for predicting plantation forest growth." Forest Ecology and Management 238: 231-243

**Maximum Biomass & Site Productivity:**
- Roxburgh et al. (2019) "A revised above-ground maximum biomass layer for the Australian continent." Forest Ecology and Management 432: 264-275
- Kesteven and Landsberg (2004) "Developing a national forest productivity model for Australia." NCAS Technical Report No. 27, Australian Greenhouse Office

**Environmental Plantings:**
- Paul et al. (2018) "Using measured stocks of biomass and litter carbon to constrain modelled estimates of sequestration of soil organic carbon under contrasting mixed-species environmental plantings." Science of The Total Environment 615: 348-359

### Active Forest Management

**AFM Framework:**
- Bennett et al. (2024) "Active Management: A Definition and Considerations for Implementation in Forests of Temperate Australia." Australian Forestry 87(3): 125-147

### Climate & Fire

**Climate Impacts:**
- IPCC (2023) Climate Change 2023: Synthesis Report. Geneva: IPCC
- Wardlaw (2021) "The effect of climate change on the health and productivity of Australia's temperate eucalypt forests." Australian Forestry

**Fire Regime Changes:**
- Boer et al. (2021) "Multi-decadal increase of forest burned area in Australia is linked to climate change." Nature Communications 12: 6921
- Furlaud et al. (2021) "Bioclimatic drivers of fire severity across the Australian geographical range of giant Eucalyptus forests." Journal of Ecology 109: 2444-2457
- McColl-Gausden et al. (2022) "The fuel-climate-fire conundrum: How will fire regimes change in temperate eucalypt forests under climate change?" Global Change Biology 28(17): 5211-5226

### Economics & Carbon Markets

**Cost-Effectiveness:**
- Austin et al. (2020) "The economic costs of planting, preserving, and managing the world's forests to mitigate climate change." Nature Communications 11: 5946
- Busch et al. (2024) "Cost-effectiveness of natural forest regeneration." Nature Climate Change 14(9): 996-1002
- Evans (2018) "Effective incentives for reforestation: lessons from Australia's carbon farming policies." Current Opinion in Environmental Sustainability 32: 38-45
- Jonson & Freudenberger (2011) "Restore and sequester: Estimating biomass in native Australian woodland ecosystems for their carbon-funded restoration." Australian Journal of Botany 59(7): 640
- Pacheco et al. (2024) [Add full citation if available]

**Global Carbon Sequestration:**
- Cook-Patton et al. (2020) "Mapping carbon accumulation potential from global natural forest regrowth." Nature 585: 545-550

### Validation Data

**Australian Forest Inventories:**
- Volkova et al. (2015) "Empirical estimates of aboveground carbon in open eucalyptus forests of south-eastern Australia and its potential implication for national carbon accounting." Forests 6(10): 3395-3411

**Additional Parameters:**
- Mokany et al. (2006) "Critical analysis of root:shoot ratios in terrestrial biomes." Global Change Biology 12: 84-96
- Ximenes & Wright (2006) "Forests, wood and Australia's carbon balance." Pre-published. DOI: 10.13140/RG.2.1.3297.4886

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **Healthy Forests Foundation** - Project support and guidance
- **FullCAM Development Team** - Australian Government forest carbon model
- **Scientific Community** - Peer review and validation partnerships sought

---

## 📧 Contact

**Author:** Pia Angelike  
**Organization:** Healthy Forests Foundation  
**Email:** pia.angelike@healthyforestsfoundation.org

**For validation partnerships, peer review feedback, or collaboration inquiries, please contact us.**

---

## 📚 Glossary

**ACCU:** Australian Carbon Credit Unit - One tonne CO₂e equivalent

**AFM:** Active Forest Management - Temporary restoration interventions in degraded forests that work toward obsolescence

**AFW:** Acacia Forest Woodland - Forest type with lower biomass potential (~49 t/ha)

**AGB/BGB:** Above/Below-ground biomass - Living tree biomass components

**EOF:** Eucalypt Open Forest - Forest type with moderate biomass potential (~170 t/ha)

**ETOF:** Eucalypt Tall Open Forest - Forest type with high biomass potential (~290 t/ha)

**FPI:** Forest Productivity Index - Climate effect metric that adjusts growth based on temperature and rainfall

**FullCAM:** Full Carbon Accounting Model - Australian Government's national forest carbon model

**IRR:** Internal Rate of Return - Economic metric for project profitability

**MRV:** Monitoring, Reporting, and Verification - Carbon credit project requirements

**NPV:** Net Present Value - Economic metric for project value

**RMSE:** Root Mean Square Error - Statistical accuracy metric

**TYF:** Tree Yield Formula - Mathematical growth equation validated against 9,300+ plots

**Validation:** Independent verification of model predictions against field measurements

---

**Forest Carbon Lite v0.1** - Professional Review Draft for scenario exploration and preliminary analysis. Built on validated components (FullCAM TYF), requiring integrated system validation before operational deployment. Seeking validation partnerships and peer review feedback.