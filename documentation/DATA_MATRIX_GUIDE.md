# Data Matrix Generator - Usage Guide

The Data Matrix Generator (`data_matrix_generator.py`) creates comparison matrices by generating **new plots from CSV data** rather than pulling existing plot images. This provides more flexibility for custom data-driven comparisons and allows you to create matrices even when individual plot images don't exist.

## Key Features

✅ **Data-Driven**: Creates plots directly from CSV data files  
✅ **Multiple Matrix Types**: Carbon stocks, additionality, and economics matrices  
✅ **Flexible Visualization**: Custom plot generation with consistent styling  
✅ **High Quality**: 300 DPI output suitable for publications  
✅ **No Dependencies on Existing Plots**: Works even if individual plot images are missing  

## Quick Start

### 1. List Available Options
```bash
python data_matrix_generator.py --list
```

### 2. Create Different Types of Matrices
```bash
# Carbon stocks comparison matrix
python data_matrix_generator.py --scenarios ETOF_degraded_paris_target_intensive ETOF_degraded_current_intensive --matrix-type carbon_stocks

# Additionality comparison matrix  
python data_matrix_generator.py --scenarios ETOF_degraded_paris_target_intensive ETOF_degraded_current_intensive --matrix-type additionality

# Economics comparison matrix
python data_matrix_generator.py --scenarios ETOF_degraded_paris_target_intensive ETOF_degraded_current_intensive --matrix-type economics

# Combined matrix showing both carbon stocks and additionality
python data_matrix_generator.py --scenarios ETOF_degraded_paris_target_intensive ETOF_degraded_current_intensive --matrix-type combined

# Multi-row matrices (2x3, 3x3, etc.)
python data_matrix_generator.py --scenarios SCENARIO1 SCENARIO2 SCENARIO3 SCENARIO4 SCENARIO5 SCENARIO6 --matrix-type carbon_stocks --max-per-row 3
```

## Available Matrix Types

### 1. **Carbon Stocks Matrix** (`--matrix-type carbon_stocks`)
- **Data Source**: `sequestration_curves.csv`
- **Shows**: All three scenarios (baseline, management, reforestation) for each scenario
- **Use Case**: Compare total carbon accumulation across different scenarios
- **Output**: `carbon_stocks_data_matrix_Xscenarios.png`

### 2. **Additionality Matrix** (`--matrix-type additionality`)
- **Data Source**: `sequestration_curves.csv`
- **Shows**: Carbon additionality (difference from baseline or vs 0 for reforestation)
- **Use Case**: Compare the additional carbon benefits of interventions
- **Output**: `additionality_data_matrix_Xscenarios.png`

### 3. **Economics Matrix** (`--matrix-type economics`)
- **Data Source**: `results_summary.csv`
- **Shows**: NPV and other economic metrics as bar charts
- **Use Case**: Compare economic performance across scenarios
- **Output**: `economics_data_matrix_Xscenarios.png`

### 4. **Combined Matrix** (`--matrix-type combined`)
- **Data Source**: `sequestration_curves.csv`
- **Shows**: Both carbon stocks AND additionality side-by-side for each scenario
- **Use Case**: Comprehensive analysis showing both absolute values and benefits
- **Output**: `combined_carbon_additionality_matrix_Xscenarios.png`

## Available Data Files

The generator automatically detects and uses these CSV files:
- `sequestration_curves.csv` - Carbon stock trajectories over time
- `results_summary.csv` - Summary results including economic metrics
- `finance_results.csv` - Detailed financial analysis
- `cashflow_breakdown.csv` - Annual cashflow breakdowns

## Example Commands

### Compare Carbon Stocks Across Climate Scenarios
```bash
python data_matrix_generator.py --scenarios ETOF_degraded_current_intensive ETOF_degraded_paris_target_intensive ETOF_degraded_paris_overshoot_intensive --matrix-type carbon_stocks
```

### Compare Management Intensity
```bash
python data_matrix_generator.py --scenarios ETOF_degraded_current_intensive ETOF_degraded_current_moderate --matrix-type additionality
```

### Compare Economic Performance
```bash
python data_matrix_generator.py --scenarios ETOF_degraded_paris_target_intensive ETOF_degraded_paris_target_moderate --matrix-type economics
```

### Combined Analysis (Carbon Stocks + Additionality)
```bash
python data_matrix_generator.py --scenarios ETOF_degraded_current_intensive ETOF_degraded_paris_target_intensive ETOF_degraded_paris_overshoot_intensive --matrix-type combined
```

### Multi-Row Matrices (2x3, 3x3, etc.)
```bash
# 2x3 matrix (6 scenarios in 2 rows of 3)
python data_matrix_generator.py --scenarios ETOF_degraded_paris_target_intensive ETOF_degraded_current_intensive ETOF_degraded_paris_overshoot_intensive ETOF_degraded_paris_target_moderate ETOF_degraded_current_moderate ETOF_degraded_paris_overshoot_moderate --matrix-type carbon_stocks --max-per-row 3

# 3x3 matrix (9 scenarios in 3 rows of 3)
python data_matrix_generator.py --scenarios SCENARIO1 SCENARIO2 SCENARIO3 SCENARIO4 SCENARIO5 SCENARIO6 SCENARIO7 SCENARIO8 SCENARIO9 --matrix-type additionality --max-per-row 3

# Combined matrix in 2x3 layout
python data_matrix_generator.py --scenarios ETOF_degraded_paris_target_intensive ETOF_degraded_current_intensive ETOF_degraded_paris_overshoot_intensive ETOF_degraded_paris_target_moderate ETOF_degraded_current_moderate ETOF_degraded_paris_overshoot_moderate --matrix-type combined --max-per-row 3

# Exact grid control (2x3 matrix)
python data_matrix_generator.py --scenarios SCENARIO1 SCENARIO2 SCENARIO3 SCENARIO4 SCENARIO5 SCENARIO6 --matrix-type combined --grid-rows 2 --grid-cols 3
```

## Output Files

All matrices are saved to the `output_matrix` folder with descriptive names:

- `carbon_stocks_data_matrix_3scenarios.png` - Carbon stocks comparison
- `additionality_data_matrix_2scenarios.png` - Additionality comparison  
- `economics_data_matrix_2scenarios.png` - Economics comparison

## Data Matrix vs Plot Matrix

| Feature | Data Matrix Generator | Plot Matrix Generator |
|---------|----------------------|----------------------|
| **Data Source** | CSV files | Existing PNG images |
| **Flexibility** | High - custom plots | Medium - existing plots only |
| **Speed** | Slower - generates plots | Faster - arranges images |
| **Customization** | Full control over plot style | Limited to existing plot style |
| **Dependencies** | Requires CSV data | Requires existing PNG files |
| **Use Case** | Custom analysis, new visualizations | Quick comparison of existing plots |

## Grid Layout Control

Control the matrix layout using the `--max-per-row` parameter:

### Default Behavior
- **Carbon/Additionality/Economics matrices**: 4 scenarios per row (default)
- **Combined matrices**: 3 scenarios per row (default, because each scenario takes 2 columns)

### Custom Layouts
```bash
# Single row (all scenarios in one row)
python data_matrix_generator.py --scenarios SCENARIO1 SCENARIO2 SCENARIO3 SCENARIO4 --matrix-type carbon_stocks --max-per-row 4

# 2x2 matrix (4 scenarios in 2 rows of 2)
python data_matrix_generator.py --scenarios SCENARIO1 SCENARIO2 SCENARIO3 SCENARIO4 --matrix-type carbon_stocks --max-per-row 2

# 2x3 matrix (6 scenarios in 2 rows of 3)
python data_matrix_generator.py --scenarios SCENARIO1 SCENARIO2 SCENARIO3 SCENARIO4 SCENARIO5 SCENARIO6 --matrix-type carbon_stocks --max-per-row 3

# 3x3 matrix (9 scenarios in 3 rows of 3)
python data_matrix_generator.py --scenarios SCENARIO1 SCENARIO2 SCENARIO3 SCENARIO4 SCENARIO5 SCENARIO6 SCENARIO7 SCENARIO8 SCENARIO9 --matrix-type carbon_stocks --max-per-row 3
```

### Layout Recommendations
- **Small matrices (≤4 scenarios)**: Use `--max-per-row 4` for single row
- **Medium matrices (5-9 scenarios)**: Use `--max-per-row 3` for 2x3 or 3x3 layout
- **Large matrices (10+ scenarios)**: Use `--max-per-row 4` for compact layout
- **Combined matrices**: Use `--max-per-row 3` (each scenario takes 2 columns: stocks + additionality)

### Exact Grid Control
For precise control over the matrix layout, use `--grid-rows` and `--grid-cols`:

```bash
# 2x3 matrix (2 rows, 3 columns)
python data_matrix_generator.py --scenarios SCENARIO1 SCENARIO2 SCENARIO3 SCENARIO4 SCENARIO5 SCENARIO6 --matrix-type carbon_stocks --grid-rows 2 --grid-cols 3

# 3x2 matrix (3 rows, 2 columns)  
python data_matrix_generator.py --scenarios SCENARIO1 SCENARIO2 SCENARIO3 SCENARIO4 SCENARIO5 SCENARIO6 --matrix-type additionality --grid-rows 3 --grid-cols 2

# 2x3 combined matrix (2 rows, 3 columns of scenarios, each taking 2 plot columns)
python data_matrix_generator.py --scenarios SCENARIO1 SCENARIO2 SCENARIO3 SCENARIO4 SCENARIO5 SCENARIO6 --matrix-type combined --grid-rows 2 --grid-cols 3
```

### Combined Matrix Layout Clarification
The combined matrix is special because each scenario gets **2 plots side-by-side** (carbon stocks + additionality):

- `--max-per-row 2` = 2 scenarios per row = 4 total columns (2 scenarios × 2 plots each)
- `--max-per-row 3` = 3 scenarios per row = 6 total columns (3 scenarios × 2 plots each)

**Example**: 3 scenarios with `--max-per-row 2` creates:
- Row 1: Scenario 1 (stocks + additionality) + Scenario 2 (stocks + additionality)
- Row 2: Scenario 3 (stocks + additionality) + empty space

## Tips

1. **Start with Carbon Stocks**: Most comprehensive view of scenario differences
2. **Use Additionality for Benefits**: Shows the actual carbon benefits of interventions
3. **Check Economics for Viability**: Compare financial performance across scenarios
4. **Use Multi-Row Layouts**: Better use of screen space for 6+ scenarios
5. **Combine with Plot Matrix**: Use both generators for comprehensive analysis

## Requirements

- Python 3.7+
- matplotlib
- pandas
- numpy
- forest_carbon package (for color management)

## Error Handling

The generator will:
- Skip scenarios without required data files
- Show clear error messages for missing scenarios
- Continue processing valid scenarios even if some fail

## Advanced Usage

### Custom Data Analysis
You can extend the generator by adding new matrix types:
1. Add new data file types to `self.data_files`
2. Create new matrix methods following the existing pattern
3. Add new choices to the `--matrix-type` argument

### Integration with Existing Workflow
```bash
# Generate data matrices for analysis
python data_matrix_generator.py --scenarios SCENARIO1 SCENARIO2 --matrix-type carbon_stocks

# Generate plot matrices for presentation  
python plot_matrix_generator.py --scenarios SCENARIO1 SCENARIO2 --plot-types total_carbon_stocks_all_scenarios

# Both create matrices in output_matrix/ for easy comparison
```
