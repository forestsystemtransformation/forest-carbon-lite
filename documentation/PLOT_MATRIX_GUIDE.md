# Plot Matrix Generator - Usage Guide

The Plot Matrix Generator creates comparison matrices by pulling existing plots from different scenarios into one PNG file. **No new computations needed** - it simply arranges existing plot images in a grid format.

## Key Features

✅ **Legend Cropping**: Automatically crops repetitive legends from individual plots (default behavior)  
✅ **Multiple Matrix Types**: Plot-type matrices, scenario matrices, and custom matrices  
✅ **Flexible Layout**: Control grid size and arrangement  
✅ **High Quality**: 300 DPI output suitable for publications  

## Quick Start

### 1. List Available Options
```bash
python plot_matrix_generator.py --list
```

### 2. Create Matrices with Cropped Legends (Default)
```bash
# Compare one plot type across all scenarios
python plot_matrix_generator.py --plot-type total_carbon_stocks_all_scenarios

# Show all plots for one scenario
python plot_matrix_generator.py --scenario ETOF_burnt_paris_overshoot_intensive

# Custom matrix
python plot_matrix_generator.py --scenarios SCENARIO1 SCENARIO2 --plot-types PLOT1 PLOT2
```

### 3. Keep Legends (If Needed)
```bash
# Same commands but with --keep-legends flag
python plot_matrix_generator.py --plot-type additionality --keep-legends
python plot_matrix_generator.py --scenario ETOF_paris_target_moderate --keep-legends
python plot_matrix_generator.py --scenarios SCENARIO1 SCENARIO2 --plot-types PLOT1 PLOT2 --keep-legends
```

## Legend Cropping Details

**Default Behavior**: Legends are automatically cropped from the right 25% of each plot to reduce clutter in matrix format.

**When to Use Each Option**:
- **Cropped legends** (default): Best for comparing patterns across scenarios
- **Keep legends** (`--keep-legends`): When you need to see exact values or legend details

## Available Plot Types

1. `additionality` - Carbon Additionality
2. `biomass_all_scenarios` - Biomass All Scenarios  
3. `carbon_pools_breakdown_baseline` - Carbon Pools - Baseline
4. `carbon_pools_breakdown_management` - Carbon Pools - Management
5. `carbon_pools_breakdown_reforestation` - Carbon Pools - Reforestation
6. `carbon_pools_comparison` - Carbon Pools Comparison
7. `economics_management` - Economics - Management
8. `economics_reforestation` - Economics - Reforestation
9. `management_minus_reforestation` - Management vs Reforestation
10. `reforestation_minus_losses` - Reforestation Minus Losses
11. `total_carbon_stocks_all_scenarios` - Total Carbon Stocks

## Example Commands

### Compare Carbon Stocks Across Climate Scenarios
```bash
python plot_matrix_generator.py --scenarios ETOF_burnt_paris_overshoot_intensive ETOF_burnt_paris_target_intensive ETOF_paris_overshoot_intensive --plot-types total_carbon_stocks_all_scenarios
```

### Compare Management Intensity
```bash
python plot_matrix_generator.py --scenarios ETOF_burnt_paris_overshoot_intensive ETOF_burnt_paris_overshoot_moderate --plot-types additionality carbon_pools_comparison economics_management
```

### Show All Plots for One Scenario
```bash
python plot_matrix_generator.py --scenario ETOF_burnt_paris_overshoot_intensive --max-per-row 4
```

### Control Grid Layout
```bash
python plot_matrix_generator.py --plot-type additionality --max-per-row 6
```

## Output Files

All matrices are saved to the `output_matrix` folder with descriptive names:

- `total_carbon_stocks_all_scenarios_matrix_comparison.png` - All scenarios for one plot type
- `ETOF_burnt_paris_overshoot_intensive_all_plots_matrix.png` - All plots for one scenario  
- `custom_matrix_3x2.png` - Custom 3 scenarios × 2 plot types matrix

## Tips

1. **Start with default settings** - legend cropping makes matrices much cleaner
2. **Use `--keep-legends`** only when you need to see exact values
3. **Adjust `--max-per-row`** for different grid layouts (default: 4)
4. **Run without arguments** to generate useful default matrices

## Requirements

- Python 3.7+
- matplotlib
- PIL (Pillow)
- numpy

The script automatically detects available scenarios and plot types from your `output` directory.
