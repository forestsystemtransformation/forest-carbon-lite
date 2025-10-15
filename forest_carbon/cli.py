#!/usr/bin/env python3
"""
Command-line interface for Forest Carbon Lite.

This module provides the main CLI entry points for the forest carbon simulation system.
"""

import sys
import click
from pathlib import Path
from typing import Optional

from forest_carbon.core.simulator import ForestCarbonSimulator
from forest_carbon.scenarios.manager import ScenarioManager
from forest_carbon.analysis.comprehensive import ComprehensiveAnalyzer

@click.group()
@click.version_option(version='8.0.0')
def main():
    """Forest Carbon Lite - Dynamic Carbon Sequestration Simulator
    
    A comprehensive Python implementation of a "FullCAM-lite" forest carbon accounting model,
    featuring the Tree Yield Formula (TYF) growth engine with dynamic scenario generation
    and climate change integration.
    
    Examples:
        # Run single simulation with reproducibility
        fcl simulate --site ETOF --years 25 --plot --seed 42
        
        # Run scenario analysis (natural reforestation) with reproducibility
        fcl analyze --site ETOF,EOF --climate current,paris --years 25 --seed 123
        
        # Run scenario analysis with managed reforestation and uncertainty
        fcl analyze --site ETOF --climate current --management intensive_managed_reforestation --years 25 --uncertainty --seed 456
        
        # Compare natural vs managed reforestation (reproducible)
        fcl analyze --site ETOF --climate current --management intensive,intensive_managed_reforestation --years 25 --seed 789
        
        # Run comprehensive analysis on existing results
        fcl comprehensive --results-path output/batch_results.csv
        
        # Reproducible uncertainty analysis
        fcl simulate --site ETOF --years 25 --uncertainty --seed 42

    Management Levels:
        baseline                    - No management effects (degraded baseline)
        moderate                    - Moderate management effects
        adaptive                    - Adaptive management effects  
        intensive                   - Intensive management effects (natural reforestation)
        intensive_managed_reforestation - Intensive management effects (managed reforestation)
    
    Reproducibility:
        Use --seed parameter for reproducible results across runs.
        Same seed = identical results, different seeds = different stochastic outcomes.
        Essential for scientific research and experiment replication.
    """
    pass

@main.command()
@click.option('--site', required=True, 
              type=click.Choice(['EOF', 'ETOF', 'AFW', 'EW_OW', 'Mallee', 'Other_FW', 'Rainforest', 'Shrubland']),
              help='Site/forest type')
@click.option('--years', type=int, default=25, help='Simulation years')
@click.option('--area', type=float, default=1000.0, help='Area in hectares')
@click.option('--output', type=str, default='output', help='Output directory')
@click.option('--config', type=str, help='Custom config file')
@click.option('--climate', type=str, help='Climate configuration')
@click.option('--plot', is_flag=True, help='Generate plots')
@click.option('--optional-plots', is_flag=True, help='Include optional specialized plots (reforestation minus losses, management minus reforestation)')
@click.option('--uncertainty/--no-uncertainty', default=False, help='Enable or disable uncertainty analysis')
@click.option('--no-validate', is_flag=True, help='Disable validation')
@click.option('--seed', type=int, help='Random seed for reproducibility')
def simulate(site, years, area, output, config, climate, plot, optional_plots, uncertainty, no_validate, seed):
    """Run single simulation."""
    print("Forest Carbon Lite - Single Simulation")
    print("=" * 50)
    
    simulator = ForestCarbonSimulator(
        forest_type=site,
        years=years,
        area_ha=area,
        output_dir=output,
        config_file=config,
        climate_config=climate,
        validate_config=not no_validate,
        enable_uncertainty=uncertainty,
        seed=seed
    )
    
    results = simulator.run(generate_plots=plot, include_optional_plots=optional_plots)
    
    print(f"\n✅ Simulation completed!")
    print(f"📁 Results saved to: {output}/")
    
    if plot:
        print(f"📊 Plots saved to: {output}/plots/")
    
    if uncertainty:
        print(f"📈 Uncertainty analysis saved to: {output}/[scenario_name]/uncertainty_analysis/")

@main.command()
@click.option('--site', type=str, default='ETOF,EOF,AFW',
              help='Comma-separated site/forest types')
@click.option('--climate', type=str, default='current,paris_target,paris_overshoot',
              help='Comma-separated climate scenarios')
@click.option('--management', type=str, default='baseline,adaptive,intensive',
              help='Comma-separated management levels (baseline,moderate,adaptive,intensive,intensive_managed_reforestation)')
@click.option('--years', type=int, default=25, help='Simulation years')
@click.option('--workers', type=int, default=4, help='Number of workers')
@click.option('--plot', is_flag=True, help='Generate individual plots')
@click.option('--optional-plots', is_flag=True, help='Include optional specialized plots (reforestation minus losses, management minus reforestation)')
@click.option('--uncertainty/--no-uncertainty', default=False, help='Enable or disable uncertainty analysis')
@click.option('--output-dir', type=str, default='output', help='Output directory')
@click.option('--seed', type=int, help='Random seed for reproducibility')
def analyze(site, climate, management, years, workers, plot, optional_plots, uncertainty, output_dir, seed):
    """Run scenario analysis."""
    print("Forest Carbon Lite - Scenario Analysis")
    print("=" * 50)
    
    manager = ScenarioManager()
    
    # Parse forest types, climates, and managements
    forest_types_list = [f.strip() for f in site.split(',')]
    climates_list = [c.strip() for c in climate.split(',')]
    managements_list = [m.strip() for m in management.split(',')]
    
    results = manager.run_analysis(
        forest_types=forest_types_list,
        climates=climates_list,
        managements=managements_list,
        years=years,
        workers=workers,
        generate_plots=plot,
        include_optional_plots=optional_plots,
        enable_uncertainty=uncertainty,
        output_dir=output_dir,
        seed=seed
    )
    
    print(f"\n✅ Scenario analysis completed!")
    print(f"📁 Results saved to: {output_dir}/")
    print(f"📊 Analysis report: {output_dir}/analysis/")

@main.command()
@click.option('--results-path', type=str, default='output/batch_results.csv',
              help='Path to batch results CSV')
@click.option('--output-dir', type=str, default='output_years/analysis',
              help='Output directory for analysis')
def comprehensive(results_path, output_dir):
    """Run comprehensive analysis on existing results."""
    print("Forest Carbon Lite - Comprehensive Analysis")
    print("=" * 50)
    
    analyzer = ComprehensiveAnalyzer(results_path=Path(results_path))
    analyzer.run_complete_analysis(output_dir=Path(output_dir))
    
    print(f"\n✅ Comprehensive analysis completed!")
    print(f"📁 Results saved to: {output_dir}/")

@main.command()
@click.option('--site', type=str, default='ETOF', help='Site/forest type')
@click.option('--climate', type=str, default='current', help='Climate scenario')
@click.option('--management', type=str, default='intensive', help='Management level')
@click.option('--years', type=int, default=52, help='Simulation years')
@click.option('--area', type=float, default=1000.0, help='Area in hectares')
@click.option('--output-dir', type=str, help='Output directory (auto-generated if not specified)')
@click.option('--no-plots', is_flag=True, help='Skip plot generation')
def afm(site, climate, management, years, area, output_dir, no_plots):
    """Run AFM vs Degrading analysis (no reforestation scenarios)."""
    import subprocess
    import sys
    
    # Build command for the AFM simulator
    cmd = [
        sys.executable, 
        'forest_carbon/tools/custom_afm_simulator.py',
        '--site', site,
        '--climate', climate,
        '--management', management,
        '--years', str(years),
        '--area', str(area)
    ]
    
    if output_dir:
        cmd.extend(['--output-dir', output_dir])
    
    if no_plots:
        cmd.append('--no-plots')
    
    # Run the AFM simulator
    subprocess.run(cmd)

@main.command(name='plot-matrix')
@click.option('--list', 'list_available', is_flag=True, help='List available scenarios and plot types')
@click.option('--plot-type', type=str, help='Plot type to compare across scenarios')
@click.option('--scenario', type=str, help='Scenario to show all plots for')
@click.option('--scenarios', type=str, help='Comma-separated scenarios for custom matrix')
@click.option('--plot-types', type=str, help='Comma-separated plot types for custom matrix')
@click.option('--keep-legends', is_flag=True, help='Keep legends (default: auto-crop)')
@click.option('--max-per-row', type=int, default=4, help='Maximum plots per row')
def plot_matrix(list_available, plot_type, scenario, scenarios, plot_types, keep_legends, max_per_row):
    """Create plot comparison matrices."""
    import subprocess
    import sys
    
    cmd = [sys.executable, 'forest_carbon/tools/plot_matrix_generator.py']
    
    if list_available:
        cmd.append('--list')
    elif plot_type:
        cmd.extend(['--plot-type', plot_type])
    elif scenario:
        cmd.extend(['--scenario', scenario])
    elif scenarios:
        cmd.extend(['--scenarios', scenarios])
        if plot_types:
            cmd.extend(['--plot-types', plot_types])
    
    if keep_legends:
        cmd.append('--keep-legends')
    
    cmd.extend(['--max-per-row', str(max_per_row)])
    
    subprocess.run(cmd)

@main.command(name='data-matrix')
@click.option('--list', 'list_available', is_flag=True, help='List available scenarios and matrix types')
@click.option('--scenarios', type=str, required=True, help='Comma-separated scenarios')
@click.option('--matrix-type', type=click.Choice(['carbon_stocks', 'additionality', 'economics', 'combined']), 
              required=True, help='Type of matrix to generate')
@click.option('--max-per-row', type=int, default=3, help='Maximum plots per row')
def data_matrix(list_available, scenarios, matrix_type, max_per_row):
    """Create data-driven comparison matrices."""
    import subprocess
    import sys
    
    cmd = [sys.executable, 'forest_carbon/tools/data_matrix_generator.py']
    
    if list_available:
        cmd.append('--list')
    else:
        cmd.extend(['--scenarios', scenarios])
        cmd.extend(['--matrix-type', matrix_type])
        cmd.extend(['--max-per-row', str(max_per_row)])
    
    subprocess.run(cmd)

# Create individual command functions for the console scripts
def simulate_command():
    """Entry point for forest-carbon-simulate console script."""
    simulate.main(prog_name='forest-carbon-simulate')

def analyze_command():
    """Entry point for forest-carbon-analyze console script."""
    analyze.main(prog_name='forest-carbon-analyze')

def comprehensive_command():
    """Entry point for forest-carbon-comprehensive console script."""
    comprehensive.main(prog_name='forest-carbon-comprehensive')

if __name__ == "__main__":
    main()

