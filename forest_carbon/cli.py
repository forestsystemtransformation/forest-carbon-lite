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
        fcl simulate --forest ETOF --years 25 --plot --seed 42
        
        # Run scenario analysis (natural reforestation) with reproducibility
        fcl analyze --forest-types ETOF,EOF --climates current,paris_target --years 25 --seed 123
        
        # Run scenario analysis with managed reforestation and uncertainty
        fcl analyze --forest-types ETOF --climates current --managements intensive_managed_reforestation --years 25 --uncertainty --seed 456
        
        # Compare natural vs managed reforestation (reproducible)
        fcl analyze --forest-types ETOF --climates current --managements intensive,intensive_managed_reforestation --years 25 --seed 789
        
        # Run comprehensive analysis on existing results
        fcl comprehensive --results-path output/batch_results.csv
        
        # Reproducible uncertainty analysis
        fcl simulate --forest ETOF --years 25 --uncertainty --seed 42

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
@click.option('--forest', required=True, 
              type=click.Choice(['EOF', 'ETOF', 'AFW', 'EW_OW', 'Mallee', 'Other_FW', 'Rainforest', 'Shrubland']),
              help='Forest type')
@click.option('--years', type=int, default=25, help='Simulation years')
@click.option('--area', type=float, default=1000.0, help='Area in hectares')
@click.option('--output', type=str, default='output', help='Output directory')
@click.option('--config', type=str, help='Custom config file')
@click.option('--climate', type=str, help='Climate configuration')
@click.option('--plot', is_flag=True, help='Generate plots')
@click.option('--uncertainty/--no-uncertainty', default=False, help='Enable or disable uncertainty analysis')
@click.option('--no-validate', is_flag=True, help='Disable validation')
@click.option('--seed', type=int, help='Random seed for reproducibility')
def simulate(forest, years, area, output, config, climate, plot, uncertainty, no_validate, seed):
    """Run single simulation."""
    print("🌲 Forest Carbon Lite - Single Simulation")
    print("=" * 50)
    
    simulator = ForestCarbonSimulator(
        forest_type=forest,
        years=years,
        area_ha=area,
        output_dir=output,
        config_file=config,
        climate_config=climate,
        validate_config=not no_validate,
        enable_uncertainty=uncertainty,
        seed=seed
    )
    
    results = simulator.run(generate_plots=plot)
    
    print(f"\n✅ Simulation completed!")
    print(f"📁 Results saved to: {output}/")
    
    if plot:
        print(f"📊 Plots saved to: {output}/plots/")
    
    if uncertainty:
        print(f"📈 Uncertainty analysis saved to: {output}/[scenario_name]/uncertainty_analysis/")

@main.command()
@click.option('--forest-types', type=str, default='ETOF,EOF,AFW',
              help='Comma-separated forest types')
@click.option('--climates', type=str, default='current,paris_target,paris_overshoot',
              help='Comma-separated climate scenarios')
@click.option('--managements', type=str, default='baseline,adaptive,intensive',
              help='Comma-separated management levels (baseline,moderate,adaptive,intensive,intensive_managed_reforestation)')
@click.option('--years', type=int, default=25, help='Simulation years')
@click.option('--workers', type=int, default=4, help='Number of workers')
@click.option('--plots', is_flag=True, help='Generate individual plots')
@click.option('--uncertainty/--no-uncertainty', default=False, help='Enable or disable uncertainty analysis')
@click.option('--output-dir', type=str, default='output', help='Output directory')
@click.option('--seed', type=int, help='Random seed for reproducibility')
def analyze(forest_types, climates, managements, years, workers, plots, uncertainty, output_dir, seed):
    """Run scenario analysis."""
    print("🌲 Forest Carbon Lite - Scenario Analysis")
    print("=" * 50)
    
    manager = ScenarioManager()
    
    # Parse forest types, climates, and managements
    forest_types_list = [f.strip() for f in forest_types.split(',')]
    climates_list = [c.strip() for c in climates.split(',')]
    managements_list = [m.strip() for m in managements.split(',')]
    
    results = manager.run_analysis(
        forest_types=forest_types_list,
        climates=climates_list,
        managements=managements_list,
        years=years,
        workers=workers,
        generate_plots=plots,
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
    print("🌲 Forest Carbon Lite - Comprehensive Analysis")
    print("=" * 50)
    
    analyzer = ComprehensiveAnalyzer(results_path=Path(results_path))
    analyzer.run_complete_analysis(output_dir=Path(output_dir))
    
    print(f"\n✅ Comprehensive analysis completed!")
    print(f"📁 Results saved to: {output_dir}/")

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

