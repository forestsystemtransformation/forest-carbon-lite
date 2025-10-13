#!/usr/bin/env python3
"""
Main Script for Forest Carbon Scenario Analysis

This script orchestrates the complete scenario analysis workflow:
1. Generate scenario configurations
2. Run simulations in parallel
3. Analyze results and generate reports

Usage:
    python -m forest_carbon.scenarios.manager [--forest-types ETOF,EOF,AFW] [--climates current,paris_target,paris_overshoot] [--managements baseline,adaptive,intensive] [--years 25] [--workers 4] [--plots]
"""

import argparse
from pathlib import Path
import pandas as pd
import sys
from typing import List, Optional

from .builder import (
    ScenarioBuilder, 
    ForestType, 
    ClimateScenario, 
    ManagementLevel,
    ScenarioGenerator
)
from .runner import BatchRunner
from .analyzer import ScenarioAnalyzer

class ScenarioManager:
    """Main orchestrator for scenario analysis workflow."""
    
    def __init__(self):
        """Initialize scenario manager."""
        self.builder = ScenarioBuilder()
        self.generator = ScenarioGenerator()
        self.runner = BatchRunner()
        self.analyzer = None
    
    def run_analysis(self, forest_types: List[str], climates: List[str], 
                    managements: List[str], years: int = 25, 
                    workers: int = 4, generate_plots: bool = False,
                    enable_uncertainty: bool = False, output_dir: str = "output",
                    seed: Optional[int] = None) -> dict:
        """
        Run complete scenario analysis workflow.
        
        Args:
            forest_types: List of forest types to analyze
            climates: List of climate scenarios to analyze
            managements: List of management levels to analyze
            years: Simulation duration in years
            workers: Number of parallel workers
            generate_plots: Whether to generate individual plots
            enable_uncertainty: Whether to enable uncertainty analysis
            output_dir: Output directory for results
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with analysis results and metadata
        """
        print("🌲 Forest Carbon Scenario Analysis System")
        print("=" * 60)
        
        print(f"\nConfiguration:")
        print(f"  Forest Types: {forest_types}")
        print(f"  Climate Scenarios: {climates}")
        print(f"  Management Levels: {managements}")
        print(f"  Years: {years}")
        print(f"  Workers: {workers}")
        print(f"  Generate Plots: {generate_plots}")
        
        # Calculate total scenarios
        total_scenarios = len(forest_types) * len(climates) * len(managements)
        print(f"  Total Scenarios: {total_scenarios}")
        
        # Step 1: Generate scenarios
        print(f"\n{'='*60}")
        print("Step 1: Generating Scenario Configurations")
        print(f"{'='*60}")
        
        scenarios = self.generator.generate_matrix(
            forest_types=forest_types,
            climates=climates,
            managements=managements,
            time_periods=[f"2025-{2025+years}"]  # Convert years to time period
        )
        
        print(f"Generated {len(scenarios)} scenario configurations")
        
        # Save scenarios
        manifest = self.generator.save_all_scenarios(scenarios)
        print(f"Saved manifest to: configs/generated/scenario_manifest.csv")
        
        # Show sample scenarios
        print(f"\nSample scenarios:")
        for i, row in manifest.head(5).iterrows():
            print(f"  {row['name']} ({row['years']} years)")
        
        if len(scenarios) > 5:
            print(f"  ... and {len(scenarios) - 5} more")
        
        # Step 2: Run simulations
        print(f"\n{'='*60}")
        print("Step 2: Running Simulations")
        print(f"{'='*60}")
        
        # Check if any scenarios were generated
        if manifest.empty:
            print("No valid scenarios to run. Please check your configuration parameters.")
            return {
                'successful': 0,
                'failed': 0,
                'results': pd.DataFrame(),
                'metadata': {
                    'forest_types': forest_types,
                    'climates': climates,
                    'managements': managements,
                    'years': years,
                    'workers': workers
                }
            }
        
        runner = BatchRunner(n_workers=workers)
        
        config_paths = [Path(f) for f in manifest['file']]
        
        # Prepare simulation parameters
        sim_params = {
            'area_ha': 1000.0,
            'generate_plots': generate_plots,
            'enable_uncertainty': enable_uncertainty,
            'seed': seed
        }
        
        # Run batch simulations
        results = runner.run_batch(config_paths, **sim_params)
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        results_path = output_path / "batch_results.csv"
        results.to_csv(results_path, index=False)
        print(f"Results saved to: {results_path}")
        
        # Show summary
        successful = results[results['status'] == 'success']
        failed = results[results['status'] == 'failed']
        
        print(f"\nSimulation Summary:")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        
        if len(failed) > 0:
            print(f"\nFailed scenarios:")
            for _, row in failed.iterrows():
                print(f"  {row['scenario']}: {row.get('error', 'Unknown error')}")
        
        # Step 3: Analyze results
        if len(successful) > 0:
            print(f"\n{'='*60}")
            print("Step 3: Analyzing Results")
            print(f"{'='*60}")
            
            # Update analyzer with results path
            analyzer = ScenarioAnalyzer(results_path=results_path)
            analyzer.generate_report(output_dir=Path(f"{output_dir}/analysis"))
            
            # Show key findings
            print(f"\nKey Findings:")
            
            # Best scenario by additionality
            best_additionality = successful.loc[successful['management_additionality'].idxmax()]
            print(f"  Best Carbon Additionality: {best_additionality['scenario']} ({best_additionality['management_additionality']:.1f} t CO2e/ha)")
            
            # Best scenario by NPV
            best_npv = successful.loc[successful['management_npv'].idxmax()]
            print(f"  Best NPV: {best_npv['scenario']} (${best_npv['management_npv']:.0f}/ha)")
            
            # Forest type performance
            forest_performance = successful.groupby('forest_type')['management_additionality'].mean()
            best_forest = forest_performance.idxmax()
            print(f"  Best Forest Type: {best_forest} ({forest_performance[best_forest]:.1f} t CO2e/ha avg)")
            
            # Management performance
            mgmt_performance = successful.groupby('management')['management_additionality'].mean()
            best_mgmt = mgmt_performance.idxmax()
            print(f"  Best Management: {best_mgmt} ({mgmt_performance[best_mgmt]:.1f} t CO2e/ha avg)")
            
            # Climate impact
            climate_performance = successful.groupby('climate')['management_additionality'].mean()
            best_climate = climate_performance.idxmax()
            worst_climate = climate_performance.idxmin()
            print(f"  Climate Impact: {best_climate} best ({climate_performance[best_climate]:.1f} t CO2e/ha), {worst_climate} worst ({climate_performance[worst_climate]:.1f} t CO2e/ha)")
        
        print(f"\n{'='*60}")
        print("✓ Scenario Analysis Complete!")
        print(f"{'='*60}")
        print(f"Results available in: {output_dir}/")
        print(f"Analysis report: {output_dir}/analysis/scenario_analysis_report.md")
        print(f"Visualizations: {output_dir}/analysis/comprehensive_analysis.png")
        
        return {
            'scenarios_generated': len(scenarios),
            'simulations_successful': len(successful),
            'simulations_failed': len(failed),
            'results_path': str(results_path),
            'analysis_path': f"{output_dir}/analysis/"
        }

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive forest carbon scenario analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all default scenarios
    python -m forest_carbon.scenarios.manager
    
    # Run specific forest types and climates
    python -m forest_carbon.scenarios.manager --forest-types ETOF,AFW --climates current,paris_target
    
    # Run with custom years and generate plots
    python -m forest_carbon.scenarios.manager --years 30 --plots
    
    # Run with more workers
    python -m forest_carbon.scenarios.manager --workers 8
        """
    )
    
    parser.add_argument(
        '--site',
        type=str,
        default='ETOF,EOF,AFW',
        help='Comma-separated site/forest types (default: ETOF,EOF,AFW)'
    )
    
    parser.add_argument(
        '--climates',
        type=str,
        default='current,paris_target,paris_overshoot',
        help='Comma-separated climate scenarios (default: current,paris_target,paris_overshoot)'
    )
    
    parser.add_argument(
        '--managements',
        type=str,
        default='baseline,adaptive,intensive',
        help='Comma-separated management levels (default: baseline,adaptive,intensive)'
    )
    
    parser.add_argument(
        '--years',
        type=int,
        default=25,
        help='Simulation duration in years (default: 25)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Generate individual plots for each scenario'
    )
    
    parser.add_argument(
        '--area',
        type=float,
        default=1000.0,
        help='Project area in hectares (default: 1000)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory (default: output)'
    )
    
    return parser.parse_args()

def parse_string_list(available_values: List[str], value_str: str, category_name: str) -> List[str]:
    """Parse comma-separated string into validated string list."""
    values = [v.strip() for v in value_str.split(',')]
    valid_values = []
    
    for value in values:
        if value in available_values:
            valid_values.append(value)
        else:
            print(f"Warning: Invalid {category_name} value: {value}")
            print(f"Available {category_name} options: {', '.join(available_values)}")
    
    return valid_values

def main():
    """Main function to run scenario analysis."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Get available options from dynamic discovery
    available_forests = ForestType.get_available_types()
    available_climates = ClimateScenario.get_available_scenarios()
    available_managements = ManagementLevel.get_available_levels()
    
    # Parse string lists with validation
    forest_types = parse_string_list(available_forests, args.forest_types, "forest type")
    climates = parse_string_list(available_climates, args.climates, "climate scenario")
    managements = parse_string_list(available_managements, args.managements, "management level")
    
    if not forest_types or not climates or not managements:
        print("Error: No valid configurations found!")
        print("Available options:")
        print(f"  Forest Types: {available_forests}")
        print(f"  Climate Scenarios: {available_climates}")
        print(f"  Management Levels: {available_managements}")
        return
    
    # Initialize scenario manager
    manager = ScenarioManager()
    
    # Run analysis
    results = manager.run_analysis(
        forest_types=forest_types,
        climates=climates,
        managements=managements,
        years=args.years,
        workers=args.workers,
        generate_plots=args.plots,
        output_dir=args.output_dir
    )
    
    return results

if __name__ == "__main__":
    main()
