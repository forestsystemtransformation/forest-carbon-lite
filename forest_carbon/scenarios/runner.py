#!/usr/bin/env python3
"""
Batch Runner for Forest Carbon Scenarios

Runs multiple scenarios in parallel and compiles results.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import pandas as pd
import logging
from pathlib import Path
import yaml
import sys
from tqdm import tqdm

class BatchRunner:
    """Run multiple scenarios in parallel."""
    
    def __init__(self, n_workers: int = None):
        """
        Initialize batch runner.
        
        Args:
            n_workers: Number of parallel workers (default: CPU count - 1)
        """
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def run_scenario(self, config_path: Path, 
                    years: int = None, 
                    area_ha: float = 1000,
                    generate_plots: bool = False,
                    enable_uncertainty: bool = False,
                    seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a single scenario.
        
        Args:
            config_path: Path to scenario configuration file
            years: Simulation duration in years
            area_ha: Area in hectares
            generate_plots: Whether to generate plots
            enable_uncertainty: Whether to enable uncertainty analysis
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with results and metadata
        """
        try:
            # Import here to avoid pickling issues
            from ..core.simulator import ForestCarbonSimulator
            
            # Load config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract metadata
            metadata = config.get('scenario_metadata', {})
            scenario_name = metadata.get('name', config_path.stem)
            
            # Use years from config if not specified
            if years is None:
                years = metadata.get('simulation_years', 25)
            
            # Create new output directory structure: output/{scenario_name}/{years} years/
            output_dir = Path(f"output/{scenario_name}/{years} years")
            
            # Run simulation
            simulator = ForestCarbonSimulator(
                forest_type=metadata.get('forest_type', 'ETOF'),
                years=years,
                area_ha=area_ha,
                output_dir=str(output_dir),
                config_file=str(config_path.absolute()),
                enable_uncertainty=enable_uncertainty,
                seed=seed
            )
            
            results = simulator.run(generate_plots=generate_plots)
            
            # Extract key metrics
            summary = results.summary
            
            # Get final results for each scenario
            baseline_result = summary[summary['scenario'] == 'baseline'].iloc[0] if len(summary[summary['scenario'] == 'baseline']) > 0 else None
            management_result = summary[summary['scenario'] == 'management'].iloc[0] if len(summary[summary['scenario'] == 'management']) > 0 else None
            reforestation_result = summary[summary['scenario'] == 'reforestation'].iloc[0] if len(summary[summary['scenario'] == 'reforestation']) > 0 else None
            
            result_summary = {
                'scenario': scenario_name,
                'forest_type': metadata.get('forest_type'),
                'climate': metadata.get('climate_scenario'),
                'management': metadata.get('management_level'),
                'time_period': metadata.get('time_period'),
                'years': years,
                'area_ha': area_ha,
                'status': 'success',
                'output_dir': str(output_dir)
            }
            
            # Add baseline results
            if baseline_result is not None:
                result_summary.update({
                    'baseline_final_co2e': baseline_result.get('final_co2e_stock', 0),
                    'baseline_final_agb': baseline_result.get('total_agb', 0),
                    'baseline_disturbances': baseline_result.get('disturbance_events', 0)
                })
            
            # Add management results
            if management_result is not None:
                result_summary.update({
                    'management_final_co2e': management_result.get('final_co2e_stock', 0),
                    'management_final_agb': management_result.get('total_agb', 0),
                    'management_disturbances': management_result.get('disturbance_events', 0),
                    'management_npv': management_result.get('npv', 0),
                    'management_irr': management_result.get('irr', 0),
                    'management_credits': management_result.get('total_credits', 0)
                })
            
            # Add reforestation results
            if reforestation_result is not None:
                result_summary.update({
                    'reforestation_final_co2e': reforestation_result.get('final_co2e_stock', 0),
                    'reforestation_final_agb': reforestation_result.get('total_agb', 0),
                    'reforestation_disturbances': reforestation_result.get('disturbance_events', 0),
                    'reforestation_npv': reforestation_result.get('npv', 0),
                    'reforestation_irr': reforestation_result.get('irr', 0),
                    'reforestation_credits': reforestation_result.get('total_credits', 0)
                })
            
            # Calculate additionality
            if baseline_result is not None and management_result is not None:
                result_summary['management_additionality'] = (
                    management_result.get('final_co2e_stock', 0) - 
                    baseline_result.get('final_co2e_stock', 0)
                )
            
            if reforestation_result is not None:
                # Reforestation additionality = Reforestation - 0 (bare ground)
                result_summary['reforestation_additionality'] = (
                    reforestation_result.get('final_co2e_stock', 0) - 0
                )
            
            return result_summary
            
        except Exception as e:
            self.logger.error(f"Error in scenario {config_path}: {str(e)}")
            return {
                'scenario': config_path.stem,
                'status': 'failed',
                'error': str(e)
            }
    
    def run_batch(self, config_paths: List[Path], **sim_params) -> pd.DataFrame:
        """
        Run multiple scenarios in parallel.
        
        Args:
            config_paths: List of configuration file paths
            **sim_params: Additional simulation parameters
            
        Returns:
            DataFrame with results from all scenarios
        """
        results = []
        
        print(f"Running {len(config_paths)} scenarios with {self.n_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all jobs
            futures = {
                executor.submit(self.run_scenario, path, **sim_params): path
                for path in config_paths
            }
            
            # Process results as they complete
            with tqdm(total=len(config_paths), desc="Running scenarios") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    
                    if result['status'] == 'success':
                        self.logger.info(f"✓ Completed: {result['scenario']}")
                    else:
                        self.logger.error(f"✗ Failed: {result['scenario']}")
        
        return pd.DataFrame(results)
    
    def run_from_manifest(self, manifest_path: Path, **sim_params) -> pd.DataFrame:
        """
        Run scenarios from a manifest file.
        
        Args:
            manifest_path: Path to scenario manifest CSV
            **sim_params: Additional simulation parameters
            
        Returns:
            DataFrame with results from all scenarios
        """
        manifest_df = pd.read_csv(manifest_path)
        config_paths = [Path(f) for f in manifest_df['file']]
        
        return self.run_batch(config_paths, **sim_params)

def main():
    """Example usage of batch runner."""
    
    print("🌲 Forest Carbon Batch Runner")
    print("=" * 50)
    
    # Initialize batch runner
    runner = BatchRunner(n_workers=4)
    
    # Run scenarios from manifest
    manifest_path = Path("configs/generated/scenario_manifest.csv")
    
    if manifest_path.exists():
        print(f"Running scenarios from manifest: {manifest_path}")
        
        results = runner.run_from_manifest(
            manifest_path,
            years=25,
            area_ha=1000,
            generate_plots=False  # Set to True if you want individual plots
        )
        
        # Save results
        results_path = Path("output/batch_results.csv")
        results.to_csv(results_path, index=False)
        print(f"Results saved to: {results_path}")
        
        # Show summary
        successful = results[results['status'] == 'success']
        failed = results[results['status'] == 'failed']
        
        print(f"\nSummary:")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        
        if len(successful) > 0:
            print(f"\nTop performing scenarios by management additionality:")
            top_scenarios = successful.nlargest(5, 'management_additionality')
            for _, row in top_scenarios.iterrows():
                print(f"  {row['scenario']}: {row['management_additionality']:.1f} t CO2e/ha")
    
    else:
        print(f"Manifest not found: {manifest_path}")
        print("Run scenario_builder.py first to generate scenarios")

if __name__ == "__main__":
    main()