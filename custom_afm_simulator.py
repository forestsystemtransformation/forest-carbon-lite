#!/usr/bin/env python3
"""
Custom AFM vs Degrading Simulator
Runs only baseline and management scenarios (no reforestation)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from forest_carbon.core.simulator import ForestCarbonSimulator
import pandas as pd
from typing import Dict, Tuple, List

class AFMOnlySimulator(ForestCarbonSimulator):
    """Custom simulator that runs only AFM vs Degrading scenarios."""
    
    def run_afm_scenarios(self) -> Dict[str, Tuple[pd.DataFrame, List[Dict]]]:
        """Run only baseline (degrading) and management (AFM) scenarios."""
        scenarios = ['baseline', 'management']  # No reforestation
        results = {}
        
        for scenario in scenarios:
            print(f"Running {scenario} scenario...")
            df, pools = self.simulate_scenario(scenario)
            
            # Validate results are not empty
            assert not df.empty, f"Simulation results are empty for scenario {scenario}"
            assert len(pools) > 0, f"Carbon pools history is empty for scenario {scenario}"
            assert len(df) == self.years, f"Expected {self.years} years of data, got {len(df)} for scenario {scenario}"
            
            results[scenario] = (df, pools)
        
        # Validate scenarios completed successfully
        assert len(results) == 2, f"Expected 2 scenarios, got {len(results)}"
        assert all(scenario in results for scenario in scenarios), "Missing scenario results"
        
        return results
    
    def run_afm_analysis(self, generate_plots: bool = True) -> Dict:
        """Run AFM vs Degrading analysis with custom plotting."""
        print("🌲 AFM vs Degrading Forest Analysis")
        print("=" * 50)
        
        # Run only the two scenarios we care about
        scenario_results = self.run_afm_scenarios()
        
        # Compile results (similar to original but only 2 scenarios)
        results = self._compile_afm_results(scenario_results)
        
        # Save results
        self._save_afm_results(results)
        
        # Generate plots if requested
        if generate_plots:
            self._generate_afm_plots(scenario_results)
        
        return results
    
    def _compile_afm_results(self, scenario_results: Dict) -> Dict:
        """Compile results for AFM analysis."""
        # Create summary DataFrame with only baseline and management
        summary_data = []
        
        for scenario, (df, pools) in scenario_results.items():
            final_row = df.iloc[-1]
            
            # Calculate economic metrics only for management scenario
            if scenario == 'management':
                # Use the economics model to calculate NPV, IRR, etc.
                abatement_series = df['total_co2e'] - scenario_results['baseline'][0]['total_co2e']
                cashflow = self.economics_model.calculate_cashflow(
                    abatement_series, scenario, self.area_ha
                )
                npv = self.economics_model.calculate_npv(cashflow)
                irr = self.economics_model.calculate_irr(cashflow)
                payback = self.economics_model.calculate_payback_period(cashflow)
                summary = self.economics_model.generate_summary(cashflow)
            else:
                # Baseline has no economic benefits
                npv = irr = payback = None
                summary = {}
            
            summary_data.append({
                'scenario': scenario,
                'forest_type': self.forest_type,
                'final_co2e_stock': final_row['total_co2e'],
                'mean_annual_increment': final_row['total_co2e'] / self.years,
                'total_agb': final_row['total_agb'],
                'disturbance_events': df['disturbance'].sum(),
                'npv': npv,
                'irr': irr,
                'payback_period': payback,
                'total_revenue': summary.get('revenue', None),
                'total_costs': summary.get('total_costs', None),
                'total_credits': summary.get('credits_tCO2e', None),
                'avg_carbon_price': summary.get('carbon_price', None)
            })
        
        return {
            'summary': pd.DataFrame(summary_data),
            'scenario_results': scenario_results,
            'carbon_pools': {scenario: pools for scenario, (_, pools) in scenario_results.items()}
        }
    
    def _save_afm_results(self, results: Dict):
        """Save AFM analysis results."""
        # Save summary CSV
        summary_path = self.output_dir / 'afm_analysis_summary.csv'
        results['summary'].to_csv(summary_path, index=False)
        print(f"[SUCCESS] AFM analysis summary saved to: {summary_path}")
        
        # Save detailed results
        for scenario, (df, pools) in results['scenario_results'].items():
            scenario_dir = self.output_dir / f"{scenario}_results"
            scenario_dir.mkdir(exist_ok=True)
            
            # Save time series
            df.to_csv(scenario_dir / f"{scenario}_time_series.csv", index=False)
            
            # Save carbon pools
            pools_df = pd.DataFrame(pools)
            pools_df.to_csv(scenario_dir / f"{scenario}_carbon_pools.csv", index=False)
    
    def _generate_afm_plots(self, scenario_results: Dict):
        """Generate plots for AFM analysis."""
        from forest_carbon.visualization.plotter import Plotter
        
        plotter = Plotter(self.output_dir)
        
        # Prepare data in the exact format expected by the plotter
        # This should match the format from regular ETOF current intensive results
        baseline_df = scenario_results['baseline'][0].copy()
        management_df = scenario_results['management'][0].copy()
        
        # Create the combined DataFrame in calendar years format (with 'scenario' column)
        # This matches the format that the plotter expects for the first branch
        baseline_plot_df = baseline_df[['year', 'total_co2e']].copy()
        baseline_plot_df['scenario'] = 'baseline'
        baseline_plot_df['calendar_year'] = baseline_plot_df['year']
        
        management_plot_df = management_df[['year', 'total_co2e']].copy()
        management_plot_df['scenario'] = 'management'
        management_plot_df['calendar_year'] = management_plot_df['year']
        
        # Combine the dataframes
        combined_df = pd.concat([baseline_plot_df, management_plot_df], ignore_index=True)
        
        plot_results = {self.forest_type: combined_df}
        
        # Generate the total carbon stocks plot (this is the main one you want)
        plotter.plot_total_carbon_stocks_all_scenarios(plot_results, [self.forest_type])
        
        # Generate the custom additionality plot (AFM vs baseline difference)
        self._plot_afm_additionality(plotter, plot_results, [self.forest_type])
        
        print(f"[SUCCESS] AFM analysis plots saved to: {self.output_dir}/plots/")
    
    def _plot_afm_additionality(self, plotter, results: Dict, forest_types: List[str]):
        """Plot AFM additionality (custom version with renamed title)."""
        import matplotlib.pyplot as plt
        from forest_carbon.utils.colors import color_manager
        
        fig, ax = plt.subplots(figsize=(plotter.fig_width, plotter.fig_height))
        
        for forest_type in forest_types:
            df = results[forest_type]
            
            # Check data format and calculate additionality accordingly
            if 'scenario' in df.columns:
                # Calendar years format - calculate additionality by scenario
                baseline_data = df[df['scenario'] == 'baseline']
                management_data = df[df['scenario'] == 'management']
                
                if not baseline_data.empty and not management_data.empty:
                    # Calculate additionality for management (vs baseline)
                    # Ensure data is aligned by year
                    baseline_sorted = baseline_data.sort_values('year')
                    management_sorted = management_data.sort_values('year')
                    management_additionality = management_sorted['total_co2e'].values - baseline_sorted['total_co2e'].values
                    mgmt_style = color_manager.get_scenario_style('management')
                    ax.plot(management_sorted['calendar_year'], management_additionality, 
                           label=f'{forest_type} - AFM vs Degrading Forest',
                           **mgmt_style)
        
        # Customize the plot
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Carbon additionality (t CO2e/ha)', fontsize=12, fontweight='bold')
        ax.set_title('AFM Carbon Additionality', fontsize=14, fontweight='bold')
        
        # Add legend with better positioning
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fancybox=True, shadow=True)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add zero line for reference
        ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.6)
        
        # Add some styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
        plt.tight_layout()
        
        # Save with custom filename
        plotter._save_figure('additionality', formats=['png', 'svg'])

def main():
    """Command-line interface for AFM-only simulator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AFM vs Degrading Forest Analysis')
    parser.add_argument('--forest-type', default='ETOF', help='Forest type (default: ETOF)')
    parser.add_argument('--climate', default='current', help='Climate scenario (default: current)')
    parser.add_argument('--management', default='intensive', help='Management level (default: intensive)')
    parser.add_argument('--years', type=int, default=52, help='Simulation years (default: 52)')
    parser.add_argument('--area', type=float, default=1000.0, help='Area in hectares (default: 1000)')
    parser.add_argument('--output-dir', default=None, help='Output directory (auto-generated if not specified)')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    
    args = parser.parse_args()
    
    # Generate output directory name if not specified
    if args.output_dir is None:
        args.output_dir = f'output/{args.forest_type}_{args.climate}_{args.management}_afm_only'
    
    print(f"🌲 AFM vs Degrading Analysis")
    print(f"Forest Type: {args.forest_type}")
    print(f"Climate: {args.climate}")
    print(f"Management: {args.management}")
    print(f"Years: {args.years}")
    print(f"Area: {args.area} ha")
    print(f"Output: {args.output_dir}")
    print("="*50)
    
    # Initialize simulator with the proper intensive management configuration
    # Use the generated ETOF current intensive config which has the right TYF calibrations
    config_file = f"configs/generated/{args.forest_type}_{args.climate}_{args.management}.yaml"
    
    # Check if the config file exists
    if not Path(config_file).exists():
        print(f"Warning: Config file {config_file} not found, using default configuration")
        simulator = AFMOnlySimulator(
            forest_type=args.forest_type,
            years=args.years,
            area_ha=args.area,
            output_dir=args.output_dir
        )
    else:
        simulator = AFMOnlySimulator(
            forest_type=args.forest_type,
            years=args.years,
            area_ha=args.area,
            output_dir=args.output_dir,
            config_file=config_file
        )
        print(f"Using configuration: {config_file}")
    
    # Run AFM analysis
    results = simulator.run_afm_analysis(generate_plots=not args.no_plots)
    
    # Print summary
    print("\n" + "="*50)
    print("AFM vs Degrading Analysis Results")
    print("="*50)
    print(results['summary'].to_string(index=False))
    
    # Calculate additionality
    baseline_final = results['summary'][results['summary']['scenario'] == 'baseline']['final_co2e_stock'].iloc[0]
    management_final = results['summary'][results['summary']['scenario'] == 'management']['final_co2e_stock'].iloc[0]
    additionality = management_final - baseline_final
    
    print(f"\nKey Results:")
    print(f"  Degrading Baseline: {baseline_final:.1f} t CO2e/ha")
    print(f"  AFM Management: {management_final:.1f} t CO2e/ha")
    print(f"  Carbon Additionality: {additionality:.1f} t CO2e/ha")
    
    if results['summary']['npv'].iloc[1] is not None:  # Management scenario NPV
        print(f"  AFM NPV: ${results['summary']['npv'].iloc[1]:,.0f}/ha")
    
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
