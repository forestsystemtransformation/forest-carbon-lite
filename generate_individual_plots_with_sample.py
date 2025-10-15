#!/usr/bin/env python3
"""
Generate Individual Plots with Sample Data

This script creates individual plots for each graph in the comprehensive analysis matrix,
using sample data if no results are available.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class IndividualPlotGeneratorWithSample:
    """Generate individual plots from comprehensive analysis results or sample data."""
    
    def __init__(self, results_path: Path = Path("output/batch_results.csv")):
        """Initialize plot generator."""
        self.results_path = results_path
        self.results = None
        self.load_results()
    
    def load_results(self):
        """Load results from CSV or create sample data."""
        if self.results_path.exists():
            self.results = pd.read_csv(self.results_path)
            print(f"📊 Loaded {len(self.results)} scenario results")
        else:
            print(f"⚠️  Results file not found: {self.results_path}")
            print("🎲 Generating sample data for demonstration...")
            self.results = self._create_sample_data()
            print(f"📊 Created {len(self.results)} sample scenario results")
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for demonstration purposes."""
        np.random.seed(42)  # For reproducible results
        
        # Define scenario parameters
        forest_types = ['ETOF', 'EOF', 'AFW', 'EOFD', 'ETOFD']
        climates = ['current', 'paris', 'plus2', 'plus3']
        managements = ['m', 'i', 'ir', 'l', 'lr', 'mr', 'afm_m']
        
        # Generate all combinations
        scenarios = []
        for forest_type in forest_types:
            for climate in climates:
                for management in managements:
                    scenario = {
                        'scenario': f"{forest_type}_{climate}_{management}",
                        'forest_type': forest_type,
                        'climate': climate,
                        'management': management,
                        'time_period': '2025-2052',
                        'years': 27.0,
                        'area_ha': 1000.0,
                        'status': 'success'
                    }
                    scenarios.append(scenario)
        
        # Create DataFrame
        data = pd.DataFrame(scenarios)
        
        # Generate realistic sample data based on forest type and management
        np.random.seed(42)
        
        # Base values by forest type
        forest_base = {
            'ETOF': {'co2e': 800, 'npv': 60000, 'additionality': 400},
            'EOF': {'co2e': 600, 'npv': 45000, 'additionality': 300},
            'AFW': {'co2e': 400, 'npv': 30000, 'additionality': 200},
            'EOFD': {'co2e': 500, 'npv': 35000, 'additionality': 250},
            'ETOFD': {'co2e': 700, 'npv': 50000, 'additionality': 350}
        }
        
        # Climate multipliers
        climate_mult = {
            'current': 1.0,
            'paris_target': 1.1,
            'paris_overshoot': 0.9
        }
        
        # Management multipliers
        mgmt_mult = {
            'm': 1.0,
            'i': 1.3,
            'ir': 1.5,
            'l': 0.8,
            'lr': 1.0,
            'mr': 1.2,
            'afm_m': 1.1
        }
        
        # Generate data
        for idx, row in data.iterrows():
            base = forest_base[row['forest_type']]
            climate_factor = climate_mult[row['climate']]
            mgmt_factor = mgmt_mult[row['management']]
            
            # Add some randomness
            noise = np.random.normal(1.0, 0.1)
            
            # Calculate values
            baseline_co2e = base['co2e'] * climate_factor * noise
            management_co2e = baseline_co2e + (base['additionality'] * mgmt_factor * climate_factor * noise)
            reforestation_co2e = baseline_co2e + (base['additionality'] * 0.8 * climate_factor * noise)
            
            data.loc[idx, 'baseline_final_co2e'] = baseline_co2e
            data.loc[idx, 'baseline_final_agb'] = baseline_co2e * 0.2
            data.loc[idx, 'baseline_disturbances'] = np.random.poisson(3)
            
            data.loc[idx, 'management_final_co2e'] = management_co2e
            data.loc[idx, 'management_final_agb'] = management_co2e * 0.2
            data.loc[idx, 'management_disturbances'] = np.random.poisson(2)
            data.loc[idx, 'management_npv'] = base['npv'] * mgmt_factor * climate_factor * noise
            data.loc[idx, 'management_irr'] = np.random.uniform(15, 50)
            data.loc[idx, 'management_credits'] = (management_co2e - baseline_co2e) * 1000
            
            data.loc[idx, 'reforestation_final_co2e'] = reforestation_co2e
            data.loc[idx, 'reforestation_final_agb'] = reforestation_co2e * 0.2
            data.loc[idx, 'reforestation_disturbances'] = np.random.poisson(2)
            data.loc[idx, 'reforestation_npv'] = base['npv'] * 0.6 * climate_factor * noise
            data.loc[idx, 'reforestation_irr'] = np.random.uniform(10, 30)
            data.loc[idx, 'reforestation_credits'] = (reforestation_co2e - baseline_co2e) * 1000
            
            data.loc[idx, 'management_additionality'] = management_co2e - baseline_co2e
            data.loc[idx, 'reforestation_additionality'] = reforestation_co2e - 0  # Reforestation vs bare ground
        
        return data
    
    def generate_all_individual_plots(self, output_dir: Path = Path("output/individual_plots")):
        """Generate all individual plots."""
        if self.results.empty:
            print("❌ No data available for plotting")
            return
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎨 Generating individual plots in: {output_dir}")
        print("=" * 60)
        
        # Generate all individual plots
        plots = [
            ("01_climate_impact", "Climate Impact on Carbon Stocks", self._plot_climate_impact),
            ("02_management_effectiveness", "Management Effectiveness", self._plot_management_effectiveness),
            ("03_forest_type_comparison", "Forest Type Performance Comparison", self._plot_forest_type_comparison),
            ("04_economic_heatmap", "Economic Performance Heatmap", self._plot_economic_heatmap),
            ("05_additionality_analysis", "Additionality Analysis", self._plot_additionality_analysis),
            ("06_disturbance_resilience", "Disturbance Resilience", self._plot_disturbance_resilience),
            ("07_scenario_ranking", "Top Scenario Rankings", self._plot_scenario_ranking),
            ("08_cost_effectiveness", "Cost Effectiveness Analysis", self._plot_cost_effectiveness),
            ("09_npv_distribution", "NPV Distribution", self._plot_npv_distribution),
            ("10_carbon_evolution", "Carbon Stock Evolution", self._plot_carbon_evolution),
            ("11_management_climate_interaction", "Management-Climate Interaction", self._plot_management_climate_interaction),
            ("12_summary_statistics", "Summary Statistics", self._plot_summary_statistics)
        ]
        
        for plot_id, plot_name, plot_func in plots:
            try:
                output_path = output_dir / f"{plot_id}_{plot_name.lower().replace(' ', '_')}.png"
                print(f"📈 Generating: {plot_name}")
                plot_func(self.results, output_path)
                print(f"   ✅ Saved: {output_path}")
            except Exception as e:
                print(f"   ❌ Error generating {plot_name}: {e}")
        
        print(f"\n🎉 Individual plots generation complete!")
        print(f"📁 All plots saved in: {output_dir}")
    
    def _plot_climate_impact(self, data: pd.DataFrame, output_path: Path):
        """Plot climate impact on carbon stocks."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        pivot = data.pivot_table(
            values='management_final_co2e',
            index='climate',
            columns='forest_type',
            aggfunc='mean'
        )
        
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Climate Impact on Final Carbon Stock', fontsize=16, fontweight='bold')
        ax.set_ylabel('CO2e (t/ha)', fontsize=12)
        ax.set_xlabel('Climate Scenario', fontsize=12)
        ax.legend(title='Forest Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_management_effectiveness(self, data: pd.DataFrame, output_path: Path):
        """Plot management effectiveness."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        data['abatement'] = data['management_final_co2e'] - data['baseline_final_co2e']
        
        pivot = data.pivot_table(
            values='abatement',
            index='management',
            columns='forest_type',
            aggfunc='mean'
        )
        
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Management Effectiveness (Carbon Abatement)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Additional CO2e (t/ha)', fontsize=12)
        ax.set_xlabel('Management Level', fontsize=12)
        ax.legend(title='Forest Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_forest_type_comparison(self, data: pd.DataFrame, output_path: Path):
        """Plot forest type comparison."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        forest_summary = data.groupby('forest_type').agg({
            'management_final_co2e': 'mean',
            'management_npv': 'mean',
            'management_additionality': 'mean'
        }).reset_index()
        
        x = range(len(forest_summary))
        width = 0.25
        
        ax.bar([i - width for i in x], forest_summary['management_final_co2e'], 
               width, label='Final CO2e', alpha=0.8)
        ax.bar(x, forest_summary['management_npv'] / 1000, 
               width, label='NPV (k$)', alpha=0.8)
        ax.bar([i + width for i in x], forest_summary['management_additionality'], 
               width, label='Additionality', alpha=0.8)
        
        ax.set_title('Forest Type Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12)
        ax.set_xlabel('Forest Type', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(forest_summary['forest_type'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_economic_heatmap(self, data: pd.DataFrame, output_path: Path):
        """Plot economic performance heatmap."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        pivot = data.pivot_table(
            values='management_npv',
            index='management',
            columns='climate',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot, annot=True, fmt='.0f', ax=ax, cmap='RdYlGn', center=0)
        ax.set_title('NPV by Management and Climate', fontsize=16, fontweight='bold')
        ax.set_xlabel('Climate Scenario', fontsize=12)
        ax.set_ylabel('Management Level', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_additionality_analysis(self, data: pd.DataFrame, output_path: Path):
        """Plot additionality analysis."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        data['abatement'] = data['management_final_co2e'] - data['baseline_final_co2e']
        
        # Create scatter plot
        scatter = ax.scatter(data['baseline_final_co2e'], data['abatement'], 
                           c=data['management_npv'], cmap='viridis', alpha=0.7, s=100)
        
        ax.set_xlabel('Baseline Carbon Stock (t CO2e/ha)', fontsize=12)
        ax.set_ylabel('Management Additionality (t CO2e/ha)', fontsize=12)
        ax.set_title('Additionality vs Baseline Stock', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('NPV ($/ha)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_disturbance_resilience(self, data: pd.DataFrame, output_path: Path):
        """Plot disturbance resilience."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        pivot = data.pivot_table(
            values='management_disturbances',
            index='climate',
            columns='management',
            aggfunc='mean'
        )
        
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Disturbance Events by Climate and Management', fontsize=16, fontweight='bold')
        ax.set_ylabel('Number of Disturbances', fontsize=12)
        ax.set_xlabel('Climate Scenario', fontsize=12)
        ax.legend(title='Management Level', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_scenario_ranking(self, data: pd.DataFrame, output_path: Path):
        """Plot top scenarios ranking."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate combined score
        data['combined_score'] = (
            (data['management_additionality'] / data['management_additionality'].max()) * 0.6 +
            (data['management_npv'] / data['management_npv'].max()) * 0.4
        )
        
        top_scenarios = data.nlargest(10, 'combined_score')
        
        bars = ax.barh(range(len(top_scenarios)), top_scenarios['combined_score'])
        ax.set_yticks(range(len(top_scenarios)))
        ax.set_yticklabels([name.replace('_', '\n') for name in top_scenarios['scenario']])
        ax.set_xlabel('Combined Score', fontsize=12)
        ax.set_title('Top 10 Scenarios by Combined Score', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Color bars by forest type
        colors = {'ETOF': '#1f77b4', 'EOF': '#ff7f0e', 'AFW': '#2ca02c', 'EOFD': '#d62728', 'ETOFD': '#9467bd'}
        for i, (_, row) in enumerate(top_scenarios.iterrows()):
            bars[i].set_color(colors.get(row['forest_type'], '#888888'))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cost_effectiveness(self, data: pd.DataFrame, output_path: Path):
        """Plot cost effectiveness analysis."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Assume intervention costs based on management level (using actual values from data)
        cost_mapping = {
            'm': 200,      # moderate
            'i': 500,      # intensive
            'ir': 1000,    # intensive reforestation
            'l': 100,      # light
            'lr': 300,     # light reforestation
            'mr': 800,     # moderate reforestation
            'afm_m': 400   # adaptive forest management moderate
        }
        
        data['intervention_cost'] = data['management'].map(cost_mapping).fillna(0)
        data['cost_effectiveness'] = data['management_additionality'] / (data['intervention_cost'] + 1)
        
        pivot = data.pivot_table(
            values='cost_effectiveness',
            index='management',
            columns='forest_type',
            aggfunc='mean'
        )
        
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Cost Effectiveness (t CO2e per $)', fontsize=16, fontweight='bold')
        ax.set_ylabel('CO2e per Dollar', fontsize=12)
        ax.set_xlabel('Management Level', fontsize=12)
        ax.legend(title='Forest Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_npv_distribution(self, data: pd.DataFrame, output_path: Path):
        """Plot NPV distribution."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create box plot of NPV by management level
        sns.boxplot(data=data, x='management', y='management_npv', ax=ax)
        ax.set_title('NPV Distribution by Management Level', fontsize=16, fontweight='bold')
        ax.set_ylabel('NPV ($/ha)', fontsize=12)
        ax.set_xlabel('Management Level', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_carbon_evolution(self, data: pd.DataFrame, output_path: Path):
        """Plot carbon stock evolution."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot baseline vs management final carbon stocks
        scatter = ax.scatter(data['baseline_final_co2e'], data['management_final_co2e'], 
                  alpha=0.7, s=100, c=data['management_npv'], cmap='viridis')
        
        # Add diagonal line (no change)
        min_val = min(data['baseline_final_co2e'].min(), data['management_final_co2e'].min())
        max_val = max(data['baseline_final_co2e'].max(), data['management_final_co2e'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='No Change')
        
        ax.set_xlabel('Baseline Final CO2e (t/ha)', fontsize=12)
        ax.set_ylabel('Management Final CO2e (t/ha)', fontsize=12)
        ax.set_title('Carbon Stock Evolution: Baseline vs Management', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        scatter = ax.collections[0]
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('NPV ($/ha)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_management_climate_interaction(self, data: pd.DataFrame, output_path: Path):
        """Plot management vs climate interaction."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create interaction plot
        pivot = data.pivot_table(
            values='management_additionality',
            index='management',
            columns='climate',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot, annot=True, fmt='.1f', ax=ax, cmap='RdYlGn', center=0)
        ax.set_title('Management-Climate Interaction on Additionality', fontsize=16, fontweight='bold')
        ax.set_xlabel('Climate Scenario', fontsize=12)
        ax.set_ylabel('Management Level', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_summary_statistics(self, data: pd.DataFrame, output_path: Path):
        """Plot summary statistics."""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Calculate key statistics
        stats = {
            'Total Scenarios': len(data),
            'Avg Additionality': f"{data['management_additionality'].mean():.1f} t CO2e/ha",
            'Max Additionality': f"{data['management_additionality'].max():.1f} t CO2e/ha",
            'Avg NPV': f"${data['management_npv'].mean():.0f}/ha",
            'Best Forest Type': data.groupby('forest_type')['management_additionality'].mean().idxmax(),
            'Best Management': data.groupby('management')['management_additionality'].mean().idxmax(),
            'Best Climate': data.groupby('climate')['management_additionality'].mean().idxmax()
        }
        
        # Create text summary
        text = "Key Statistics:\n\n"
        for key, value in stats.items():
            text += f"{key}: {value}\n"
        
        ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        ax.set_title('Summary Statistics', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to generate individual plots."""
    print("🎨 Individual Plot Generator for Forest Carbon Analysis")
    print("=" * 60)
    
    # Initialize plot generator
    generator = IndividualPlotGeneratorWithSample()
    
    # Generate all individual plots
    generator.generate_all_individual_plots()
    
    print("\n📋 Plot Summary:")
    print("1. Climate Impact on Carbon Stocks")
    print("2. Management Effectiveness")
    print("3. Forest Type Performance Comparison")
    print("4. Economic Performance Heatmap")
    print("5. Additionality Analysis")
    print("6. Disturbance Resilience")
    print("7. Top Scenario Rankings")
    print("8. Cost Effectiveness Analysis")
    print("9. NPV Distribution")
    print("10. Carbon Stock Evolution")
    print("11. Management-Climate Interaction")
    print("12. Summary Statistics")

if __name__ == "__main__":
    main()
