#!/usr/bin/env python3
"""
Comprehensive Forest Carbon Scenario Analysis

Generates ALL individual plots, tables, and comprehensive outputs for scenario analysis.
This consolidates the functionality from the standalone comprehensive_analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import json
import warnings

class ComprehensiveAnalyzer:
    """Generate comprehensive analysis with all individual plots and tables."""
    
    def __init__(self, results_path: Path = Path("output/batch_results.csv")):
        """Initialize analyzer."""
        self.results_path = results_path
        self.results = None
        self.load_results()
    
    def load_results(self):
        """Load results from CSV."""
        if self.results_path.exists():
            self.results = pd.read_csv(self.results_path)
            print(f"Loaded {len(self.results)} scenario results")
        else:
            print(f"Results file not found: {self.results_path}")
            self.results = pd.DataFrame()
    
    def run_complete_analysis(self, output_dir: Path = Path("output_years/analysis")):
        """Run complete analysis generating all individual plots and tables."""
        if self.results.empty:
            print("No results to analyze")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter successful results
        successful = self.results[self.results['status'] == 'success'].copy()
        
        if successful.empty:
            print("No successful results to analyze")
            return

        print(f"🌲 Running Comprehensive Analysis on {len(successful)} scenarios")
        print("=" * 60)
        
        # Setup plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Generate all individual plots
        self._generate_individual_plots(successful, output_dir)
        
        # Generate all tables
        self._generate_tables(successful, output_dir)
        
        # Generate comprehensive analysis
        self._generate_comprehensive_analysis(successful, output_dir)
        
        # Generate statistical analysis
        self._generate_statistical_analysis(successful, output_dir)
        
        # Generate summary report
        self._generate_summary_report(successful, output_dir)
        
        print(f"\n✅ Complete Analysis Finished!")
        print(f"📁 All outputs saved to: {output_dir}/")
        print(f"📊 Generated {len(list(output_dir.glob('*.png')))} plots")
        print(f"📋 Generated {len(list(output_dir.glob('*.csv')))} tables")
        print(f"📄 Generated {len(list(output_dir.glob('*.md')))} reports")
    
    def _generate_individual_plots(self, data: pd.DataFrame, output_dir: Path):
        """Generate all individual plots."""
        print("📊 Generating individual plots...")
        
        # 1. Climate Impact Plot
        self._plot_climate_impact(data, output_dir / "01_climate_impact.png")
        
        # 2. Management Effectiveness Plot
        self._plot_management_effectiveness(data, output_dir / "02_management_effectiveness.png")
        
        # 3. Forest Type Comparison Plot
        self._plot_forest_type_comparison(data, output_dir / "03_forest_type_comparison.png")
        
        # 4. Economic Performance Heatmap
        self._plot_economic_heatmap(data, output_dir / "04_economic_heatmap.png")
        
        # 5. Additionality Analysis Plot
        self._plot_additionality_analysis(data, output_dir / "05_additionality_analysis.png")
        
        # 6. Disturbance Resilience Plot
        self._plot_disturbance_resilience(data, output_dir / "06_disturbance_resilience.png")
        
        # 7. Scenario Ranking Plot
        self._plot_scenario_ranking(data, output_dir / "07_scenario_ranking.png")
        
        # 8. Cost Effectiveness Plot
        self._plot_cost_effectiveness(data, output_dir / "08_cost_effectiveness.png")
        
        # 9. NPV Distribution Plot
        self._plot_npv_distribution(data, output_dir / "09_npv_distribution.png")
        
        # 10. Carbon Stock Evolution Plot
        self._plot_carbon_evolution(data, output_dir / "10_carbon_evolution.png")
        
        # 11. Management vs Climate Interaction Plot
        self._plot_management_climate_interaction(data, output_dir / "11_management_climate_interaction.png")
        
        # 12. Summary Statistics Plot
        self._plot_summary_statistics(data, output_dir / "12_summary_statistics.png")
        
        print("✅ Individual plots generated")
    
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
        colors = {'ETOF': '#1f77b4', 'EOF': '#ff7f0e', 'AFW': '#2ca02c'}
        for i, (_, row) in enumerate(top_scenarios.iterrows()):
            bars[i].set_color(colors.get(row['forest_type'], '#888888'))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cost_effectiveness(self, data: pd.DataFrame, output_path: Path):
        """Plot cost effectiveness analysis."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Assume intervention costs based on management level
        cost_mapping = {
            'baseline': 0,
            'light': 200,
            'moderate': 500,
            'intensive': 1000
        }
        
        data['intervention_cost'] = data['management'].map(cost_mapping)
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
        ax.scatter(data['baseline_final_co2e'], data['management_final_co2e'], 
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
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_tables(self, data: pd.DataFrame, output_dir: Path):
        """Generate all tables."""
        print("📋 Generating tables...")
        
        # 1. Scenario Summary Table
        scenario_summary = data[['scenario', 'forest_type', 'climate', 'management', 
                               'management_additionality', 'management_npv']].copy()
        scenario_summary.to_csv(output_dir / "scenario_summary.csv", index=False)
        
        # 2. Forest Type Performance Table
        forest_performance = data.groupby('forest_type').agg({
            'management_additionality': ['mean', 'std', 'min', 'max'],
            'management_npv': ['mean', 'std', 'min', 'max']
        }).round(2)
        forest_performance.to_csv(output_dir / "forest_type_performance.csv")
        
        # 3. Management Level Performance Table
        management_performance = data.groupby('management').agg({
            'management_additionality': ['mean', 'std', 'min', 'max'],
            'management_npv': ['mean', 'std', 'min', 'max']
        }).round(2)
        management_performance.to_csv(output_dir / "management_performance.csv")
        
        # 4. Climate Impact Table
        climate_impact = data.groupby('climate').agg({
            'management_additionality': ['mean', 'std', 'min', 'max'],
            'management_npv': ['mean', 'std', 'min', 'max']
        }).round(2)
        climate_impact.to_csv(output_dir / "climate_impact.csv")
        
        # 5. Top Scenarios Table
        top_scenarios = data.nlargest(10, 'management_additionality')[['scenario', 'forest_type', 'climate', 'management', 'management_additionality', 'management_npv']]
        top_scenarios.to_csv(output_dir / "top_scenarios.csv", index=False)
        
        # 6. Cost Effectiveness Table
        cost_mapping = {'baseline': 0, 'light': 200, 'moderate': 500, 'intensive': 1000}
        data['intervention_cost'] = data['management'].map(cost_mapping)
        data['cost_effectiveness'] = data['management_additionality'] / (data['intervention_cost'] + 1)
        
        cost_effectiveness = data.groupby('management').agg({
            'cost_effectiveness': ['mean', 'std', 'min', 'max'],
            'intervention_cost': 'mean'
        }).round(2)
        cost_effectiveness.to_csv(output_dir / "cost_effectiveness.csv")
        
        print("✅ Tables generated")
    
    def _generate_comprehensive_analysis(self, data: pd.DataFrame, output_dir: Path):
        """Generate comprehensive analysis plot."""
        print("📊 Generating comprehensive analysis...")
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Climate Impact
        ax1 = plt.subplot(3, 3, 1)
        pivot = data.pivot_table(values='management_final_co2e', index='climate', columns='forest_type', aggfunc='mean')
        pivot.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Climate Impact on Carbon Stock', fontweight='bold')
        ax1.set_ylabel('CO2e (t/ha)')
        ax1.legend(title='Forest Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. Management Effectiveness
        ax2 = plt.subplot(3, 3, 2)
        data['abatement'] = data['management_final_co2e'] - data['baseline_final_co2e']
        pivot = data.pivot_table(values='abatement', index='management', columns='forest_type', aggfunc='mean')
        pivot.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Management Effectiveness', fontweight='bold')
        ax2.set_ylabel('Additional CO2e (t/ha)')
        ax2.legend(title='Forest Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Economic Performance
        ax3 = plt.subplot(3, 3, 3)
        pivot = data.pivot_table(values='management_npv', index='management', columns='climate', aggfunc='mean')
        sns.heatmap(pivot, annot=True, fmt='.0f', ax=ax3, cmap='RdYlGn', center=0)
        ax3.set_title('NPV by Management and Climate', fontweight='bold')
        
        # 4. Additionality Analysis
        ax4 = plt.subplot(3, 3, 4)
        scatter = ax4.scatter(data['baseline_final_co2e'], data['abatement'], 
                            c=data['management_npv'], cmap='viridis', alpha=0.7, s=100)
        ax4.set_xlabel('Baseline Carbon Stock (t CO2e/ha)')
        ax4.set_ylabel('Management Additionality (t CO2e/ha)')
        ax4.set_title('Additionality vs Baseline Stock', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('NPV ($/ha)')
        
        # 5. Scenario Ranking
        ax5 = plt.subplot(3, 3, 5)
        data['combined_score'] = (
            (data['management_additionality'] / data['management_additionality'].max()) * 0.6 +
            (data['management_npv'] / data['management_npv'].max()) * 0.4
        )
        top_scenarios = data.nlargest(10, 'combined_score')
        bars = ax5.barh(range(len(top_scenarios)), top_scenarios['combined_score'])
        ax5.set_yticks(range(len(top_scenarios)))
        ax5.set_yticklabels([name.replace('_', '\n') for name in top_scenarios['scenario']])
        ax5.set_xlabel('Combined Score')
        ax5.set_title('Top 10 Scenarios', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Cost Effectiveness
        ax6 = plt.subplot(3, 3, 6)
        cost_mapping = {'baseline': 0, 'light': 200, 'moderate': 500, 'intensive': 1000}
        data['intervention_cost'] = data['management'].map(cost_mapping)
        data['cost_effectiveness'] = data['management_additionality'] / (data['intervention_cost'] + 1)
        pivot = data.pivot_table(values='cost_effectiveness', index='management', columns='forest_type', aggfunc='mean')
        pivot.plot(kind='bar', ax=ax6, width=0.8)
        ax6.set_title('Cost Effectiveness', fontweight='bold')
        ax6.set_ylabel('CO2e per Dollar')
        ax6.legend(title='Forest Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)
        
        # 7. NPV Distribution
        ax7 = plt.subplot(3, 3, 7)
        sns.boxplot(data=data, x='management', y='management_npv', ax=ax7)
        ax7.set_title('NPV Distribution by Management', fontweight='bold')
        ax7.set_ylabel('NPV ($/ha)')
        ax7.tick_params(axis='x', rotation=45)
        ax7.grid(True, alpha=0.3)
        
        # 8. Carbon Evolution
        ax8 = plt.subplot(3, 3, 8)
        ax8.scatter(data['baseline_final_co2e'], data['management_final_co2e'], 
                   alpha=0.7, s=100, c=data['management_npv'], cmap='viridis')
        min_val = min(data['baseline_final_co2e'].min(), data['management_final_co2e'].min())
        max_val = max(data['baseline_final_co2e'].max(), data['management_final_co2e'].max())
        ax8.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='No Change')
        ax8.set_xlabel('Baseline Final CO2e (t/ha)')
        ax8.set_ylabel('Management Final CO2e (t/ha)')
        ax8.set_title('Carbon Stock Evolution', fontweight='bold')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Summary Statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        stats = {
            'Total Scenarios': len(data),
            'Avg Additionality': f"{data['management_additionality'].mean():.1f} t CO2e/ha",
            'Max Additionality': f"{data['management_additionality'].max():.1f} t CO2e/ha",
            'Avg NPV': f"${data['management_npv'].mean():.0f}/ha",
            'Best Forest Type': data.groupby('forest_type')['management_additionality'].mean().idxmax(),
            'Best Management': data.groupby('management')['management_additionality'].mean().idxmax(),
            'Best Climate': data.groupby('climate')['management_additionality'].mean().idxmax()
        }
        text = "Key Statistics:\n\n"
        for key, value in stats.items():
            text += f"{key}: {value}\n"
        ax9.text(0.1, 0.9, text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Comprehensive Forest Carbon Scenario Analysis', fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'comprehensive_analysis.svg', bbox_inches='tight')
        plt.close()
        
        print("✅ Comprehensive analysis generated")
    
    def _generate_statistical_analysis(self, data: pd.DataFrame, output_dir: Path):
        """Generate statistical analysis."""
        print("📈 Generating statistical analysis...")
        
        from scipy import stats
        
        stats_results = {}
        
        # ANOVA for climate effect (only if we have multiple climate scenarios)
        if data['climate'].nunique() > 1:
            climate_groups = [group['management_additionality'].values 
                              for name, group in data.groupby('climate')]
            f_stat, p_value = stats.f_oneway(*climate_groups)
            stats_results['climate_effect'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        else:
            stats_results['climate_effect'] = {
                'f_statistic': None,
                'p_value': None,
                'significant': None,
                'note': 'Single climate scenario - no comparison possible'
            }
        
        # ANOVA for management effect (only if we have multiple management levels)
        if data['management'].nunique() > 1:
            mgmt_groups = [group['management_additionality'].values 
                           for name, group in data.groupby('management')]
            f_stat_mgmt, p_value_mgmt = stats.f_oneway(*mgmt_groups)
            stats_results['management_effect'] = {
                'f_statistic': f_stat_mgmt,
                'p_value': p_value_mgmt,
                'significant': p_value_mgmt < 0.05
            }
        else:
            stats_results['management_effect'] = {
                'f_statistic': None,
                'p_value': None,
                'significant': None,
                'note': 'Single management level - no comparison possible'
            }
        
        # ANOVA for forest type effect (only if we have multiple forest types)
        if data['forest_type'].nunique() > 1:
            forest_groups = [group['management_additionality'].values 
                             for name, group in data.groupby('forest_type')]
            f_stat_forest, p_value_forest = stats.f_oneway(*forest_groups)
            stats_results['forest_type_effect'] = {
                'f_statistic': f_stat_forest,
                'p_value': p_value_forest,
                'significant': p_value_forest < 0.05
            }
        else:
            stats_results['forest_type_effect'] = {
                'f_statistic': None,
                'p_value': None,
                'significant': None,
                'note': 'Single forest type - no comparison possible'
            }
        
        # Add descriptive statistics for single-group scenarios
        if data['forest_type'].nunique() == 1:
            forest_type = data['forest_type'].iloc[0]
            stats_results['descriptive_stats'] = {
                'forest_type': forest_type,
                'mean_additionality': data['management_additionality'].mean(),
                'std_additionality': data['management_additionality'].std(),
                'min_additionality': data['management_additionality'].min(),
                'max_additionality': data['management_additionality'].max(),
                'mean_npv': data['management_npv'].mean(),
                'std_npv': data['management_npv'].std()
            }
        
        # Save statistical results
        with open(output_dir / 'statistical_analysis.json', 'w') as f:
            json.dump(stats_results, f, indent=2, default=str)
        
        print("✅ Statistical analysis generated")
    
    def _generate_summary_report(self, data: pd.DataFrame, output_dir: Path):
        """Generate summary report."""
        print("📄 Generating summary report...")
        
        report = f"""# Comprehensive Forest Carbon Scenario Analysis Report

## Executive Summary

This report analyzes {len(data)} forest carbon scenarios across different forest types, climate conditions, and management levels.

## Key Findings

### Best Performing Scenarios

"""
        
        # Top scenarios by additionality
        top_additionality = data.nlargest(5, 'management_additionality')
        report += "#### Top 5 Scenarios by Carbon Additionality:\n\n"
        for i, (_, row) in enumerate(top_additionality.iterrows(), 1):
            report += f"{i}. **{row['scenario']}**: {row['management_additionality']:.1f} t CO2e/ha\n"
        
        # Top scenarios by NPV
        top_npv = data.nlargest(5, 'management_npv')
        report += "\n#### Top 5 Scenarios by NPV:\n\n"
        for i, (_, row) in enumerate(top_npv.iterrows(), 1):
            report += f"{i}. **{row['scenario']}**: ${row['management_npv']:.0f}/ha\n"
        
        # Forest type comparison
        forest_summary = data.groupby('forest_type').agg({
            'management_additionality': 'mean',
            'management_npv': 'mean'
        }).round(1)
        
        report += "\n### Forest Type Performance:\n\n"
        report += "| Forest Type | Avg Additionality (t CO2e/ha) | Avg NPV ($/ha) |\n"
        report += "|-------------|-------------------------------|----------------|\n"
        for forest, row in forest_summary.iterrows():
            report += f"| {forest} | {row['management_additionality']} | {row['management_npv']} |\n"
        
        # Management level comparison
        mgmt_summary = data.groupby('management').agg({
            'management_additionality': 'mean',
            'management_npv': 'mean'
        }).round(1)
        
        report += "\n### Management Level Performance:\n\n"
        report += "| Management Level | Avg Additionality (t CO2e/ha) | Avg NPV ($/ha) |\n"
        report += "|------------------|-------------------------------|----------------|\n"
        for mgmt, row in mgmt_summary.iterrows():
            report += f"| {mgmt} | {row['management_additionality']} | {row['management_npv']} |\n"
        
        # Climate impact
        climate_summary = data.groupby('climate').agg({
            'management_additionality': 'mean',
            'management_npv': 'mean'
        }).round(1)
        
        report += "\n### Climate Impact:\n\n"
        report += "| Climate Scenario | Avg Additionality (t CO2e/ha) | Avg NPV ($/ha) |\n"
        report += "|------------------|-------------------------------|----------------|\n"
        for climate, row in climate_summary.iterrows():
            report += f"| {climate} | {row['management_additionality']} | {row['management_npv']} |\n"
        
        report += f"""

## Statistical Analysis

"""
        
        # Add statistical analysis based on what's available
        if data['climate'].nunique() > 1:
            climate_var = data.groupby('climate')['management_additionality'].var().mean()
            report += f"- **Climate Effect**: Variance = {climate_var:.2f}\n"
        
        if data['management'].nunique() > 1:
            mgmt_var = data.groupby('management')['management_additionality'].var().mean()
            report += f"- **Management Effect**: Variance = {mgmt_var:.2f}\n"
        
        if data['forest_type'].nunique() > 1:
            forest_var = data.groupby('forest_type')['management_additionality'].var().mean()
            report += f"- **Forest Type Effect**: Variance = {forest_var:.2f}\n"
        
        # Add descriptive statistics for single forest type
        if data['forest_type'].nunique() == 1:
            forest_type = data['forest_type'].iloc[0]
            report += f"- **{forest_type} Forest Analysis**:\n"
            report += f"  - Mean Additionality: {data['management_additionality'].mean():.1f} t CO2e/ha\n"
            report += f"  - Standard Deviation: {data['management_additionality'].std():.1f} t CO2e/ha\n"
            report += f"  - Range: {data['management_additionality'].min():.1f} - {data['management_additionality'].max():.1f} t CO2e/ha\n"
        
        report += """

## Recommendations

"""
        
        # Add recommendations based on available data
        if data['forest_type'].nunique() > 1:
            best_forest = data.groupby('forest_type')['management_additionality'].mean().idxmax()
            report += f"1. **Best Forest Type**: {best_forest} shows highest average additionality\n"
        
        if data['management'].nunique() > 1:
            best_mgmt = data.groupby('management')['management_additionality'].mean().idxmax()
            report += f"2. **Best Management Level**: {best_mgmt} provides best carbon benefits\n"
        
        if data['climate'].nunique() > 1:
            worst_climate = data.groupby('climate')['management_additionality'].mean().idxmin()
            report += f"3. **Climate Resilience**: Consider climate adaptation strategies for {worst_climate} scenarios\n"
        
        # Add specific recommendations for single forest type analysis
        if data['forest_type'].nunique() == 1:
            forest_type = data['forest_type'].iloc[0]
            best_scenario = data.loc[data['management_additionality'].idxmax()]
            report += f"1. **Best {forest_type} Scenario**: {best_scenario['scenario']} with {best_scenario['management_additionality']:.1f} t CO2e/ha additionality\n"
            report += f"2. **Management Strategy**: {best_scenario['management']} management provides optimal carbon benefits\n"
            report += f"3. **Climate Considerations**: {best_scenario['climate']} climate shows best performance in this analysis\n"
        
        report += """

## Files Generated

### Individual Plots:
- `01_climate_impact.png`: Climate impact on carbon stocks
- `02_management_effectiveness.png`: Management effectiveness analysis
- `03_forest_type_comparison.png`: Forest type performance comparison
- `04_economic_heatmap.png`: Economic performance heatmap
- `05_additionality_analysis.png`: Additionality vs baseline analysis
- `06_disturbance_resilience.png`: Disturbance resilience analysis
- `07_scenario_ranking.png`: Top scenarios ranking
- `08_cost_effectiveness.png`: Cost effectiveness analysis
- `09_npv_distribution.png`: NPV distribution by management
- `10_carbon_evolution.png`: Carbon stock evolution
- `11_management_climate_interaction.png`: Management-climate interaction
- `12_summary_statistics.png`: Key statistics summary

### Tables:
- `scenario_summary.csv`: Complete scenario summary
- `forest_type_performance.csv`: Forest type performance metrics
- `management_performance.csv`: Management level performance metrics
- `climate_impact.csv`: Climate impact metrics
- `top_scenarios.csv`: Top performing scenarios
- `cost_effectiveness.csv`: Cost effectiveness analysis

### Comprehensive Analysis:
- `comprehensive_analysis.png`: Complete visualization suite
- `comprehensive_analysis.svg`: Vector version of comprehensive analysis
- `statistical_analysis.json`: Statistical test results
- `comprehensive_report.md`: This report

---
*Report generated by Comprehensive Forest Carbon Scenario Analysis System*
"""
        
        with open(output_dir / "comprehensive_report.md", "w", encoding='utf-8') as f:
            f.write(report)
        
        print("✅ Summary report generated")
    

def main():
    """Run comprehensive analysis."""
    print("🌲 Comprehensive Forest Carbon Scenario Analysis")
    print("=" * 60)
    
    analyzer = ComprehensiveAnalyzer()
    analyzer.run_complete_analysis()
    
    print("\n🎉 Analysis Complete!")
    print("📁 Check the 'output_years/analysis' directory for all results")

if __name__ == "__main__":
    main()
