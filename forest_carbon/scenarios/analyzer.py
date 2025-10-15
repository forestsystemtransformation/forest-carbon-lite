#!/usr/bin/env python3
"""
Scenario Analysis and Visualization

Analyzes batch results and generates comprehensive comparison plots and reports.
This is a new version that fixes the indentation and functionality issues.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List
import yaml
import json

class ScenarioAnalyzer:
    """Analyze and visualize scenario results."""
    
    def _format_scenario_name(self, scenario: str) -> str:
        """Format scenario name for display."""
        if scenario.lower() == 'baseline':
            return 'Degrading baseline'
        elif scenario.lower() == 'management':
            return 'AFM (Active/Adaptive)'
        elif scenario.lower() == 'reforestation':
            return 'Reforestation'
        else:
            return scenario.capitalize()
    
    def __init__(self, results_path: Path = Path("output/batch_results.csv")):
        """
        Initialize analyzer.

        Args:
            results_path: Path to batch results CSV
        """
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
    
    def generate_comparison_plots(self, output_dir: Path = Path("output/analysis")):
        """Generate comprehensive comparison plots."""
        if self.results.empty:
            print("No results to analyze")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        # Filter successful results
        successful = self.results[self.results['status'] == 'success'].copy()

        if successful.empty:
            print("No successful results to analyze")
            return

        # Setup plot style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))

        # 1. Climate Impact on Carbon Stocks
        ax1 = plt.subplot(3, 3, 1)
        self._plot_climate_impact_comprehensive(successful, ax1)

        # 2. Management Effectiveness
        ax2 = plt.subplot(3, 3, 2)
        self._plot_management_effectiveness_comprehensive(successful, ax2)

        # 3. Forest Type Comparison
        ax3 = plt.subplot(3, 3, 3)
        self._plot_forest_type_comparison_comprehensive(successful, ax3)

        # 4. Economic Performance Heatmap
        ax4 = plt.subplot(3, 3, 4)
        self._plot_economic_heatmap_comprehensive(successful, ax4)

        # 5. Additionality Analysis
        ax5 = plt.subplot(3, 3, 5)
        self._plot_additionality_analysis(successful, ax5)

        # 6. Disturbance Resilience
        ax6 = plt.subplot(3, 3, 6)
        self._plot_disturbance_resilience(successful, ax6)

        # 7. Scenario Ranking
        ax7 = plt.subplot(3, 3, 7)
        self._plot_scenario_ranking(successful, ax7)

        # 8. Cost Effectiveness
        ax8 = plt.subplot(3, 3, 8)
        self._plot_cost_effectiveness(successful, ax8)

        # 9. Summary Statistics
        ax9 = plt.subplot(3, 3, 9)
        self._plot_summary_statistics(successful, ax9)

        plt.suptitle('Comprehensive Forest Carbon Scenario Analysis', fontsize=20, fontweight='bold')
        plt.tight_layout()
        
        # Save plots
        plt.savefig(output_dir / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive analysis plots saved to: {output_dir}/")
    
    
    def _plot_climate_impact(self, data: pd.DataFrame, ax):
        """Plot climate impact on carbon stocks."""
        pivot = data.pivot_table(
            values='management_final_co2e',
            index='climate',
            columns='forest_type',
            aggfunc='mean'
        )
        
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Climate Impact on Final Carbon Stock', fontweight='bold')
        ax.set_ylabel('CO2e (t/ha)')
        ax.legend(title='Forest Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_management_effectiveness(self, data: pd.DataFrame, ax):
        """Plot management effectiveness."""
        pivot = data.pivot_table(
            values='management_additionality',
            index='management',
            columns='forest_type',
            aggfunc='mean'
        )
        
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Management Effectiveness (Carbon Abatement)', fontweight='bold')
        ax.set_ylabel('Additional CO2e (t/ha)')
        ax.legend(title='Forest Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_forest_type_comparison(self, data: pd.DataFrame, ax):
        """Plot forest type comparison."""
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
        
        ax.set_title('Forest Type Performance Comparison', fontweight='bold')
        ax.set_ylabel('Value')
        ax.set_xticks(x)
        ax.set_xticklabels(forest_summary['forest_type'])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_economic_heatmap(self, data: pd.DataFrame, ax):
        """Plot economic performance heatmap."""
        pivot = data.pivot_table(
            values='management_npv',
            index='management',
            columns='climate',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot, annot=True, fmt='.0f', ax=ax, cmap='RdYlGn', center=0)
        ax.set_title('NPV by Management and Climate', fontweight='bold')
    
    def _plot_additionality_analysis(self, data: pd.DataFrame, ax):
        """Plot additionality analysis."""
        scatter = ax.scatter(data['baseline_final_co2e'], data['management_additionality'], 
                           c=data['management_npv'], cmap='viridis', alpha=0.7, s=100)
        
        ax.set_xlabel('Baseline Carbon Stock (t CO2e/ha)')
        ax.set_ylabel('Management Additionality (t CO2e/ha)')
        ax.set_title('Additionality vs Baseline Stock', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('NPV ($/ha)')
    
    def _plot_disturbance_resilience(self, data: pd.DataFrame, ax):
        """Plot disturbance resilience."""
        pivot = data.pivot_table(
            values='management_disturbances',
            index='climate',
            columns='management',
            aggfunc='mean'
        )
        
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Disturbance Events by Climate and Management', fontweight='bold')
        ax.set_ylabel('Number of Disturbances')
        ax.legend(title='Management Level', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_scenario_ranking(self, data: pd.DataFrame, ax):
        """Plot top scenarios ranking."""
        # Calculate combined score
        data['combined_score'] = (
            (data['management_additionality'] / data['management_additionality'].max()) * 0.6 +
            (data['management_npv'] / data['management_npv'].max()) * 0.4
        )
        
        top_scenarios = data.nlargest(10, 'combined_score')
        
        bars = ax.barh(range(len(top_scenarios)), top_scenarios['combined_score'])
        ax.set_yticks(range(len(top_scenarios)))
        ax.set_yticklabels([name.replace('_', '\n') for name in top_scenarios['scenario']])
        ax.set_xlabel('Combined Score')
        ax.set_title('Top 10 Scenarios by Combined Score', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Color bars by forest type
        colors = {'ETOF': '#1f77b4', 'EOF': '#ff7f0e', 'AFW': '#2ca02c'}
        for i, (_, row) in enumerate(top_scenarios.iterrows()):
            bars[i].set_color(colors.get(row['forest_type'], '#888888'))
    
    def _plot_cost_effectiveness(self, data: pd.DataFrame, ax):
        """Plot cost effectiveness analysis."""
        # Assume intervention costs based on management level
        cost_mapping = {
            'baseline': 0,
            'light': 200,
            'moderate': 500,
            'intensive': 1000,
            'adaptive': 500,
            'adaptive_managed_reforestation': 2000,
            'intensive_managed_reforestation': 3000
        }
        
        data['intervention_cost'] = data['management'].map(cost_mapping)
        # Fill missing values with default cost
        data['intervention_cost'] = data['intervention_cost'].fillna(1000)
        
        data['cost_effectiveness'] = data['management_additionality'] / (data['intervention_cost'] + 1)
        
        pivot = data.pivot_table(
            values='cost_effectiveness',
            index='management',
            columns='forest_type',
            aggfunc='mean'
        )
        
        # Check if pivot table has data
        if pivot.empty or pivot.isna().all().all():
            ax.text(0.5, 0.5, 'No cost effectiveness data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Cost Effectiveness (t CO2e per $)', fontweight='bold')
            return
        
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Cost Effectiveness (t CO2e per $)', fontweight='bold')
        ax.set_ylabel('CO2e per Dollar')
        ax.legend(title='Forest Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_summary_statistics(self, data: pd.DataFrame, ax):
        """Plot summary statistics."""
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
        
        ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    def generate_report(self, output_dir: Path = Path("output/analysis")):
        """Generate comprehensive analysis report."""
        if self.results.empty:
            print("No results to analyze")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter successful results
        successful = self.results[self.results['status'] == 'success'].copy()
        
        if successful.empty:
            print("No successful results to analyze")
            return

        # Generate plots
        self.generate_comparison_plots(output_dir)
        
        # Generate individual plots
        self._generate_individual_plots(successful, output_dir)
        
        # Generate statistical analysis
        self._generate_statistical_analysis(successful, output_dir)
        
        # Generate summary report
        self._generate_summary_report(successful, output_dir)
        
        print(f"Analysis complete! Results saved to: {output_dir}/")
    
    def _generate_statistical_analysis(self, data: pd.DataFrame, output_dir: Path):
        """Generate statistical analysis."""
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
        
        # Save statistical results
        with open(output_dir / 'statistical_analysis.json', 'w') as f:
            json.dump(stats_results, f, indent=2, default=str)
    
    def _generate_summary_report(self, data: pd.DataFrame, output_dir: Path):
        """Generate summary report."""
        report = f"""# Forest Carbon Scenario Analysis Report

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
        
        report += """

---
*Report generated by Forest Carbon Scenario Analysis System*
"""
        
        with open(output_dir / "scenario_analysis_report.md", "w", encoding='utf-8') as f:
            f.write(report)
    
    def _generate_individual_plots(self, data: pd.DataFrame, output_dir: Path):
        """Generate all individual plots."""
        print("📊 Generating individual plots...")
        
        # Create individual_plots subdirectory
        individual_dir = output_dir / "individual_plots"
        individual_dir.mkdir(exist_ok=True)
        
        # 1. Climate Impact Plot
        self._plot_climate_impact(data, individual_dir / "01_climate_impact.png")
        
        # 2. Management Effectiveness Plot
        self._plot_management_effectiveness(data, individual_dir / "02_management_effectiveness.png")
        
        # 3. Forest Type Comparison Plot
        self._plot_forest_type_comparison(data, individual_dir / "03_forest_type_comparison.png")
        
        # 4. Economic Performance Heatmap
        self._plot_economic_heatmap_individual(data, individual_dir / "04_economic_heatmap.png")
        
        # 5. Additionality Analysis Plot
        self._plot_additionality_analysis_individual(data, individual_dir / "05_additionality_analysis.png")
        
        # 6. Disturbance Resilience Plot
        self._plot_disturbance_resilience_individual(data, individual_dir / "06_disturbance_resilience.png")
        
        # 7. Scenario Ranking Plot
        self._plot_scenario_ranking_individual(data, individual_dir / "07_scenario_ranking.png")
        
        # 8. Cost Effectiveness Plot
        self._plot_cost_effectiveness_individual(data, individual_dir / "08_cost_effectiveness.png")
        
        # 9. NPV Distribution Plot
        self._plot_npv_distribution_individual(data, individual_dir / "09_npv_distribution.png")
        
        # 10. Carbon Stock Evolution Plot
        self._plot_carbon_evolution_individual(data, individual_dir / "10_carbon_evolution.png")
        
        # 11. Management vs Climate Interaction Plot
        self._plot_management_climate_interaction_individual(data, individual_dir / "11_management_climate_interaction.png")
        
        # 12. Summary Statistics Plot
        self._plot_summary_statistics_individual(data, individual_dir / "12_summary_statistics.png")
        
        print("✅ Individual plots generated")
    
    def _plot_climate_impact_comprehensive(self, data: pd.DataFrame, ax):
        """Plot climate impact on carbon stocks for comprehensive analysis."""
        pivot = data.pivot_table(
            values='management_final_co2e',
            index='climate',
            columns='forest_type',
            aggfunc='mean'
        )
        
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Climate Impact on Final Carbon Stock', fontweight='bold')
        ax.set_ylabel('CO2e (t/ha)')
        ax.set_xlabel('Climate Scenario')
        ax.legend(title='Forest Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_management_effectiveness_comprehensive(self, data: pd.DataFrame, ax):
        """Plot management effectiveness for comprehensive analysis."""
        data['abatement'] = data['management_final_co2e'] - data['baseline_final_co2e']
        
        pivot = data.pivot_table(
            values='abatement',
            index='management',
            columns='forest_type',
            aggfunc='mean'
        )
        
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Management Effectiveness (Carbon Abatement)', fontweight='bold')
        ax.set_ylabel('Additional CO2e (t/ha)')
        ax.set_xlabel('Management Level')
        ax.legend(title='Forest Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_forest_type_comparison_comprehensive(self, data: pd.DataFrame, ax):
        """Plot forest type comparison for comprehensive analysis."""
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
        
        ax.set_title('Forest Type Performance Comparison', fontweight='bold')
        ax.set_ylabel('Value')
        ax.set_xlabel('Forest Type')
        ax.set_xticks(x)
        ax.set_xticklabels(forest_summary['forest_type'])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_economic_heatmap_comprehensive(self, data: pd.DataFrame, ax):
        """Plot economic performance heatmap for comprehensive analysis."""
        pivot = data.pivot_table(
            values='management_npv',
            index='management',
            columns='climate',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot, annot=True, fmt='.0f', ax=ax, cmap='RdYlGn', center=0)
        ax.set_title('NPV by Management and Climate', fontweight='bold')
        ax.set_xlabel('Climate Scenario')
        ax.set_ylabel('Management Level')
    
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
    
    def _plot_economic_heatmap_individual(self, data: pd.DataFrame, output_path: Path):
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
    
    def _plot_additionality_analysis_individual(self, data: pd.DataFrame, output_path: Path):
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
    
    def _plot_disturbance_resilience_individual(self, data: pd.DataFrame, output_path: Path):
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
    
    def _plot_scenario_ranking_individual(self, data: pd.DataFrame, output_path: Path):
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
    
    def _plot_cost_effectiveness_individual(self, data: pd.DataFrame, output_path: Path):
        """Plot cost effectiveness analysis."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Assume intervention costs based on management level
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
    
    def _plot_npv_distribution_individual(self, data: pd.DataFrame, output_path: Path):
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
    
    def _plot_carbon_evolution_individual(self, data: pd.DataFrame, output_path: Path):
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
    
    def _plot_management_climate_interaction_individual(self, data: pd.DataFrame, output_path: Path):
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
    
    def _plot_summary_statistics_individual(self, data: pd.DataFrame, output_path: Path):
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
    """Example usage of scenario analyzer."""
    
    print("🌲 Forest Carbon Scenario Analyzer")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = ScenarioAnalyzer()
    
    if analyzer.results.empty:
        print("No results found. Run batch analysis first.")
        return
    
    # Generate analysis
    analyzer.generate_report()
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
