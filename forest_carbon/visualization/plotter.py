"""Visualization module for creating publication-quality plots."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.colors import color_manager

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class Plotter:
    """Creates various plots for carbon sequestration analysis."""
    
    def _format_label(self, text: str) -> str:
        """Format text to proper sentence case."""
        # Handle special cases
        if text.lower() == 'co2e':
            return 'CO2e'
        elif text.lower() == 'agb':
            return 'AGB'
        elif text.lower() == 'bgb':
            return 'BGB'
        elif text.lower() == 'baseline':
            return 'Degrading baseline'
        elif text.lower() == 'management':
            return 'AFM (Active/Adaptive)'
        elif text.lower() == 'reforestation':
            return 'Reforestation'
        else:
            # Convert to sentence case (first letter uppercase, rest lowercase)
            return text.capitalize()
    
    def __init__(self, output_dir: Path):
        """
        Initialize plotter with output directory.
        
        Args:
            output_dir: Directory for saving plots
        """
        self.output_dir = output_dir
        self.plots_dir = output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default figure parameters
        self.fig_width = 12  # Increased for better readability
        self.fig_height = 8  # Increased to accommodate legends underneath
        self.dpi = 300
        
        # Set default font sizes for publication quality
        self.title_fontsize = 18
        self.label_fontsize = 16
        self.legend_fontsize = 14
        self.tick_fontsize = 14
    
    def _save_figure(self, filename: str, formats: List[str] = ['png', 'svg']) -> Dict[str, Path]:
        """
        Save current figure in multiple formats and return paths.
        
        Args:
            filename: Base filename (without extension)
            formats: List of formats to save ('png', 'svg', 'pdf', 'eps')
            
        Returns:
            Dictionary mapping format to file path
        """
        paths = {}
        
        for fmt in formats:
            if fmt == 'png':
                path = self.plots_dir / f'{filename}.png'
                plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
            elif fmt == 'svg':
                path = self.plots_dir / f'{filename}.svg'
                plt.savefig(path, bbox_inches='tight')
            elif fmt == 'pdf':
                path = self.plots_dir / f'{filename}.pdf'
                plt.savefig(path, bbox_inches='tight')
            elif fmt == 'eps':
                path = self.plots_dir / f'{filename}.eps'
                plt.savefig(path, bbox_inches='tight')
            else:
                continue
            
            paths[fmt] = path
        
        plt.close()
        return paths
    
    def plot_biomass_comparison(self, results: Dict[str, pd.DataFrame], 
                               forest_types: List[str]):
        """
        Plot AGB comparison across scenarios.
        
        Args:
            results: Dictionary of results DataFrames
            forest_types: List of forest types
        """
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        for forest_type in forest_types:
            df = results[forest_type]
            
            # Check data format and plot accordingly
            if 'scenario' in df.columns:
                # Calendar years format - plot by scenario
                for scenario in ['baseline', 'management', 'reforestation']:
                    scenario_data = df[df['scenario'] == scenario]
                    if not scenario_data.empty:
                        label = f'{forest_type} - {self._format_label(scenario)}'
                        scenario_style = color_manager.get_scenario_style(scenario)
                        ax.plot(scenario_data['year'], scenario_data['total_agb'], 
                               label=label, 
                               **scenario_style)
            else:
                # Regular format - plot by scenario columns
                for scenario in ['baseline', 'management', 'reforestation']:
                    if f'{scenario}_agb' in df.columns:
                        label = f'{forest_type} - {self._format_label(scenario)}'
                        scenario_style = color_manager.get_scenario_style(scenario)
                        ax.plot(df['year'], df[f'{scenario}_agb'], 
                               label=label, 
                               **scenario_style)
        
        ax.set_xlabel('Year', fontsize=self.label_fontsize, fontweight='bold')
        ax.set_ylabel('Above-ground biomass (t/ha)', fontsize=self.label_fontsize, fontweight='bold')
        ax.set_title('Biomass accumulation - all scenarios', fontsize=self.title_fontsize, fontweight='bold', pad=20)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=self.legend_fontsize)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=self.tick_fontsize)
        
        plt.tight_layout()
        return self._save_figure('biomass_all_scenarios')
    
    def plot_carbon_pools_breakdown(self, pools_data: Dict[str, List[Dict]], 
                                   scenario: str):
        """
        Plot breakdown of carbon pools over time for a scenario.
        
        Args:
            pools_data: Carbon pools data over time
            scenario: Scenario name
        """
        # Convert to DataFrame
        df = pd.DataFrame(pools_data[scenario])
        
        # Select pools to plot
        pools = ['agb', 'bgb', 'litter', 'active_soil', 'slow_soil', 'char']
        
        fig, axes = plt.subplots(2, 3, figsize=(self.fig_width * 1.5, self.fig_height * 1.5))
        axes = axes.flatten()
        
        pool_names = {
            'agb': 'Above-ground biomass',
            'bgb': 'Below-ground biomass',
            'litter': 'Litter',
            'active_soil': 'Active soil carbon',
            'slow_soil': 'Slow soil carbon',
            'char': 'Charcoal'
        }
        
        for idx, pool in enumerate(pools):
            ax = axes[idx]
            if pool in df.columns:
                # Get color from color manager
                pool_color = color_manager.get_carbon_pool_color(pool)
                # Convert to CO2e
                values = df[pool] * 3.67  # C to CO2
                ax.fill_between(df['year'], 0, values, alpha=0.7, color=pool_color)
                ax.plot(df['year'], values, color=pool_color, linewidth=2)
                
                ax.set_xlabel('Year', fontsize=self.label_fontsize, fontweight='bold')
                ax.set_ylabel('CO2e (t/ha)', fontsize=self.label_fontsize, fontweight='bold')
                ax.set_title(pool_names[pool], fontsize=self.title_fontsize, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', which='major', labelsize=self.tick_fontsize)
        
        plt.suptitle(f'Carbon pools breakdown - {self._format_label(scenario)}', 
                    fontsize=self.title_fontsize, fontweight='bold')
        plt.tight_layout()
        return self._save_figure(f'carbon_pools_breakdown_{scenario}')
    
    def plot_carbon_pools_comparison(self, final_pools: Dict[str, Dict]):
        """
        Plot stacked bar chart comparing final carbon pools.
        
        Args:
            final_pools: Final carbon pools for each scenario
        """
        scenarios = list(final_pools.keys())
        pools = ['agb', 'bgb', 'litter', 'active_soil', 'slow_soil', 'char', 'slash']
        
        # Prepare data
        data = {pool: [] for pool in pools}
        
        for scenario in scenarios:
            for pool in pools:
                # Convert to CO2e
                value = final_pools[scenario].get(pool, 0) * 3.67
                data[pool].append(value)
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        x = np.arange(len(scenarios))
        width = 0.6
        
        bottom = np.zeros(len(scenarios))
        for idx, pool in enumerate(pools):
            pool_color = color_manager.get_carbon_pool_color(pool)
            ax.bar(x, data[pool], width, label=self._format_label(pool.replace('_', ' ')),
                  bottom=bottom, color=pool_color)
            bottom += data[pool]
        
        ax.set_xlabel('Scenario', fontsize=self.label_fontsize, fontweight='bold')
        ax.set_ylabel('Carbon stock (t CO2e/ha)', fontsize=self.label_fontsize, fontweight='bold')
        ax.set_title('Final carbon pools comparison', fontsize=self.title_fontsize, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([self._format_label(s) for s in scenarios], fontsize=self.tick_fontsize)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=self.legend_fontsize)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', which='major', labelsize=self.tick_fontsize)
        
        plt.tight_layout()
        return self._save_figure('carbon_pools_comparison')
    
    def plot_economics(self, cashflow_df: pd.DataFrame, scenario: str):
        """
        Plot economic analysis results.
        
        Args:
            cashflow_df: Cashflow DataFrame
            scenario: Scenario name
        """
        fig, axes = plt.subplots(2, 2, figsize=(self.fig_width * 1.2, self.fig_height * 1.2))
        
        # Plot 1: Annual cashflow
        ax = axes[0, 0]
        revenue_color = color_manager.get_economic_color('revenue')
        costs_color = color_manager.get_economic_color('costs')
        net_color = color_manager.get_economic_color('net')
        
        ax.bar(cashflow_df['year'], cashflow_df['revenue'], alpha=0.7, 
              label='Revenue', color=revenue_color)
        ax.bar(cashflow_df['year'], -cashflow_df['total_costs'], alpha=0.7,
              label='Costs', color=costs_color)
        ax.plot(cashflow_df['year'], cashflow_df['net_cashflow'], 
               color=net_color, linewidth=2, label='Net')
        ax.set_xlabel('Year', fontsize=self.label_fontsize, fontweight='bold')
        ax.set_ylabel('Cash Flow ($)', fontsize=self.label_fontsize, fontweight='bold')
        ax.set_title('Annual cash flow', fontsize=self.title_fontsize, fontweight='bold')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=self.legend_fontsize)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=self.tick_fontsize)
        
        # Plot 2: Cumulative discounted cashflow
        ax = axes[0, 1]
        npv_color = color_manager.get_economic_color('npv')
        ax.plot(cashflow_df['year'], cashflow_df['cumulative_discounted'],
               linewidth=2, color=npv_color)
        ax.fill_between(cashflow_df['year'], 0, cashflow_df['cumulative_discounted'],
                       alpha=0.3, color=npv_color)
        ax.set_xlabel('Year', fontsize=self.label_fontsize, fontweight='bold')
        ax.set_ylabel('Cumulative NPV ($)', fontsize=self.label_fontsize, fontweight='bold')
        ax.set_title('Cumulative discounted cash flow', fontsize=self.title_fontsize, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=self.tick_fontsize)
        
        # Plot 3: Carbon price trajectory
        ax = axes[1, 0]
        price_color = color_manager.get_economic_color('carbon_price')
        ax.plot(cashflow_df['year'], cashflow_df['carbon_price'],
               linewidth=2, color=price_color)
        ax.set_xlabel('Year', fontsize=self.label_fontsize, fontweight='bold')
        ax.set_ylabel('Carbon price ($/tCO2e)', fontsize=self.label_fontsize, fontweight='bold')
        ax.set_title('Carbon price trajectory', fontsize=self.title_fontsize, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=self.tick_fontsize)
        
        # Plot 4: Credits generation
        ax = axes[1, 1]
        credits_color = color_manager.get_economic_color('revenue')  # Use revenue color for credits
        ax.bar(cashflow_df['year'], cashflow_df['credits_tCO2e'],
              alpha=0.7, color=credits_color)
        ax.set_xlabel('Year', fontsize=self.label_fontsize, fontweight='bold')
        ax.set_ylabel('Credits (tCO2e)', fontsize=self.label_fontsize, fontweight='bold')
        ax.set_title('Carbon credits generation', fontsize=self.title_fontsize, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=self.tick_fontsize)
        
        plt.suptitle(f'Economic analysis - {self._format_label(scenario)}', 
                    fontsize=self.title_fontsize, fontweight='bold')
        plt.tight_layout()
        return self._save_figure(f'economics_{scenario}')
    
    def plot_total_carbon_stocks_all_scenarios(self, results: Dict[str, pd.DataFrame], 
                                             forest_types: List[str]):
        """
        Plot total carbon stock for all three scenarios individually.
        
        Args:
            results: Dictionary of results DataFrames
            forest_types: List of forest types
        """
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        for forest_type in forest_types:
            df = results[forest_type]
            
            # Check data format and plot accordingly
            if 'scenario' in df.columns:
                # Calendar years format - plot by scenario
                baseline_data = df[df['scenario'] == 'baseline']
                management_data = df[df['scenario'] == 'management']
                reforestation_data = df[df['scenario'] == 'reforestation']
                
                # Check if baseline and management curves are identical (baseline management scenario)
                curves_identical = False
                if not baseline_data.empty and not management_data.empty:
                    baseline_sorted = baseline_data.sort_values('calendar_year')
                    management_sorted = management_data.sort_values('calendar_year')
                    if len(baseline_sorted) == len(management_sorted):
                        # Check if values are identical (within small tolerance)
                        curves_identical = np.allclose(
                            baseline_sorted['total_co2e'].values, 
                            management_sorted['total_co2e'].values, 
                            rtol=1e-10, atol=1e-10
                        )
                
                # Plot baseline
                if not baseline_data.empty:
                    label = f'{self._format_label("baseline")} (Total stock)'
                    style = color_manager.get_scenario_style('baseline')
                    if curves_identical:
                        # Special styling for identical curves
                        style['linewidth'] = 3
                        style['alpha'] = 0.8
                        style['linestyle'] = '-'
                    ax.plot(baseline_data['calendar_year'], baseline_data['total_co2e'], 
                           label=label, **style)
                
                # Plot management
                if not management_data.empty:
                    label = f'{self._format_label("management")} (Total stock)'
                    style = color_manager.get_scenario_style('management')
                    if curves_identical:
                        # Special styling for identical curves - make it dashed to show overlap
                        style['linewidth'] = 3
                        style['alpha'] = 0.8
                        style['linestyle'] = '--'
                        label += ' (OVERLAPPING)'
                    ax.plot(management_data['calendar_year'], management_data['total_co2e'], 
                           label=label, **style)
                
                # Plot reforestation
                if not reforestation_data.empty:
                    label = f'{self._format_label("reforestation")} (Total stock)'
                    style = color_manager.get_scenario_style('reforestation')
                    ax.plot(reforestation_data['calendar_year'], reforestation_data['total_co2e'], 
                           label=label, **style)
                
                # Add annotation if curves are identical
                if curves_identical:
                    ax.text(0.02, 0.98, '✓ BASELINE MANAGEMENT: Curves are identical\n(Red solid + Blue dashed = Perfect overlap)', 
                           transform=ax.transAxes, fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                           verticalalignment='top')
            else:
                # Regular format - plot by scenario columns
                for scenario in ['baseline', 'management', 'reforestation']:
                    if f'{scenario}_co2e' in df.columns:
                        label = f'{self._format_label(scenario)} (Total stock)'
                        style = color_manager.get_scenario_style(scenario)
                        ax.plot(df['year'], df[f'{scenario}_co2e'], 
                               label=label,
                               **style)
        
        # Customize the plot
        ax.set_xlabel('Year', fontsize=self.label_fontsize, fontweight='bold')
        ax.set_ylabel('Total carbon stock (t CO2e/ha)', fontsize=self.label_fontsize, fontweight='bold')
        ax.set_title('Total carbon stock - all scenarios', fontsize=self.title_fontsize, fontweight='bold', pad=20)
        
        # Add legend with better positioning
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=self.legend_fontsize)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add zero line for reference
        ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.6)
        
        # Add some styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
        # Set tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=self.tick_fontsize)
        
        # Set axis limits for better visualization
        # Don't force x-axis to start at 0 for calendar years
        
        plt.tight_layout()
        return self._save_figure('total_carbon_stocks_all_scenarios')
    
    def plot_net_carbon_abatement(self, results: Dict[str, pd.DataFrame], 
                                 forest_types: List[str]):
        """
        Plot net carbon abatement (carbon sequestration from baseline 0).
        Shows all carbon curves starting from 0 to highlight actual carbon benefits.
        
        Args:
            results: Dictionary of results DataFrames
            forest_types: List of forest types
        """
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        for forest_type in forest_types:
            df = results[forest_type]
            
            # Check data format and plot accordingly
            if 'scenario' in df.columns:
                # Calendar years format - plot by scenario
                baseline_data = df[df['scenario'] == 'baseline']
                management_data = df[df['scenario'] == 'management']
                reforestation_data = df[df['scenario'] == 'reforestation']
                
                # Check if baseline and management curves are identical (baseline management scenario)
                curves_identical = False
                if not baseline_data.empty and not management_data.empty:
                    baseline_sorted = baseline_data.sort_values('calendar_year')
                    management_sorted = management_data.sort_values('calendar_year')
                    if len(baseline_sorted) == len(management_sorted):
                        curves_identical = np.allclose(
                            baseline_sorted['total_co2e'].values, 
                            management_sorted['total_co2e'].values, 
                            rtol=1e-10, atol=1e-10
                        )
                
                # Plot baseline (net abatement from 0)
                if not baseline_data.empty:
                    label = f'{self._format_label("baseline")} (Net abatement)'
                    style = color_manager.get_scenario_style('baseline')
                    if curves_identical:
                        style['linewidth'] = 3
                        style['alpha'] = 0.8
                        style['linestyle'] = '-'
                    ax.plot(baseline_data['calendar_year'], baseline_data['total_co2e'], 
                           label=label, **style)
                
                # Plot management (net abatement from 0)
                if not management_data.empty:
                    label = f'{self._format_label("management")} (Net abatement)'
                    style = color_manager.get_scenario_style('management')
                    if curves_identical:
                        style['linewidth'] = 3
                        style['alpha'] = 0.8
                        style['linestyle'] = '--'
                        label += ' (OVERLAPPING)'
                    ax.plot(management_data['calendar_year'], management_data['total_co2e'], 
                           label=label, **style)
                
                # Plot reforestation (net abatement from 0)
                if not reforestation_data.empty:
                    label = f'{self._format_label("reforestation")} (Net abatement)'
                    style = color_manager.get_scenario_style('reforestation')
                    ax.plot(reforestation_data['calendar_year'], reforestation_data['total_co2e'], 
                           label=label, **style)
                
                # Add annotation if curves are identical
                if curves_identical:
                    ax.text(0.02, 0.98, '✓ BASELINE MANAGEMENT: Curves are identical\n(Red solid + Blue dashed = Perfect overlap)', 
                           transform=ax.transAxes, fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                           verticalalignment='top')
            else:
                # Regular format - plot by scenario columns
                for scenario in ['baseline', 'management', 'reforestation']:
                    if f'{scenario}_co2e' in df.columns:
                        label = f'{self._format_label(scenario)} (Net abatement)'
                        style = color_manager.get_scenario_style(scenario)
                        ax.plot(df['year'], df[f'{scenario}_co2e'], 
                               label=label, **style)
        
        # Customize the plot
        ax.set_xlabel('Year', fontsize=self.label_fontsize, fontweight='bold')
        ax.set_ylabel('Net carbon abatement (t CO2e/ha)', fontsize=self.label_fontsize, fontweight='bold')
        ax.set_title('Net carbon abatement - all scenarios (from baseline 0)', fontsize=self.title_fontsize, fontweight='bold', pad=20)
        
        # Add legend with better positioning
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=self.legend_fontsize)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add zero line for reference
        ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.6)
        
        # Add some styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
        # Set tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=self.tick_fontsize)
        
        # Set y-axis to start from 0 to show net abatement clearly
        ax.set_ylim(bottom=0)
        
        # Add annotation explaining the plot
        ax.text(0.02, 0.02, 'All curves show net carbon sequestration from baseline 0\nHigher curves = greater carbon benefits', 
               transform=ax.transAxes, fontsize=9, style='italic',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
               verticalalignment='bottom')
        
        plt.tight_layout()
        return self._save_figure('net_carbon_abatement')
    
    def plot_reforestation_minus_losses(self, results: Dict[str, pd.DataFrame], 
                                       forest_types: List[str]) -> str:
        """
        Plot reforestation curve minus baseline losses.
        Shows the true net benefit of reforestation accounting for existing forest degradation.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for forest_type in forest_types:
            if forest_type not in results:
                continue
                
            df = results[forest_type]
            
            # Calculate net carbon abatement (change from initial state)
            baseline_net = df['baseline_co2e'] - df['baseline_co2e'].iloc[0]
            reforestation_net = df['reforestation_co2e'] - df['reforestation_co2e'].iloc[0]
            
            # Calculate reforestation minus baseline losses
            # reforestation gains - baseline losses = 257 - 448 = -191
            # Use absolute value of baseline losses since they're negative
            reforestation_minus_losses = reforestation_net - abs(baseline_net)
            
            # Plot the curves
            ax.plot(df['year'], baseline_net, 
                   color='red', linewidth=2, linestyle='-', 
                   label=f'{forest_type} Baseline Losses (Carbon Lost)')
            ax.plot(df['year'], reforestation_net, 
                   color='green', linewidth=2, linestyle='--', 
                   label=f'{forest_type} Reforestation Gains (New Carbon)')
            ax.plot(df['year'], reforestation_minus_losses, 
                   color='darkgreen', linewidth=3, linestyle='-', 
                   label=f'{forest_type} Reforestation - Losses (Net Benefit)')
            
            # Add zero line for reference
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Year', fontsize=self.label_fontsize, fontweight='bold')
        ax.set_ylabel('Carbon (t CO₂e/ha)', fontsize=self.label_fontsize, fontweight='bold')
        ax.set_title('Reforestation Minus Baseline Losses\n(True Net Benefit of Reforestation)', 
                    fontsize=self.title_fontsize, fontweight='bold', pad=20)
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=self.legend_fontsize)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=self.tick_fontsize)
        
        # Formatting
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
        # Add annotation explaining the plot
        ax.text(0.02, 0.98, 'Shows true net benefit of reforestation\nRed = baseline losses, Green dashed = new gains, Dark green = net benefit', 
               transform=ax.transAxes, fontsize=10, style='italic',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
               verticalalignment='top')
        
        plt.tight_layout()
        return self._save_figure('reforestation_minus_losses')
    
    def plot_management_minus_reforestation(self, results: Dict[str, pd.DataFrame], 
                                           forest_types: List[str]) -> str:
        """
        Plot management benefits minus reforestation gains.
        Shows what happens when you manage existing forest but don't plant new trees.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for forest_type in forest_types:
            if forest_type not in results:
                continue
                
            df = results[forest_type]
            
            # Calculate net carbon abatement (change from initial state)
            baseline_net = df['baseline_co2e'] - df['baseline_co2e'].iloc[0]
            management_net = df['management_co2e'] - df['management_co2e'].iloc[0]
            reforestation_net = df['reforestation_co2e'] - df['reforestation_co2e'].iloc[0]
            
            # Calculate management benefits (carbon kept vs baseline)
            management_benefits = management_net - baseline_net
            
            # Calculate management minus reforestation gains
            # management benefits - reforestation gains = 331 - 257 = +74
            management_minus_reforestation = management_benefits - reforestation_net
            
            # Plot the curves
            ax.plot(df['year'], management_benefits, 
                   color='blue', linewidth=2, linestyle='-', 
                   label=f'{forest_type} Management Benefits (Carbon Kept)')
            ax.plot(df['year'], reforestation_net, 
                   color='green', linewidth=2, linestyle='--', 
                   label=f'{forest_type} Reforestation Gains (New Carbon)')
            ax.plot(df['year'], management_minus_reforestation, 
                   color='darkblue', linewidth=3, linestyle='-', 
                   label=f'{forest_type} Management - Reforestation (Net Benefit)')
            
            # Add zero line for reference
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Year', fontsize=self.label_fontsize, fontweight='bold')
        ax.set_ylabel('Carbon (t CO₂e/ha)', fontsize=self.label_fontsize, fontweight='bold')
        ax.set_title('Management Benefits Minus Reforestation Gains\n(Managing Existing Forest Without Planting New Trees)', 
                    fontsize=self.title_fontsize, fontweight='bold', pad=20)
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=self.legend_fontsize)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=self.tick_fontsize)
        
        # Formatting
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
        # Add annotation explaining the plot
        ax.text(0.02, 0.98, 'Shows net benefit of managing existing forest without reforestation\nBlue = carbon kept, Green dashed = new gains, Dark blue = net benefit', 
               transform=ax.transAxes, fontsize=10, style='italic',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
               verticalalignment='top')
        
        plt.tight_layout()
        return self._save_figure('management_minus_reforestation')
    
    def plot_project_level_additionality(self, results: Dict[str, pd.DataFrame], 
                                       forest_types: List[str]):
        """
        Plot project-level carbon additionality for individual investment decisions.
        
        Each intervention evaluated independently:
        - Management: Management CO2e - Baseline CO2e (additional carbon from improved management)
        - Reforestation: Reforestation CO2e - 0 (additional carbon from new planting)
        
        Used for: "How much additional carbon does each project type provide independently?"
        
        Args:
            results: Dictionary of results DataFrames
            forest_types: List of forest types
        """
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        for forest_type in forest_types:
            df = results[forest_type]
            
            # Check data format and calculate additionality accordingly
            if 'scenario' in df.columns:
                # Calendar years format - calculate additionality by scenario
                baseline_data = df[df['scenario'] == 'baseline']
                management_data = df[df['scenario'] == 'management']
                reforestation_data = df[df['scenario'] == 'reforestation']
                
                if not baseline_data.empty and not management_data.empty:
                    # Calculate additionality for management (vs baseline)
                    # Ensure data is aligned by year
                    baseline_sorted = baseline_data.sort_values('year')
                    management_sorted = management_data.sort_values('year')
                    management_additionality = management_sorted['total_co2e'].values - baseline_sorted['total_co2e'].values
                    mgmt_style = color_manager.get_scenario_style('management')
                    ax.plot(management_sorted['calendar_year'], management_additionality, 
                           label=f'{forest_type} - Management vs Degraded Forest',
                           **mgmt_style)
                
                if not reforestation_data.empty:
                    # Calculate additionality for reforestation (vs 0 - new planting)
                    # Ensure data is aligned by year
                    reforestation_sorted = reforestation_data.sort_values('year')
                    reforestation_additionality = reforestation_sorted['total_co2e'].values - 0  # vs 0 (new planting)
                    refor_style = color_manager.get_scenario_style('reforestation')
                    ax.plot(reforestation_sorted['calendar_year'], reforestation_additionality,
                           label=f'{forest_type} - Reforestation vs Bare Land',
                           **refor_style)
            else:
                # Regular format - calculate additionality by scenario columns
                if 'management_co2e' in df.columns and 'baseline_co2e' in df.columns:
                    management_additionality = df['management_co2e'] - df['baseline_co2e']
                    mgmt_style = color_manager.get_scenario_style('management')
                    ax.plot(df['year'], management_additionality, 
                           label=f'{forest_type} - Management vs Degraded Forest',
                           **mgmt_style)
                
                if 'reforestation_co2e' in df.columns:
                    reforestation_additionality = df['reforestation_co2e'] - 0  # vs 0 (new planting)
                    refor_style = color_manager.get_scenario_style('reforestation')
                    ax.plot(df['year'], reforestation_additionality,
                           label=f'{forest_type} - Reforestation vs Bare Land',
                           **refor_style)
        
        # Customize the plot
        ax.set_xlabel('Year', fontsize=self.label_fontsize, fontweight='bold')
        ax.set_ylabel('Carbon additionality (t CO2e/ha)', fontsize=self.label_fontsize, fontweight='bold')
        ax.set_title('Carbon Additionality', fontsize=self.title_fontsize, fontweight='bold', pad=20)
        
        
        # Add legend with better positioning
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=self.legend_fontsize)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add zero line for reference
        ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.6)
        
        # Add some styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
        # Set tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=self.tick_fontsize)
        
        # Set axis limits for better visualization
        # Don't force x-axis to start at 0 for calendar years
        
        plt.tight_layout()
        return self._save_figure('additionality')
    
    def plot_combined_additionality_and_stocks(self, results: Dict[str, pd.DataFrame], 
                                             forest_types: List[str]):
        """
        Create a combined plot showing carbon stocks as main chart with additionality as inset.
        Main chart shows total carbon stocks, inset in bottom left shows additionality.
        
        Args:
            results: Dictionary of results DataFrames
            forest_types: List of forest types
        """
        fig, ax_main = plt.subplots(figsize=(12, 8))
        
        # Create inset axes in bottom right corner
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_inset = inset_axes(ax_main, width="30%", height="25%", loc='lower right')
        
        for forest_type in forest_types:
            if forest_type not in results:
                continue
                
            df = results[forest_type]
            
            # Check data format and plot accordingly
            if 'scenario' in df.columns:
                # Calendar years format - plot by scenario
                baseline_data = df[df['scenario'] == 'baseline']
                management_data = df[df['scenario'] == 'management']
                reforestation_data = df[df['scenario'] == 'reforestation']
                
                # Plot total carbon stocks on main chart
                if not baseline_data.empty:
                    baseline_sorted = baseline_data.sort_values('year')
                    baseline_style = color_manager.get_scenario_style('baseline')
                    baseline_style['linestyle'] = '--'
                    baseline_style['alpha'] = 0.7
                    ax_main.plot(baseline_sorted['calendar_year'], baseline_sorted['total_co2e'], 
                            label='Baseline',
                            **baseline_style)
                
                if not management_data.empty:
                    management_sorted = management_data.sort_values('year')
                    mgmt_style = color_manager.get_scenario_style('management')
                    mgmt_style['linestyle'] = '-'
                    ax_main.plot(management_sorted['calendar_year'], management_sorted['total_co2e'], 
                            label='AFM',
                            **mgmt_style)
                
                if not reforestation_data.empty:
                    reforestation_sorted = reforestation_data.sort_values('year')
                    refor_style = color_manager.get_scenario_style('reforestation')
                    refor_style['linestyle'] = '-'
                    ax_main.plot(reforestation_sorted['calendar_year'], reforestation_sorted['total_co2e'], 
                            label='Reforestation',
                            **refor_style)
                
                # Calculate and plot additionality on inset chart
                if not baseline_data.empty and not management_data.empty:
                    baseline_sorted = baseline_data.sort_values('year')
                    management_sorted = management_data.sort_values('year')
                    management_additionality = management_sorted['total_co2e'].values - baseline_sorted['total_co2e'].values
                    mgmt_add_style = color_manager.get_scenario_style('management')
                    mgmt_add_style['linestyle'] = '-'
                    mgmt_add_style['linewidth'] = 2
                    ax_inset.plot(management_sorted['calendar_year'], management_additionality, 
                            **mgmt_add_style)
                
                if not reforestation_data.empty:
                    reforestation_sorted = reforestation_data.sort_values('year')
                    reforestation_additionality = reforestation_sorted['total_co2e'].values - 0
                    refor_add_style = color_manager.get_scenario_style('reforestation')
                    refor_add_style['linestyle'] = '-'
                    refor_add_style['linewidth'] = 2
                    ax_inset.plot(reforestation_sorted['calendar_year'], reforestation_additionality,
                            **refor_add_style)
            else:
                # Regular format - plot by scenario columns
                # Plot total carbon stocks on main chart
                baseline_style = color_manager.get_scenario_style('baseline')
                baseline_style['linestyle'] = '--'
                baseline_style['alpha'] = 0.7
                ax_main.plot(df['year'], df['baseline_co2e'], 
                        label='Baseline',
                        **baseline_style)
                
                mgmt_style = color_manager.get_scenario_style('management')
                mgmt_style['linestyle'] = '-'
                ax_main.plot(df['year'], df['management_co2e'], 
                        label='AFM',
                        **mgmt_style)
                
                refor_style = color_manager.get_scenario_style('reforestation')
                refor_style['linestyle'] = '-'
                ax_main.plot(df['year'], df['reforestation_co2e'], 
                        label='Reforestation',
                        **refor_style)
                
                # Calculate and plot additionality on inset chart
                if 'management_co2e' in df.columns and 'baseline_co2e' in df.columns:
                    management_additionality = df['management_co2e'] - df['baseline_co2e']
                    mgmt_add_style = color_manager.get_scenario_style('management')
                    mgmt_add_style['linestyle'] = '-'
                    mgmt_add_style['linewidth'] = 2
                    ax_inset.plot(df['year'], management_additionality, 
                            **mgmt_add_style)
                
                if 'reforestation_co2e' in df.columns:
                    reforestation_additionality = df['reforestation_co2e'] - 0
                    refor_add_style = color_manager.get_scenario_style('reforestation')
                    refor_add_style['linestyle'] = '-'
                    refor_add_style['linewidth'] = 2
                    ax_inset.plot(df['year'], reforestation_additionality,
                            **refor_add_style)
        
        # Customize the main plot
        ax_main.set_xlabel('Year', fontsize=28)
        ax_main.set_ylabel('Total carbon stock (t CO2e/ha)', fontsize=28)
        ax_main.set_title('Carbon stocks over time', fontsize=36, pad=20)
        
        # Add legend back to main chart with clean white background
        legend = ax_main.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                               ncol=3, frameon=False, fontsize=20)
        
        # Remove grid from main chart for clean background
        ax_main.grid(False)
        
        # Styling for main chart
        ax_main.spines['top'].set_visible(False)
        ax_main.spines['right'].set_visible(False)
        ax_main.spines['left'].set_linewidth(1.2)
        ax_main.spines['bottom'].set_linewidth(1.2)
        
        # Increase tick label sizes
        ax_main.tick_params(axis='both', which='major', labelsize=24)
        
        # Customize the inset plot (additionality) - completely clean
        ax_inset.set_title('Additionality', fontsize=24, pad=5)
        ax_inset.grid(True, alpha=0.3, linestyle=':')
        ax_inset.axhline(y=0, color='black', linewidth=0.5, alpha=0.6)
        
        # Remove all axis labels and numbering for clean look
        ax_inset.set_xlabel('')
        ax_inset.set_ylabel('')
        ax_inset.set_xticks([])  # Remove x-axis ticks
        ax_inset.set_yticks([])  # Remove y-axis ticks
        ax_inset.tick_params(axis='both', which='both', length=0)  # Remove tick marks
        
        # Styling for inset - minimal
        ax_inset.spines['top'].set_visible(False)
        ax_inset.spines['right'].set_visible(False)
        ax_inset.spines['left'].set_linewidth(0.8)
        ax_inset.spines['bottom'].set_linewidth(0.8)
        
        plt.tight_layout()
        return self._save_figure('combined_additionality_stocks')
    
    def plot_poster_stocks_only(self, results: Dict[str, pd.DataFrame], 
                                forest_types: List[str]):
        """
        Create a poster-style plot showing only total carbon stocks over time
        (no additionality inset).
        """
        fig, ax_main = plt.subplots(figsize=(12, 8))
        
        for forest_type in forest_types:
            if forest_type not in results:
                continue
            
            df = results[forest_type]
            
            # Plot by available format
            if 'scenario' in df.columns:
                baseline_data = df[df['scenario'] == 'baseline']
                management_data = df[df['scenario'] == 'management']
                reforestation_data = df[df['scenario'] == 'reforestation']
                
                if not baseline_data.empty:
                    baseline_sorted = baseline_data.sort_values('year')
                    baseline_style = color_manager.get_scenario_style('baseline')
                    baseline_style['linestyle'] = '--'
                    baseline_style['alpha'] = 0.7
                    ax_main.plot(baseline_sorted['calendar_year'], baseline_sorted['total_co2e'], 
                                 label='Baseline', **baseline_style)
                
                if not management_data.empty:
                    management_sorted = management_data.sort_values('year')
                    mgmt_style = color_manager.get_scenario_style('management')
                    mgmt_style['linestyle'] = '-'
                    ax_main.plot(management_sorted['calendar_year'], management_sorted['total_co2e'], 
                                 label='AFM', **mgmt_style)
                
                if not reforestation_data.empty:
                    reforestation_sorted = reforestation_data.sort_values('year')
                    refor_style = color_manager.get_scenario_style('reforestation')
                    refor_style['linestyle'] = '-'
                    ax_main.plot(reforestation_sorted['calendar_year'], reforestation_sorted['total_co2e'], 
                                 label='Reforestation', **refor_style)
            else:
                baseline_style = color_manager.get_scenario_style('baseline')
                baseline_style['linestyle'] = '--'
                baseline_style['alpha'] = 0.7
                ax_main.plot(df['year'], df['baseline_co2e'], label='Baseline', **baseline_style)
                
                mgmt_style = color_manager.get_scenario_style('management')
                mgmt_style['linestyle'] = '-'
                ax_main.plot(df['year'], df['management_co2e'], label='AFM', **mgmt_style)
                
                refor_style = color_manager.get_scenario_style('reforestation')
                refor_style['linestyle'] = '-'
                ax_main.plot(df['year'], df['reforestation_co2e'], label='Reforestation', **refor_style)
        
        # Main chart styling (poster-like)
        ax_main.set_xlabel('Year', fontsize=28)
        ax_main.set_ylabel('Total carbon stock (t CO2e/ha)', fontsize=28)
        ax_main.set_title('Carbon stocks over time', fontsize=36, pad=20)
        ax_main.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                       ncol=3, frameon=False, fontsize=20)
        ax_main.grid(False)
        ax_main.spines['top'].set_visible(False)
        ax_main.spines['right'].set_visible(False)
        ax_main.spines['left'].set_linewidth(1.2)
        ax_main.spines['bottom'].set_linewidth(1.2)
        ax_main.tick_params(axis='both', which='major', labelsize=24)
        
        plt.tight_layout()
        return self._save_figure('poster_stocks_only')

    def _create_legend_file(self):
        """Create a separate legend file for the combined plot."""
        fig, ax = plt.subplots(figsize=(8, 2))
        
        # Create dummy lines with the same styles as the main plot
        baseline_style = color_manager.get_scenario_style('baseline')
        baseline_style['linestyle'] = '--'
        baseline_style['alpha'] = 0.7
        
        mgmt_style = color_manager.get_scenario_style('management')
        mgmt_style['linestyle'] = '-'
        
        refor_style = color_manager.get_scenario_style('reforestation')
        refor_style['linestyle'] = '-'
        
        # Create dummy data points for the legend
        x_dummy = [0, 1]
        ax.plot(x_dummy, [0, 0], label='Baseline', **baseline_style)
        ax.plot(x_dummy, [0, 0], label='AFM', **mgmt_style)
        ax.plot(x_dummy, [0, 0], label='Reforestation', **refor_style)
        
        # Hide the plot area, just show legend
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Create legend with clean white background, no box or shading
        legend = ax.legend(loc='center', ncol=3, frameon=False, fontsize=24)
        
        # Remove all padding and margins
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.margins(0)
        
        # Save legend file
        legend_files = {}
        for fmt in ['png']:  # Only PNG for legend
            filename = f'legend_combined_additionality_stocks.{fmt}'
            filepath = self.plots_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight', pad_inches=0)
            legend_files[fmt] = filepath
        
        plt.close(fig)
        return legend_files