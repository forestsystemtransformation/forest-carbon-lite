#!/usr/bin/env python3
"""
Simple uncertainty analysis for forest growth and carbon sequestration.

This module provides basic uncertainty quantification focusing on biological
parameters that affect growth and carbon outcomes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json


class GrowthCarbonUncertainty:
    """Simple uncertainty analysis for growth and carbon parameters."""
    
    def __init__(self, output_dir: Path, scenario_name: str = None, seed: int = 42):
        self.output_dir = Path(output_dir)
        self.scenario_name = scenario_name
        
        # Create uncertainty analysis directory - either scenario-specific or main level
        if scenario_name:
            self.uncertainty_dir = self.output_dir / scenario_name / "uncertainty_analysis"
        else:
            self.uncertainty_dir = self.output_dir / "uncertainty_analysis"
        
        self.uncertainty_dir.mkdir(parents=True, exist_ok=True)
        
        # Use numpy's modern RNG for better reproducibility
        self.rng = np.random.default_rng(seed)
        
    def run_uncertainty_analysis(self, 
                               base_results: Dict[str, Any],
                               n_runs: int = 500) -> Dict[str, Any]:
        """
        Run Monte Carlo uncertainty analysis on growth and carbon parameters.
        
        Args:
            base_results: Results from base simulation
            n_runs: Number of Monte Carlo iterations
            
        Returns:
            Dictionary containing uncertainty analysis results
        """
        print(f"Running uncertainty analysis with {n_runs} iterations...")
        
        # Define parameter distributions based on literature/experience
        param_distributions = self._get_parameter_distributions()
        
        # Run Monte Carlo simulations
        mc_results = []
        for i in range(n_runs):
            if (i + 1) % 50 == 0:
                print(f"  Completed {i + 1}/{n_runs} iterations")
            
            # Sample parameters
            sampled_params = self._sample_parameters(param_distributions)
            
            # Calculate modified results (simplified approach)
            modified_results = self._apply_parameter_effects(base_results, sampled_params)
            modified_results['run_id'] = i
            modified_results['parameters'] = sampled_params
            
            mc_results.append(modified_results)
        
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(mc_results)
        
        # Calculate uncertainty statistics
        uncertainty_stats = self._calculate_uncertainty_statistics(results_df)
        
        # Generate visualizations
        self._create_uncertainty_plots(results_df, uncertainty_stats)
        
        # Generate report
        self._generate_uncertainty_report(uncertainty_stats, results_df)
        
        # Save results
        results_df.to_csv(self.uncertainty_dir / 'monte_carlo_results.csv', index=False)
        
        print(f"Uncertainty analysis completed. Results saved to: {self.uncertainty_dir}")
        
        return {
            'results_df': results_df,
            'uncertainty_stats': uncertainty_stats,
            'output_dir': self.uncertainty_dir
        }
    
    def _get_parameter_distributions(self) -> Dict[str, Dict[str, float]]:
        """Define parameter uncertainty distributions."""
        return {
            # Mortality rates (beta distribution, bounded 0-1)
            'mortality_degraded': {'type': 'beta', 'alpha': 2.0, 'beta': 48.0, 'mean': 0.04},
            'mortality_managed': {'type': 'beta', 'alpha': 1.5, 'beta': 48.5, 'mean': 0.03},
            'mortality_reforestation': {'type': 'beta', 'alpha': 1.0, 'beta': 49.0, 'mean': 0.02},
            
            # Disturbance probability (beta distribution)
            'disturbance_prob': {'type': 'beta', 'alpha': 3.0, 'beta': 47.0, 'mean': 0.06},
            
            # Disturbance severity (beta distribution)
            'disturbance_severity': {'type': 'beta', 'alpha': 2.0, 'beta': 3.0, 'mean': 0.4},
            
            # TYF parameters (normal distributions)
            'M_max_biomass': {'type': 'normal', 'mean': 1.0, 'std': 0.15},  # Relative to base
            'G_age_max_growth': {'type': 'normal', 'mean': 1.0, 'std': 0.1},
            'y_growth_multiplier': {'type': 'normal', 'mean': 1.0, 'std': 0.2},
            
            # FPI (Forest Productivity Index) - normal distribution
            'fpi_multiplier': {'type': 'normal', 'mean': 1.0, 'std': 0.15}
        }
    
    def _sample_parameters(self, distributions: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Sample parameters from their distributions."""
        sampled = {}
        
        for param, dist in distributions.items():
            if dist['type'] == 'beta':
                # Beta distribution
                alpha = dist['alpha']
                beta = dist['beta']
                sampled[param] = self.rng.beta(alpha, beta)
                
            elif dist['type'] == 'normal':
                # Normal distribution
                mean = dist['mean']
                std = dist['std']
                sampled[param] = self.rng.normal(mean, std)
                
            else:
                # Default to uniform
                sampled[param] = self.rng.uniform(0.8, 1.2)
        
        return sampled
    
    def _apply_parameter_effects(self, 
                                base_results: Dict[str, Any], 
                                parameters: Dict[str, float]) -> Dict[str, Any]:
        """Apply parameter effects to base results (simplified approach)."""
        
        # Extract base values
        base_agb = base_results.get('total_agb', 100.0)  # t/ha
        base_co2e = base_results.get('final_co2e_stock', 200.0)  # tCO2e/ha
        base_annual_increment = base_results.get('mean_annual_increment', 8.0)  # tCO2e/ha/yr
        
        # Calculate combined effects
        growth_effect = (parameters['M_max_biomass'] * 
                        parameters['G_age_max_growth'] * 
                        parameters['y_growth_multiplier'] * 
                        parameters['fpi_multiplier'])
        
        mortality_effect = (1.0 - parameters['mortality_degraded'] * 10.0 -  # Scale mortality impact
                           parameters['mortality_managed'] * 8.0 -
                           parameters['mortality_reforestation'] * 6.0)
        
        disturbance_effect = (1.0 - parameters['disturbance_prob'] * parameters['disturbance_severity'] * 0.5)
        
        # Apply effects
        modified_agb = base_agb * growth_effect * mortality_effect * disturbance_effect
        modified_co2e = base_co2e * growth_effect * mortality_effect * disturbance_effect
        modified_increment = base_annual_increment * growth_effect * mortality_effect * disturbance_effect
        
        return {
            'total_agb': modified_agb,
            'final_co2e_stock': modified_co2e,
            'mean_annual_increment': modified_increment,
            'growth_effect': growth_effect,
            'mortality_effect': mortality_effect,
            'disturbance_effect': disturbance_effect
        }
    
    def _calculate_uncertainty_statistics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate uncertainty statistics for key metrics."""
        
        metrics = ['total_agb', 'final_co2e_stock', 'mean_annual_increment']
        confidence_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
        
        stats = {}
        
        for metric in metrics:
            values = results_df[metric]
            percentiles = np.percentile(values, [p*100 for p in confidence_levels])
            
            stats[metric] = {
                'mean': values.mean(),
                'std': values.std(),
                'p5': percentiles[0],
                'p25': percentiles[1],
                'p50': percentiles[2],
                'p75': percentiles[3],
                'p95': percentiles[4],
                'uncertainty_range': percentiles[4] - percentiles[0],
                'uncertainty_pct': (percentiles[4] - percentiles[0]) / 2 / values.mean() * 100,
                'cv': values.std() / values.mean() * 100  # Coefficient of variation
            }
        
        # Calculate parameter correlations
        param_cols = [col for col in results_df.columns if col.startswith(('mortality_', 'disturbance_', 'M_', 'G_', 'y_', 'fpi_'))]
        correlations = {}
        
        for metric in metrics:
            correlations[metric] = {}
            for param in param_cols:
                if param in results_df.columns:
                    corr = results_df[metric].corr(results_df[param])
                    correlations[metric][param] = corr
        
        stats['correlations'] = correlations
        
        return stats
    
    def _create_uncertainty_plots(self, 
                                results_df: pd.DataFrame, 
                                stats: Dict[str, Any]) -> None:
        """Create uncertainty visualization plots."""
        
        # Set up the plotting style
        plt.style.use('default')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        
        # 1. Distribution plots for key metrics
        self._plot_metric_distributions(results_df, stats)
        
        # 2. Parameter sensitivity plots
        self._plot_parameter_sensitivity(results_df)
        
        # 3. Uncertainty bands over time (simulated)
        self._plot_uncertainty_bands_over_time(results_df, stats)
        
        # 4. Parameter importance ranking
        self._plot_parameter_importance(stats)
        
        # 5. TYF parameter correlation matrix
        self._plot_tyf_parameter_correlation_matrix(results_df)
        
        # 6. Fan charts for uncertainty visualization
        self._plot_fan_charts(results_df, stats)
        
        # 7. 3x3 dashboard for total carbon additionality
        self._plot_carbon_additionality_dashboard(results_df, stats)
        
        # Note: Sequestration bands plot now shows all scenarios in one plot
    
    def _plot_metric_distributions(self, results_df: pd.DataFrame, stats: Dict[str, Any]) -> None:
        """Plot distribution of total CO2e stock."""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric = 'final_co2e_stock'
        title = 'Total CO2e Stock (tCO2e/ha)'
        
        # Histogram
        ax.hist(results_df[metric], bins=50, alpha=0.7, color='skyblue', density=True, edgecolor='black', linewidth=0.5)
        
        # Add confidence intervals
        stat = stats[metric]
        ax.axvline(stat['p5'], color='red', linestyle='--', alpha=0.7, label='5th percentile')
        ax.axvline(stat['p95'], color='red', linestyle='--', alpha=0.7, label='95th percentile')
        ax.axvline(stat['mean'], color='darkblue', linestyle='-', linewidth=2, label='Mean')
        ax.axvline(stat['p50'], color='orange', linestyle='-', linewidth=2, label='Median')
        
        ax.set_xlabel(title)
        ax.set_ylabel('Density')
        ax.set_title(f'{title} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add uncertainty statistics to the plot
        uncertainty_text = f'Mean: {stat["mean"]:.1f} tCO2e/ha\n'
        uncertainty_text += f'90% CI: {stat["p5"]:.1f} - {stat["p95"]:.1f} tCO2e/ha\n'
        uncertainty_text += f'Uncertainty: ±{stat["uncertainty_pct"]:.1f}%'
        ax.text(0.02, 0.98, uncertainty_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.uncertainty_dir / 'metric_distributions.png', bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_sensitivity(self, results_df: pd.DataFrame) -> None:
        """Plot parameter sensitivity analysis for CO2e stock."""
        
        # Get parameter columns
        param_cols = [col for col in results_df.columns if col.startswith(('mortality_', 'disturbance_', 'M_', 'G_', 'y_', 'fpi_'))]
        metric = 'final_co2e_stock'
        
        # Create subplots - arrange in a grid
        n_params = len(param_cols)
        n_cols = min(3, n_params)  # Max 3 columns
        n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Handle single subplot case
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1 and n_cols > 1:
            axes = axes.reshape(1, -1)
        elif n_rows > 1:
            axes = axes.flatten()
        
        for i, param in enumerate(param_cols):
            if n_params == 1:
                ax = axes[0]
            elif n_rows == 1 and n_cols > 1:
                ax = axes[0, i]
            else:
                ax = axes[i]
            
            # Scatter plot
            ax.scatter(results_df[param], results_df[metric], alpha=0.6, s=20, color='purple')
            
            # Calculate correlation
            corr = results_df[param].corr(results_df[metric])
            
            ax.set_xlabel(param.replace('_', ' ').title())
            ax.set_ylabel('Total CO2e Stock (tCO2e/ha)')
            ax.set_title(f'{param.replace("_", " ").title()}\nr = {corr:.3f}')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if abs(corr) > 0.1:  # Only show trend line if correlation is meaningful
                z = np.polyfit(results_df[param], results_df[metric], 1)
                p = np.poly1d(z)
                ax.plot(results_df[param], p(results_df[param]), "r--", alpha=0.8, linewidth=2)
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.uncertainty_dir / 'parameter_sensitivity.png', bbox_inches='tight')
        plt.close()
    
    def _plot_uncertainty_bands_over_time(self, results_df: pd.DataFrame, stats: Dict[str, Any]) -> None:
        """Plot uncertainty bands over time for all scenarios in a 1x3 layout."""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        years = np.arange(0, 31)
        
        # Define scenario-specific parameters
        scenario_params = {
            'baseline': {'color': 'blue', 'label': 'Baseline', 'base_curve': 100, 'growth_rate': 0.08},
            'management': {'color': 'green', 'label': 'Management', 'base_curve': 120, 'growth_rate': 0.09},
            'reforestation': {'color': 'orange', 'label': 'Reforestation', 'base_curve': 150, 'growth_rate': 0.10}
        }
        
        scenarios = ['baseline', 'management', 'reforestation']
        
        for i, scenario in enumerate(scenarios):
            ax = axes[i]
            params = scenario_params[scenario]
            
            # Create base growth curve (logistic)
            base_curve = params['base_curve'] * (1 - np.exp(-params['growth_rate'] * years))
            
            # Add uncertainty based on our analysis
            uncertainty_factor = stats['final_co2e_stock']['uncertainty_pct'] / 100
            
            # Create multiple uncertainty curves for better visualization
            n_curves = 20
            uncertainty_curves = []
            
            for j in range(n_curves):
                # Generate random uncertainty for this curve
                uncertainty = self.rng.normal(0, uncertainty_factor, len(years)) * base_curve
                curve = base_curve + uncertainty
                uncertainty_curves.append(curve)
            
            # Convert to numpy array for easier calculation
            uncertainty_curves = np.array(uncertainty_curves)
            
            # Calculate percentiles for uncertainty bands
            p5_curve = np.percentile(uncertainty_curves, 5, axis=0)
            p25_curve = np.percentile(uncertainty_curves, 25, axis=0)
            p75_curve = np.percentile(uncertainty_curves, 75, axis=0)
            p95_curve = np.percentile(uncertainty_curves, 95, axis=0)
            
            # Plot uncertainty bands
            ax.fill_between(years, p5_curve, p95_curve, alpha=0.2, color=params['color'], label='90% CI')
            ax.fill_between(years, p25_curve, p75_curve, alpha=0.4, color=params['color'], label='50% CI')
            
            # Plot median curve
            ax.plot(years, base_curve, color=params['color'], linewidth=3, label=f'{params["label"]} (median)')
            
            # Add some individual uncertainty curves for context
            for j in range(0, n_curves, 4):  # Show every 4th curve
                ax.plot(years, uncertainty_curves[j], color=params['color'], alpha=0.1, linewidth=0.5)
            
            ax.set_xlabel('Year')
            ax.set_ylabel('Cumulative CO2e Sequestration (tCO2e/ha)')
            ax.set_title(f'{params["label"]} Scenario')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add uncertainty statistics to the plot
            uncertainty_text = f'Uncertainty: ±{stats["final_co2e_stock"]["uncertainty_pct"]:.1f}%\n'
            uncertainty_text += f'90% CI: {stats["final_co2e_stock"]["p5"]:.1f} - {stats["final_co2e_stock"]["p95"]:.1f} tCO2e/ha'
            ax.text(0.02, 0.98, uncertainty_text, transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('Carbon Sequestration Uncertainty Bands - All Scenarios', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(self.uncertainty_dir / 'sequestration_uncertainty_bands.png', bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_importance(self, stats: Dict[str, Any]) -> None:
        """Plot parameter importance ranking."""
        
        # Calculate average absolute correlation across metrics
        correlations = stats['correlations']
        param_importance = {}
        
        for param in correlations['final_co2e_stock'].keys():
            avg_abs_corr = np.mean([abs(correlations[metric][param]) for metric in correlations.keys()])
            param_importance[param] = avg_abs_corr
        
        # Sort by importance
        sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        params = [p[0].replace('_', ' ').title() for p in sorted_params]
        importance = [p[1] for p in sorted_params]
        
        bars = ax.barh(params, importance, color='skyblue', edgecolor='navy', linewidth=0.5)
        
        # Color bars by importance level
        for i, bar in enumerate(bars):
            if importance[i] > 0.3:
                bar.set_color('red')
            elif importance[i] > 0.2:
                bar.set_color('orange')
            else:
                bar.set_color('lightblue')
        
        ax.set_xlabel('Average Absolute Correlation')
        ax.set_title('Parameter Importance for Carbon Sequestration')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(importance):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.uncertainty_dir / 'parameter_importance.png', bbox_inches='tight')
        plt.close()
    
    def _plot_tyf_parameter_correlation_matrix(self, results_df: pd.DataFrame) -> None:
        """Plot correlation matrix for TYF parameters."""
        
        # Extract TYF parameters from the parameters column
        tyf_data = []
        tyf_param_names = ['M_max_biomass', 'G_age_max_growth', 'y_growth_multiplier']
        
        for idx, row in results_df.iterrows():
            # Handle parameters - could be string or dict
            try:
                params_value = row['parameters']
                if isinstance(params_value, str):
                    # Parse string representation
                    params_dict = eval(params_value)
                else:
                    # Already a dict
                    params_dict = params_value
                
                tyf_values = [params_dict.get(param, np.nan) for param in tyf_param_names]
                tyf_data.append(tyf_values)
            except Exception as e:
                print(f"Debug: Error parsing row {idx}: {e}")
                tyf_data.append([np.nan] * len(tyf_param_names))
        
        # Create DataFrame with TYF parameters
        tyf_df = pd.DataFrame(tyf_data, columns=tyf_param_names)
        
        # Remove any rows with NaN values
        tyf_df = tyf_df.dropna()
        
        if len(tyf_df) < 2:
            print("Warning: Not enough valid TYF parameter data found for correlation matrix")
            return
        
        # Create correlation matrix
        correlation_matrix = tyf_df.corr()
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(tyf_param_names)))
        ax.set_yticks(range(len(tyf_param_names)))
        ax.set_xticklabels([param.replace('_', ' ').title() for param in tyf_param_names], rotation=45, ha='right')
        ax.set_yticklabels([param.replace('_', ' ').title() for param in tyf_param_names])
        
        # Add correlation values as text
        for i in range(len(tyf_param_names)):
            for j in range(len(tyf_param_names)):
                text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
        
        # Set title
        ax.set_title('TYF Parameter Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        
        # Add grid
        ax.set_xticks(np.arange(len(tyf_param_names)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(tyf_param_names)) - 0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(self.uncertainty_dir / 'tyf_parameter_correlation_matrix.png', bbox_inches='tight')
        plt.close()
    
    def _plot_fan_charts(self, results_df: pd.DataFrame, stats: Dict[str, Any]) -> None:
        """Create fan charts showing uncertainty over time for all scenarios."""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        years = np.arange(0, 31)
        
        # Define scenario-specific parameters
        scenario_params = {
            'baseline': {'color': 'blue', 'label': 'Baseline', 'base_curve': 100, 'growth_rate': 0.08},
            'management': {'color': 'green', 'label': 'Management', 'base_curve': 120, 'growth_rate': 0.09},
            'reforestation': {'color': 'orange', 'label': 'Reforestation', 'base_curve': 150, 'growth_rate': 0.10}
        }
        
        scenarios = ['baseline', 'management', 'reforestation']
        
        for i, scenario in enumerate(scenarios):
            ax = axes[i]
            params = scenario_params[scenario]
            
            # Create base growth curve (logistic)
            base_curve = params['base_curve'] * (1 - np.exp(-params['growth_rate'] * years))
            
            # Get uncertainty from our analysis
            uncertainty_factor = stats['final_co2e_stock']['uncertainty_pct'] / 100
            
            # Create multiple uncertainty curves for fan chart
            n_curves = 100
            uncertainty_curves = []
            
            for j in range(n_curves):
                # Generate random uncertainty for this curve
                uncertainty = self.rng.normal(0, uncertainty_factor, len(years)) * base_curve
                curve = base_curve + uncertainty
                uncertainty_curves.append(curve)
            
            # Convert to numpy array for easier calculation
            uncertainty_curves = np.array(uncertainty_curves)
            
            # Calculate percentiles for fan chart
            p05 = np.percentile(uncertainty_curves, 5, axis=0)
            p25 = np.percentile(uncertainty_curves, 25, axis=0)
            p50 = np.percentile(uncertainty_curves, 50, axis=0)
            p75 = np.percentile(uncertainty_curves, 75, axis=0)
            p95 = np.percentile(uncertainty_curves, 95, axis=0)
            
            # Create fan chart with multiple confidence levels
            # 90% confidence interval (5th-95th percentile)
            ax.fill_between(years, p05, p95, alpha=0.15, color=params['color'], 
                           label='90% CI (5th-95th percentile)')
            
            # 50% confidence interval (25th-75th percentile)
            ax.fill_between(years, p25, p75, alpha=0.3, color=params['color'], 
                           label='50% CI (25th-75th percentile)')
            
            # Median line
            ax.plot(years, p50, color=params['color'], linewidth=3, 
                   label=f'{params["label"]} (median)')
            
            # Base curve for reference
            ax.plot(years, base_curve, color=params['color'], linestyle='--', 
                   alpha=0.7, linewidth=1, label='Base curve')
            
            ax.set_xlabel('Year')
            ax.set_ylabel('Cumulative CO2e Sequestration (tCO2e/ha)')
            ax.set_title(f'{params["label"]} Scenario - Fan Chart')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add uncertainty statistics to the plot
            uncertainty_text = f'Uncertainty: ±{stats["final_co2e_stock"]["uncertainty_pct"]:.1f}%\n'
            uncertainty_text += f'90% CI: {stats["final_co2e_stock"]["p5"]:.1f} - {stats["final_co2e_stock"]["p95"]:.1f} tCO2e/ha'
            ax.text(0.02, 0.98, uncertainty_text, transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('Carbon Sequestration Fan Charts - Uncertainty Over Time', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(self.uncertainty_dir / 'fan_charts.png', bbox_inches='tight')
        plt.close()
    
    def _plot_carbon_additionality_dashboard(self, results_df: pd.DataFrame, stats: Dict[str, Any]) -> None:
        """Create 3x3 dashboard for total carbon additionality on project level."""
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        years = np.arange(0, 31)
        
        # Load actual simulation results
        simulation_results = self._load_simulation_results()
        
        # Define scenario parameters with actual values
        scenario_params = {
            'baseline': {
                'color': 'blue', 
                'label': 'Baseline', 
                'final_co2e': simulation_results['baseline']['final_co2e_stock'],
                'disturbance_events': simulation_results['baseline']['disturbance_events']
            },
            'management': {
                'color': 'green', 
                'label': 'Management', 
                'final_co2e': simulation_results['management']['final_co2e_stock'],
                'disturbance_events': simulation_results['management']['disturbance_events']
            },
            'reforestation': {
                'color': 'orange', 
                'label': 'Reforestation', 
                'final_co2e': simulation_results['reforestation']['final_co2e_stock'],
                'disturbance_events': simulation_results['reforestation']['disturbance_events']
            }
        }
        
        # Get uncertainty factor
        uncertainty_factor = stats['final_co2e_stock']['uncertainty_pct'] / 100
        
        # Create realistic growth curves based on actual final values
        base_curves = {}
        for scenario, params in scenario_params.items():
            # Create a more realistic growth curve that reaches the actual final value
            final_value = params['final_co2e']
            # Use a logistic growth curve that reaches the final value
            growth_rate = 0.12  # Adjust growth rate to reach final value
            base_curves[scenario] = final_value * (1 - np.exp(-growth_rate * years / 25))  # Scale to 25 years
        
        # Row 1: Individual scenario carbon stocks
        scenarios = ['baseline', 'management', 'reforestation']
        for i, scenario in enumerate(scenarios):
            ax = axes[0, i]
            params = scenario_params[scenario]
            
            # Create uncertainty curves
            n_curves = 50
            uncertainty_curves = []
            for j in range(n_curves):
                uncertainty = self.rng.normal(0, uncertainty_factor, len(years)) * base_curves[scenario]
                curve = base_curves[scenario] + uncertainty
                uncertainty_curves.append(curve)
            
            uncertainty_curves = np.array(uncertainty_curves)
            p05 = np.percentile(uncertainty_curves, 5, axis=0)
            p95 = np.percentile(uncertainty_curves, 95, axis=0)
            p50 = np.percentile(uncertainty_curves, 50, axis=0)
            
            # Plot
            ax.fill_between(years, p05, p95, alpha=0.2, color=params['color'])
            ax.plot(years, p50, color=params['color'], linewidth=2)
            ax.plot(years, base_curves[scenario], color=params['color'], linestyle='--', alpha=0.7)
            
            ax.set_title(f'{params["label"]} Carbon Stock', fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('CO2e (t/ha)')
            ax.grid(True, alpha=0.3)
        
        # Row 2: Project Level Additionality (Management - Baseline, Reforestation - Zero)
        # Management additionality
        ax = axes[1, 0]
        management_additionality = base_curves['management'] - base_curves['baseline']
        
        # Add uncertainty to additionality
        n_curves = 50
        additionality_curves = []
        for j in range(n_curves):
            baseline_uncertainty = self.rng.normal(0, uncertainty_factor, len(years)) * base_curves['baseline']
            management_uncertainty = self.rng.normal(0, uncertainty_factor, len(years)) * base_curves['management']
            additionality = (base_curves['management'] + management_uncertainty) - (base_curves['baseline'] + baseline_uncertainty)
            additionality_curves.append(additionality)
        
        additionality_curves = np.array(additionality_curves)
        p05 = np.percentile(additionality_curves, 5, axis=0)
        p95 = np.percentile(additionality_curves, 95, axis=0)
        p50 = np.percentile(additionality_curves, 50, axis=0)
        
        ax.fill_between(years, p05, p95, alpha=0.2, color='green')
        ax.plot(years, p50, color='green', linewidth=2)
        ax.plot(years, management_additionality, color='green', linestyle='--', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax.set_title('Management Additionality\n(Management - Baseline)', fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('CO2e Additionality (t/ha)')
        ax.grid(True, alpha=0.3)
        
        # Reforestation additionality
        ax = axes[1, 1]
        reforestation_additionality = base_curves['reforestation'] - 0  # Reforestation - Zero
        
        # Add uncertainty to reforestation additionality
        additionality_curves = []
        for j in range(n_curves):
            reforestation_uncertainty = self.rng.normal(0, uncertainty_factor, len(years)) * base_curves['reforestation']
            additionality = base_curves['reforestation'] + reforestation_uncertainty
            additionality_curves.append(additionality)
        
        additionality_curves = np.array(additionality_curves)
        p05 = np.percentile(additionality_curves, 5, axis=0)
        p95 = np.percentile(additionality_curves, 95, axis=0)
        p50 = np.percentile(additionality_curves, 50, axis=0)
        
        ax.fill_between(years, p05, p95, alpha=0.2, color='orange')
        ax.plot(years, p50, color='orange', linewidth=2)
        ax.plot(years, reforestation_additionality, color='orange', linestyle='--', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax.set_title('Reforestation Additionality\n(Reforestation - Zero)', fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('CO2e Additionality (t/ha)')
        ax.grid(True, alpha=0.3)
        
        # Combined additionality
        ax = axes[1, 2]
        combined_additionality = management_additionality + reforestation_additionality
        
        # Add uncertainty to combined additionality
        additionality_curves = []
        for j in range(n_curves):
            baseline_uncertainty = self.rng.normal(0, uncertainty_factor, len(years)) * base_curves['baseline']
            management_uncertainty = self.rng.normal(0, uncertainty_factor, len(years)) * base_curves['management']
            reforestation_uncertainty = self.rng.normal(0, uncertainty_factor, len(years)) * base_curves['reforestation']
            
            mgmt_add = (base_curves['management'] + management_uncertainty) - (base_curves['baseline'] + baseline_uncertainty)
            ref_add = base_curves['reforestation'] + reforestation_uncertainty
            combined = mgmt_add + ref_add
            additionality_curves.append(combined)
        
        additionality_curves = np.array(additionality_curves)
        p05 = np.percentile(additionality_curves, 5, axis=0)
        p95 = np.percentile(additionality_curves, 95, axis=0)
        p50 = np.percentile(additionality_curves, 50, axis=0)
        
        ax.fill_between(years, p05, p95, alpha=0.2, color='purple')
        ax.plot(years, p50, color='purple', linewidth=2)
        ax.plot(years, combined_additionality, color='purple', linestyle='--', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax.set_title('Combined Additionality\n(Management + Reforestation)', fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('CO2e Additionality (t/ha)')
        ax.grid(True, alpha=0.3)
        
        # Row 3: Cumulative additionality and summary metrics
        # Cumulative management additionality
        ax = axes[2, 0]
        cumulative_mgmt = np.cumsum(management_additionality)
        
        # Add uncertainty to cumulative
        cumulative_curves = []
        for j in range(n_curves):
            baseline_uncertainty = self.rng.normal(0, uncertainty_factor, len(years)) * base_curves['baseline']
            management_uncertainty = self.rng.normal(0, uncertainty_factor, len(years)) * base_curves['management']
            annual_add = (base_curves['management'] + management_uncertainty) - (base_curves['baseline'] + baseline_uncertainty)
            cumulative = np.cumsum(annual_add)
            cumulative_curves.append(cumulative)
        
        cumulative_curves = np.array(cumulative_curves)
        p05 = np.percentile(cumulative_curves, 5, axis=0)
        p95 = np.percentile(cumulative_curves, 95, axis=0)
        p50 = np.percentile(cumulative_curves, 50, axis=0)
        
        ax.fill_between(years, p05, p95, alpha=0.2, color='green')
        ax.plot(years, p50, color='green', linewidth=2)
        ax.plot(years, cumulative_mgmt, color='green', linestyle='--', alpha=0.7)
        
        ax.set_title('Cumulative Management\nAdditionality', fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Cumulative CO2e (t/ha)')
        ax.grid(True, alpha=0.3)
        
        # Cumulative reforestation additionality
        ax = axes[2, 1]
        cumulative_ref = np.cumsum(reforestation_additionality)
        
        # Add uncertainty to cumulative
        cumulative_curves = []
        for j in range(n_curves):
            reforestation_uncertainty = self.rng.normal(0, uncertainty_factor, len(years)) * base_curves['reforestation']
            annual_add = base_curves['reforestation'] + reforestation_uncertainty
            cumulative = np.cumsum(annual_add)
            cumulative_curves.append(cumulative)
        
        cumulative_curves = np.array(cumulative_curves)
        p05 = np.percentile(cumulative_curves, 5, axis=0)
        p95 = np.percentile(cumulative_curves, 95, axis=0)
        p50 = np.percentile(cumulative_curves, 50, axis=0)
        
        ax.fill_between(years, p05, p95, alpha=0.2, color='orange')
        ax.plot(years, p50, color='orange', linewidth=2)
        ax.plot(years, cumulative_ref, color='orange', linestyle='--', alpha=0.7)
        
        ax.set_title('Cumulative Reforestation\nAdditionality', fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Cumulative CO2e (t/ha)')
        ax.grid(True, alpha=0.3)
        
        # Summary statistics
        ax = axes[2, 2]
        ax.axis('off')
        
        # Calculate summary statistics using actual simulation results
        baseline_final = simulation_results['baseline']['final_co2e_stock']
        management_final = simulation_results['management']['final_co2e_stock']
        reforestation_final = simulation_results['reforestation']['final_co2e_stock']
        
        # Calculate actual additionality
        management_additionality_actual = management_final - baseline_final
        reforestation_additionality_actual = reforestation_final - 0  # Reforestation vs zero
        combined_additionality_actual = management_additionality_actual + reforestation_additionality_actual
        
        # Create summary text with actual values
        summary_text = f"""
CARBON ADDITIONALITY SUMMARY
(Paris Agreement Adaptive Scenario)

ACTUAL SCENARIO RESULTS:
• Baseline: {baseline_final:.1f} tCO2e/ha
• Management: {management_final:.1f} tCO2e/ha
• Reforestation: {reforestation_final:.1f} tCO2e/ha

ADDITIONALITY CALCULATIONS:
• Management: {management_additionality_actual:.1f} tCO2e/ha
• Reforestation: {reforestation_additionality_actual:.1f} tCO2e/ha
• Combined: {combined_additionality_actual:.1f} tCO2e/ha

DISTURBANCE EVENTS:
• Baseline: {simulation_results['baseline']['disturbance_events']} events
• Management: {simulation_results['management']['disturbance_events']} events
• Reforestation: {simulation_results['reforestation']['disturbance_events']} events

UNCERTAINTY: ±{stats['final_co2e_stock']['uncertainty_pct']:.1f}%
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Carbon Additionality Dashboard - Project Level Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.uncertainty_dir / 'carbon_additionality_dashboard.png', bbox_inches='tight')
        plt.close()
    
    def _load_simulation_results(self) -> Dict[str, Any]:
        """Load actual simulation results from the summary JSON file."""
        import json
        
        # For scenario-specific analysis, look in the scenario directory
        # The uncertainty analysis is in output/[scenario_name]/uncertainty_analysis/
        # So we need to go up one level to get to the scenario directory
        if self.scenario_name:
            # Go up from uncertainty_analysis/ to scenario_name/
            summary_file = self.uncertainty_dir.parent / 'simulation_summary.json'
        else:
            # For centralized analysis, look in current output directory
            summary_file = self.output_dir / 'simulation_summary.json'
        
        print(f"Looking for simulation results at: {summary_file}")
        
        if not summary_file.exists():
            print(f"Warning: Simulation summary file not found at {summary_file}")
            # Fallback to default values if file doesn't exist
            return {
                'baseline': {'final_co2e_stock': 100.0, 'disturbance_events': 1},
                'management': {'final_co2e_stock': 120.0, 'disturbance_events': 1},
                'reforestation': {'final_co2e_stock': 150.0, 'disturbance_events': 1}
            }
        
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
            
            print(f"Successfully loaded simulation results from {summary_file}")
            return data['summary_results']['scenarios']
        except Exception as e:
            print(f"Warning: Could not load simulation results: {e}")
            # Fallback to default values
            return {
                'baseline': {'final_co2e_stock': 100.0, 'disturbance_events': 1},
                'management': {'final_co2e_stock': 120.0, 'disturbance_events': 1},
                'reforestation': {'final_co2e_stock': 150.0, 'disturbance_events': 1}
            }
    
    def _generate_uncertainty_report(self, stats: Dict[str, Any], results_df: pd.DataFrame) -> None:
        """Generate uncertainty analysis report."""
        
        report_path = self.uncertainty_dir / 'uncertainty_analysis_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Forest Growth & Carbon Uncertainty Analysis Report\n\n")
            f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This report presents uncertainty analysis for forest growth and carbon ")
            f.write("sequestration parameters. The analysis quantifies how parameter uncertainty ")
            f.write("affects biomass growth and carbon outcomes.\n\n")
            
            f.write("## Methodology\n\n")
            f.write("- **Monte Carlo runs**: 500 simulations\n")
            f.write("- **Parameters varied**: mortality, disturbance, TYF parameters, FPI\n")
            f.write("- **Confidence intervals**: 5th, 25th, 50th, 75th, 95th percentiles\n")
            f.write("- **Focus**: Growth and carbon sequestration only (no economics)\n\n")
            
            f.write("## Key Results\n\n")
            
            # Biomass uncertainty
            agb_stats = stats['total_agb']
            f.write(f"### Above-Ground Biomass\n")
            f.write(f"- **Mean**: {agb_stats['mean']:.1f} t/ha\n")
            f.write(f"- **90% CI**: {agb_stats['p5']:.1f} - {agb_stats['p95']:.1f} t/ha\n")
            f.write(f"- **Uncertainty**: ±{agb_stats['uncertainty_pct']:.1f}%\n")
            f.write(f"- **Coefficient of Variation**: {agb_stats['cv']:.1f}%\n\n")
            
            # Carbon uncertainty
            co2e_stats = stats['final_co2e_stock']
            f.write(f"### Total Carbon Stock\n")
            f.write(f"- **Mean**: {co2e_stats['mean']:.1f} tCO2e/ha\n")
            f.write(f"- **90% CI**: {co2e_stats['p5']:.1f} - {co2e_stats['p95']:.1f} tCO2e/ha\n")
            f.write(f"- **Uncertainty**: ±{co2e_stats['uncertainty_pct']:.1f}%\n")
            f.write(f"- **Coefficient of Variation**: {co2e_stats['cv']:.1f}%\n\n")
            
            # Annual increment uncertainty
            inc_stats = stats['mean_annual_increment']
            f.write(f"### Annual Carbon Increment\n")
            f.write(f"- **Mean**: {inc_stats['mean']:.1f} tCO2e/ha/yr\n")
            f.write(f"- **90% CI**: {inc_stats['p5']:.1f} - {inc_stats['p95']:.1f} tCO2e/ha/yr\n")
            f.write(f"- **Uncertainty**: ±{inc_stats['uncertainty_pct']:.1f}%\n")
            f.write(f"- **Coefficient of Variation**: {inc_stats['cv']:.1f}%\n\n")
            
            # Parameter importance
            f.write("## Parameter Importance\n\n")
            f.write("Parameters ranked by average correlation with carbon outcomes:\n\n")
            
            correlations = stats['correlations']
            param_importance = {}
            for param in correlations['final_co2e_stock'].keys():
                avg_abs_corr = np.mean([abs(correlations[metric][param]) for metric in correlations.keys()])
                param_importance[param] = avg_abs_corr
            
            sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
            
            for i, (param, importance) in enumerate(sorted_params, 1):
                f.write(f"{i}. **{param.replace('_', ' ').title()}**: {importance:.3f}\n")
            
            f.write("\n## Key Findings\n\n")
            f.write("1. **Parameter Sensitivity**: The analysis reveals which biological ")
            f.write("parameters have the strongest influence on carbon outcomes.\n\n")
            
            f.write("2. **Uncertainty Ranges**: All carbon metrics show significant uncertainty ")
            f.write("ranges that should be communicated to stakeholders.\n\n")
            
            f.write("3. **Risk Assessment**: The 90% confidence intervals provide a robust ")
            f.write("basis for risk assessment and project planning.\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("1. **Focus Data Collection**: Prioritize measuring parameters with high ")
            f.write("sensitivity to reduce overall uncertainty.\n\n")
            
            f.write("2. **Communicate Uncertainty**: Present carbon estimates as ranges ")
            f.write("rather than single point estimates.\n\n")
            
            f.write("3. **Monitor Key Parameters**: Establish monitoring programs for ")
            f.write("the most influential parameters identified in this analysis.\n\n")
            
            f.write("4. **Update Analysis**: Re-run uncertainty analysis as new data ")
            f.write("becomes available to refine parameter distributions.\n\n")
        
        print(f"Uncertainty report saved to: {report_path}")


def run_uncertainty_analysis_for_scenario(scenario_results: Dict[str, Any], 
                                        output_dir: Path, 
                                        n_runs: int = 500) -> Dict[str, Any]:
    """
    Run uncertainty analysis for a single scenario.
    
    Args:
        scenario_results: Results from scenario simulation
        output_dir: Output directory for the scenario
        n_runs: Number of Monte Carlo iterations
        
    Returns:
        Uncertainty analysis results
    """
    uncertainty_analyzer = GrowthCarbonUncertainty(output_dir)
    return uncertainty_analyzer.run_uncertainty_analysis(scenario_results, n_runs)
