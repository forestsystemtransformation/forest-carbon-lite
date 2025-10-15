#!/usr/bin/env python3
"""
Data Matrix Generator for Forest Carbon Lite

Creates comparison matrices by generating new plots from CSV data rather than pulling existing images.
This allows for more flexible data-driven comparisons and custom visualizations.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Optional, Tuple
import sys

# Import color management from the forest_carbon package
try:
    from forest_carbon.utils.colors import color_manager
except ImportError:
    # Fallback colors if package not available
    class SimpleColorManager:
        def get_scenario_style(self, scenario):
            colors = {
                'baseline': {'color': '#e74c3c', 'linestyle': '-', 'linewidth': 2.5, 'alpha': 0.8},
                'management': {'color': '#3498db', 'linestyle': '-', 'linewidth': 2.5, 'alpha': 0.8},
                'reforestation': {'color': '#27ae60', 'linestyle': '-', 'linewidth': 2.5, 'alpha': 0.8}
            }
            return colors.get(scenario, {'color': '#95a5a6', 'linestyle': '-', 'linewidth': 2, 'alpha': 0.7})
    
    color_manager = SimpleColorManager()

class DataMatrixGenerator:
    """Generates comparison matrices by creating new plots from CSV data."""
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the data matrix generator.
        
        Args:
            output_dir: Directory containing scenario results
        """
        self.output_dir = Path(output_dir)
        self.output_matrix_dir = Path("output_matrix")  # Matrix outputs
        
        # Available scenarios (auto-detected)
        self.available_scenarios = self._detect_scenarios()
        
        # Available data files for each scenario
        self.data_files = {
            'sequestration_curves': 'sequestration_curves.csv',
            'results_summary': 'results_summary.csv',
            'finance_results': 'finance_results.csv',
            'cashflow_breakdown': 'cashflow_breakdown.csv'
        }
    
    def _detect_scenarios(self) -> List[str]:
        """Detect available scenarios from the output directory."""
        scenarios = []
        if self.output_dir.exists():
            for item in self.output_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.') and item.name != 'analysis':
                    scenarios.append(item.name)
        return sorted(scenarios)
    
    def _get_scenario_data_dir(self, scenario: str) -> Path:
        """Get the data directory for a scenario."""
        # Try different year directories (25 years, 27 years, 52 years, etc.)
        for year_dir in self.output_dir.glob(f"{scenario}/*"):
            if year_dir.is_dir():
                data_dir = year_dir / "other"
                if data_dir.exists():
                    return data_dir
        
        # Fallback to direct scenario/other
        return self.output_dir / scenario / "other"
    
    def _scenario_name_to_title(self, scenario: str) -> str:
        """Convert scenario directory name to a readable title."""
        title = scenario.replace('_', ' ').title()
        title = title.replace('Etof', 'ETOF')
        title = title.replace('Afm', 'AFM')
        title = title.replace('Co2', 'CO2')
        return title
    
    def _load_scenario_data(self, scenario: str, data_type: str = 'sequestration_curves') -> Optional[pd.DataFrame]:
        """
        Load data for a specific scenario.
        
        Args:
            scenario: Scenario name
            data_type: Type of data to load ('sequestration_curves', 'results_summary', etc.)
            
        Returns:
            DataFrame with the data or None if not found
        """
        data_dir = self._get_scenario_data_dir(scenario)
        filename = self.data_files.get(data_type)
        
        if not filename:
            return None
        
        file_path = data_dir / filename
        if file_path.exists():
            try:
                return pd.read_csv(file_path)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                return None
        return None
    
    def create_carbon_stocks_matrix(self, scenarios: List[str], 
                                  figsize: Tuple[int, int] = (20, 15),
                                  max_scenarios_per_row: int = 4,
                                  grid_rows: Optional[int] = None,
                                  grid_cols: Optional[int] = None) -> Path:
        """
        Create a matrix comparing carbon stocks across scenarios.
        
        Args:
            scenarios: List of scenarios to compare
            figsize: Figure size (width, height)
            max_scenarios_per_row: Maximum number of scenarios per row
            
        Returns:
            Path to saved matrix image
        """
        # Filter scenarios that have data
        valid_scenarios = []
        scenario_data = {}
        
        for scenario in scenarios:
            data = self._load_scenario_data(scenario, 'sequestration_curves')
            if data is not None and 'year' in data.columns:
                valid_scenarios.append(scenario)
                scenario_data[scenario] = data
        
        if not valid_scenarios:
            raise ValueError("No scenarios found with valid sequestration curves data")
        
        # Calculate grid dimensions
        n_scenarios = len(valid_scenarios)
        
        if grid_rows is not None and grid_cols is not None:
            # Use exact grid dimensions
            n_rows = grid_rows
            n_cols = grid_cols
        else:
            # Use automatic calculation
            n_cols = min(max_scenarios_per_row, n_scenarios)
            n_rows = (n_scenarios + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Plot each scenario
        for i, scenario in enumerate(valid_scenarios):
            ax = axes[i]
            data = scenario_data[scenario]
            
            # Plot the three scenarios (baseline, management, reforestation)
            if 'baseline_co2e' in data.columns:
                style = color_manager.get_scenario_style('baseline')
                ax.plot(data['year'], data['baseline_co2e'], 
                       label='Degrading Baseline', **style)
            
            if 'management_co2e' in data.columns:
                style = color_manager.get_scenario_style('management')
                ax.plot(data['year'], data['management_co2e'], 
                       label='AFM Management', **style)
            
            if 'reforestation_co2e' in data.columns:
                style = color_manager.get_scenario_style('reforestation')
                ax.plot(data['year'], data['reforestation_co2e'], 
                       label='Reforestation', **style)
            
            # Customize the plot
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Total Carbon Stock (t CO2e/ha)', fontsize=12)
            ax.set_title(f"{self._scenario_name_to_title(scenario)}", 
                       fontsize=14, fontweight='bold', pad=15)
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3)
            
            # Add zero line for reference
            ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
        
        # Hide unused subplots
        for i in range(n_scenarios, len(axes)):
            axes[i].axis('off')
        
        # Set main title
        fig.suptitle(f"Carbon Stocks Comparison - {len(valid_scenarios)} Scenarios", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save the matrix
        self.output_matrix_dir.mkdir(exist_ok=True)
        output_path = self.output_matrix_dir / f"carbon_stocks_data_matrix_{len(valid_scenarios)}scenarios.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Data matrix saved: {output_path}")
        return output_path
    
    def create_additionality_matrix(self, scenarios: List[str],
                                  figsize: Tuple[int, int] = (20, 15),
                                  max_scenarios_per_row: int = 4,
                                  grid_rows: Optional[int] = None,
                                  grid_cols: Optional[int] = None) -> Path:
        """
        Create a matrix comparing carbon additionality across scenarios.
        
        Args:
            scenarios: List of scenarios to compare
            figsize: Figure size (width, height)
            max_scenarios_per_row: Maximum number of scenarios per row
            
        Returns:
            Path to saved matrix image
        """
        # Filter scenarios that have data
        valid_scenarios = []
        scenario_data = {}
        
        for scenario in scenarios:
            data = self._load_scenario_data(scenario, 'sequestration_curves')
            if data is not None and 'year' in data.columns:
                valid_scenarios.append(scenario)
                scenario_data[scenario] = data
        
        if not valid_scenarios:
            raise ValueError("No scenarios found with valid sequestration curves data")
        
        # Calculate grid dimensions
        n_scenarios = len(valid_scenarios)
        
        if grid_rows is not None and grid_cols is not None:
            # Use exact grid dimensions
            n_rows = grid_rows
            n_cols = grid_cols
        else:
            # Use automatic calculation
            n_cols = min(max_scenarios_per_row, n_scenarios)
            n_rows = (n_scenarios + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Plot each scenario
        for i, scenario in enumerate(valid_scenarios):
            ax = axes[i]
            data = scenario_data[scenario]
            
            # Calculate additionality (difference from baseline)
            if 'baseline_co2e' in data.columns and 'management_co2e' in data.columns:
                management_additionality = data['management_co2e'] - data['baseline_co2e']
                style = color_manager.get_scenario_style('management')
                ax.plot(data['year'], management_additionality, 
                       label='Management Additionality', **style)
            
            if 'reforestation_co2e' in data.columns:
                # Reforestation additionality vs 0 (new planting)
                reforestation_additionality = data['reforestation_co2e'] - 0
                style = color_manager.get_scenario_style('reforestation')
                ax.plot(data['year'], reforestation_additionality, 
                       label='Reforestation Additionality', **style)
            
            # Customize the plot
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Carbon Additionality (t CO2e/ha)', fontsize=12)
            ax.set_title(f"{self._scenario_name_to_title(scenario)}", 
                       fontsize=14, fontweight='bold', pad=15)
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3)
            
            # Add zero line for reference
            ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.6)
        
        # Hide unused subplots
        for i in range(n_scenarios, len(axes)):
            axes[i].axis('off')
        
        # Set main title
        fig.suptitle(f"Carbon Additionality Comparison - {len(valid_scenarios)} Scenarios", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save the matrix
        self.output_matrix_dir.mkdir(exist_ok=True)
        output_path = self.output_matrix_dir / f"additionality_data_matrix_{len(valid_scenarios)}scenarios.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Data matrix saved: {output_path}")
        return output_path
    
    def create_economics_matrix(self, scenarios: List[str],
                              figsize: Tuple[int, int] = (20, 15),
                              max_scenarios_per_row: int = 4,
                              grid_rows: Optional[int] = None,
                              grid_cols: Optional[int] = None) -> Path:
        """
        Create a matrix comparing economic metrics across scenarios.
        
        Args:
            scenarios: List of scenarios to compare
            figsize: Figure size (width, height)
            max_scenarios_per_row: Maximum number of scenarios per row
            
        Returns:
            Path to saved matrix image
        """
        # Filter scenarios that have economic data
        valid_scenarios = []
        scenario_data = {}
        
        for scenario in scenarios:
            data = self._load_scenario_data(scenario, 'results_summary')
            if data is not None and 'npv' in data.columns:
                valid_scenarios.append(scenario)
                scenario_data[scenario] = data
        
        if not valid_scenarios:
            raise ValueError("No scenarios found with valid economic data")
        
        # Calculate grid dimensions
        n_scenarios = len(valid_scenarios)
        
        if grid_rows is not None and grid_cols is not None:
            # Use exact grid dimensions
            n_rows = grid_rows
            n_cols = grid_cols
        else:
            # Use automatic calculation
            n_cols = min(max_scenarios_per_row, n_scenarios)
            n_rows = (n_scenarios + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Plot each scenario
        for i, scenario in enumerate(valid_scenarios):
            ax = axes[i]
            data = scenario_data[scenario]
            
            # Create bar chart of economic metrics
            scenarios_in_data = data['scenario'].tolist()
            npv_values = []
            labels = []
            colors = []
            
            for _, row in data.iterrows():
                if row['scenario'] == 'baseline':
                    continue  # Skip baseline (usually no economics)
                
                npv_values.append(row['npv'])
                labels.append(row['scenario'].title())
                
                # Get color for scenario
                if row['scenario'] == 'management':
                    colors.append(color_manager.get_scenario_style('management')['color'])
                elif row['scenario'] == 'reforestation':
                    colors.append(color_manager.get_scenario_style('reforestation')['color'])
                else:
                    colors.append('#95a5a6')
            
            if npv_values:
                bars = ax.bar(labels, npv_values, color=colors, alpha=0.7)
                
                # Add value labels on bars
                for bar, value in zip(bars, npv_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'${value:,.0f}', ha='center', va='bottom', fontsize=10)
            
            # Customize the plot
            ax.set_ylabel('NPV ($/ha)', fontsize=12)
            ax.set_title(f"{self._scenario_name_to_title(scenario)}", 
                       fontsize=14, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x-axis labels if needed
            if len(labels) > 2:
                ax.tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(n_scenarios, len(axes)):
            axes[i].axis('off')
        
        # Set main title
        fig.suptitle(f"Economic Performance Comparison - {len(valid_scenarios)} Scenarios", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save the matrix
        self.output_matrix_dir.mkdir(exist_ok=True)
        output_path = self.output_matrix_dir / f"economics_data_matrix_{len(valid_scenarios)}scenarios.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Data matrix saved: {output_path}")
        return output_path
    
    def create_combined_matrix(self, scenarios: List[str], 
                             figsize: Tuple[int, int] = (24, 18),
                             max_scenarios_per_row: int = 3,
                             grid_rows: Optional[int] = None,
                             grid_cols: Optional[int] = None) -> Path:
        """
        Create a combined matrix showing both carbon stocks and additionality.
        
        Args:
            scenarios: List of scenarios to compare
            figsize: Figure size (width, height)
            max_scenarios_per_row: Maximum number of scenarios per row
            grid_rows: Exact number of rows (overrides max_scenarios_per_row)
            grid_cols: Exact number of columns (overrides max_scenarios_per_row)
            
        Returns:
            Path to saved matrix image
        """
        # Filter scenarios that have data
        valid_scenarios = []
        scenario_data = {}
        
        for scenario in scenarios:
            data = self._load_scenario_data(scenario, 'sequestration_curves')
            if data is not None and 'year' in data.columns:
                valid_scenarios.append(scenario)
                scenario_data[scenario] = data
        
        if not valid_scenarios:
            raise ValueError("No scenarios found with valid sequestration curves data")
        
        # Calculate grid dimensions - 2 plots per scenario (stocks + additionality)
        n_scenarios = len(valid_scenarios)
        plots_per_scenario = 2  # Carbon stocks + additionality
        
        if grid_rows is not None and grid_cols is not None:
            # Use exact grid dimensions
            n_rows = grid_rows
            n_cols = grid_cols
        else:
            # Use automatic calculation
            n_cols = min(max_scenarios_per_row, n_scenarios)
            n_rows = (n_scenarios + n_cols - 1) // n_cols
        
        
        # Create figure with subplots for each scenario
        fig, axes = plt.subplots(n_rows, n_cols * plots_per_scenario, figsize=figsize)
        
        # Handle single row/column cases
        total_columns = n_cols * plots_per_scenario
        if n_rows == 1:
            axes = axes.reshape(1, total_columns)
        else:
            axes = axes.reshape(n_rows, total_columns)
        
        # Plot each scenario
        for i, scenario in enumerate(valid_scenarios):
            data = scenario_data[scenario]
            
            # Calculate row and column positions for this scenario
            row = i // n_cols
            col = (i % n_cols) * plots_per_scenario
            
            # Plot 1: Carbon Stocks
            if n_rows == 1:
                ax_stocks = axes[0, col]
            else:
                ax_stocks = axes[row, col]
            
            # Plot the three scenarios (baseline, management, reforestation)
            if 'baseline_co2e' in data.columns:
                style = color_manager.get_scenario_style('baseline')
                ax_stocks.plot(data['year'], data['baseline_co2e'], 
                             label='Degrading Baseline', **style)
            
            if 'management_co2e' in data.columns:
                style = color_manager.get_scenario_style('management')
                ax_stocks.plot(data['year'], data['management_co2e'], 
                             label='AFM Management', **style)
            
            if 'reforestation_co2e' in data.columns:
                style = color_manager.get_scenario_style('reforestation')
                ax_stocks.plot(data['year'], data['reforestation_co2e'], 
                             label='Reforestation', **style)
            
            # Customize the carbon stocks plot
            ax_stocks.set_xlabel('Year', fontsize=10)
            ax_stocks.set_ylabel('Total Carbon Stock\n(t CO2e/ha)', fontsize=10)
            ax_stocks.set_title(f"{self._scenario_name_to_title(scenario)}\nCarbon Stocks", 
                              fontsize=12, fontweight='bold', pad=10)
            ax_stocks.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=9)
            ax_stocks.grid(True, alpha=0.3)
            ax_stocks.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
            
            # Plot 2: Additionality
            if n_rows == 1:
                ax_additionality = axes[col + 1]
            else:
                ax_additionality = axes[row, col + 1]
            
            # Calculate additionality (difference from baseline)
            if 'baseline_co2e' in data.columns and 'management_co2e' in data.columns:
                management_additionality = data['management_co2e'] - data['baseline_co2e']
                style = color_manager.get_scenario_style('management')
                ax_additionality.plot(data['year'], management_additionality, 
                                    label='Management Additionality', **style)
            
            if 'reforestation_co2e' in data.columns:
                # Reforestation additionality vs 0 (new planting)
                reforestation_additionality = data['reforestation_co2e'] - 0
                style = color_manager.get_scenario_style('reforestation')
                ax_additionality.plot(data['year'], reforestation_additionality, 
                                    label='Reforestation Additionality', **style)
            
            # Customize the additionality plot
            ax_additionality.set_xlabel('Year', fontsize=10)
            ax_additionality.set_ylabel('Carbon Additionality\n(t CO2e/ha)', fontsize=10)
            ax_additionality.set_title(f"{self._scenario_name_to_title(scenario)}\nAdditionality", 
                                     fontsize=12, fontweight='bold', pad=10)
            ax_additionality.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=9)
            ax_additionality.grid(True, alpha=0.3)
            ax_additionality.axhline(y=0, color='black', linewidth=0.8, alpha=0.6)
        
        # Hide unused subplots
        total_cols = n_cols * plots_per_scenario
        for i in range(n_scenarios, n_rows * n_cols):
            row = i // n_cols
            col_start = (i % n_cols) * plots_per_scenario
            
            for j in range(plots_per_scenario):
                if n_rows == 1:
                    if col_start + j < total_cols:
                        axes[col_start + j].axis('off')
                else:
                    if col_start + j < total_cols:
                        axes[row, col_start + j].axis('off')
        
        # Set main title
        fig.suptitle(f"Combined Analysis: Carbon Stocks & Additionality - {len(valid_scenarios)} Scenarios", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save the matrix
        self.output_matrix_dir.mkdir(exist_ok=True)
        output_path = self.output_matrix_dir / f"combined_carbon_additionality_matrix_{len(valid_scenarios)}scenarios.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Combined matrix saved: {output_path}")
        return output_path
    
    def list_available_options(self):
        """Print available scenarios and data types."""
        print("Available Scenarios:")
        for i, scenario in enumerate(self.available_scenarios, 1):
            print(f"  {i:2d}. {scenario}")
        
        print(f"\nAvailable Data Types:")
        for i, (key, filename) in enumerate(self.data_files.items(), 1):
            print(f"  {i:2d}. {key} - {filename}")
        
        print(f"\nTotal scenarios: {len(self.available_scenarios)}")
        print(f"Total data types: {len(self.data_files)}")


def main():
    """Command-line interface for the data matrix generator."""
    parser = argparse.ArgumentParser(description="Generate comparison matrices from CSV data")
    
    parser.add_argument('--list', action='store_true', 
                       help='List available scenarios and data types')
    
    parser.add_argument('--scenarios', type=str,
                       help='Comma-separated list of scenarios to compare')
    
    parser.add_argument('--matrix-type', choices=['carbon_stocks', 'additionality', 'economics', 'combined'], 
                       default='carbon_stocks',
                       help='Type of matrix to create (default: carbon_stocks)')
    
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory containing scenarios (default: output)')
    
    parser.add_argument('--max-per-row', type=int, default=4,
                       help='Maximum scenarios per row (default: 4, use 3 for combined matrix)')
    
    parser.add_argument('--grid-rows', type=int,
                       help='Specify exact number of rows (overrides max-per-row)')
    
    parser.add_argument('--grid-cols', type=int,
                       help='Specify exact number of columns (overrides max-per-row)')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = DataMatrixGenerator(args.output_dir)
    
    # List options if requested
    if args.list:
        generator.list_available_options()
        return
    
    # Check if scenarios provided
    if not args.scenarios:
        print("Error: --scenarios is required when not using --list")
        print("Use --list to see available scenarios")
        return
    
    # Split comma-separated scenarios
    scenarios = [s.strip() for s in args.scenarios.split(',')]
    
    # Validate scenarios
    invalid_scenarios = [s for s in scenarios if s not in generator.available_scenarios]
    if invalid_scenarios:
        print(f"Error: Invalid scenarios: {invalid_scenarios}")
        print("Available scenarios:")
        for scenario in generator.available_scenarios:
            print(f"  - {scenario}")
        return
    
    # Generate matrix based on type
    try:
        if args.matrix_type == 'carbon_stocks':
            generator.create_carbon_stocks_matrix(scenarios, max_scenarios_per_row=args.max_per_row,
                                                grid_rows=args.grid_rows, grid_cols=args.grid_cols)
        elif args.matrix_type == 'additionality':
            generator.create_additionality_matrix(scenarios, max_scenarios_per_row=args.max_per_row,
                                                grid_rows=args.grid_rows, grid_cols=args.grid_cols)
        elif args.matrix_type == 'economics':
            generator.create_economics_matrix(scenarios, max_scenarios_per_row=args.max_per_row,
                                            grid_rows=args.grid_rows, grid_cols=args.grid_cols)
        elif args.matrix_type == 'combined':
            generator.create_combined_matrix(scenarios, figsize=(24, 18), 
                                           max_scenarios_per_row=args.max_per_row,
                                           grid_rows=args.grid_rows, grid_cols=args.grid_cols)
    except Exception as e:
        print(f"Error creating matrix: {e}")
        return


if __name__ == "__main__":
    main()
