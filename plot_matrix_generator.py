#!/usr/bin/env python3
"""
Plot Matrix Generator for Forest Carbon Lite

Creates comparison matrices by pulling existing plots from different scenarios
and arranging them in a grid format for easy comparison. No new computations needed.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional, Tuple
import argparse
import sys
from PIL import Image

class PlotMatrixGenerator:
    """Generates comparison matrices from existing scenario plots."""
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the plot matrix generator.
        
        Args:
            output_dir: Directory containing scenario results
        """
        self.output_dir = Path(output_dir)
        self.output_matrix_dir = Path("output_matrix")  # Matrix outputs
        
        # Standard plot types available in all scenarios
        self.plot_types = [
            'additionality',
            'biomass_all_scenarios', 
            'carbon_pools_breakdown_baseline',
            'carbon_pools_breakdown_management',
            'carbon_pools_breakdown_reforestation',
            'carbon_pools_comparison',
            'economics_management',
            'economics_reforestation',
            'management_minus_reforestation',
            'reforestation_minus_losses',
            'total_carbon_stocks_all_scenarios'
        ]
        
        # Available scenarios (auto-detected)
        self.available_scenarios = self._detect_scenarios()
        
    def _detect_scenarios(self) -> List[str]:
        """Detect available scenarios from the output directory."""
        scenarios = []
        if self.output_dir.exists():
            for item in self.output_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.') and item.name != 'analysis':
                    scenarios.append(item.name)
        return sorted(scenarios)
    
    def _get_scenario_plots_dir(self, scenario: str) -> Path:
        """Get the plots directory for a scenario."""
        # Try different year directories (25 years, 52 years, etc.)
        for year_dir in self.output_dir.glob(f"{scenario}/*"):
            if year_dir.is_dir():
                plots_dir = year_dir / "plots"
                if plots_dir.exists():
                    return plots_dir
        
        # Fallback to direct scenario/plots
        return self.output_dir / scenario / "plots"
    
    def _scenario_name_to_title(self, scenario: str) -> str:
        """Convert scenario directory name to a readable title."""
        # Replace underscores with spaces and title case
        title = scenario.replace('_', ' ').title()
        
        # Handle specific cases
        title = title.replace('Etof', 'ETOF')
        title = title.replace('Afm', 'AFM')
        title = title.replace('Co2', 'CO2')
        
        return title
    
    def _plot_type_to_title(self, plot_type: str) -> str:
        """Convert plot type filename to a readable title."""
        title_map = {
            'additionality': 'Carbon Additionality',
            'biomass_all_scenarios': 'Biomass All Scenarios',
            'carbon_pools_breakdown_baseline': 'Carbon Pools - Baseline',
            'carbon_pools_breakdown_management': 'Carbon Pools - Management',
            'carbon_pools_breakdown_reforestation': 'Carbon Pools - Reforestation',
            'carbon_pools_comparison': 'Carbon Pools Comparison',
            'economics_management': 'Economics - Management',
            'economics_reforestation': 'Economics - Reforestation',
            'management_minus_reforestation': 'Management vs Reforestation',
            'reforestation_minus_losses': 'Reforestation Minus Losses',
            'total_carbon_stocks_all_scenarios': 'Total Carbon Stocks'
        }
        return title_map.get(plot_type, plot_type.replace('_', ' ').title())
    
    def _crop_legend(self, image_array: np.ndarray, crop_right: float = 0.25) -> np.ndarray:
        """
        Crop out the legend area from the right side of the image.
        
        Args:
            image_array: Image as numpy array
            crop_right: Fraction of image width to crop from the right (default: 0.25)
            
        Returns:
            Cropped image array
        """
        height, width = image_array.shape[:2]
        crop_width = int(width * (1 - crop_right))
        return image_array[:, :crop_width]
    
    def _load_and_crop_image(self, image_path: Path, crop_legend: bool = True) -> np.ndarray:
        """
        Load image and optionally crop out the legend.
        
        Args:
            image_path: Path to the image file
            crop_legend: Whether to crop the legend from the right side
            
        Returns:
            Image array (cropped if requested)
        """
        try:
            # Load image using PIL for better control
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to numpy array
                image_array = np.array(img)
                
                # Crop legend if requested
                if crop_legend:
                    image_array = self._crop_legend(image_array)
                
                return image_array
        except Exception as e:
            raise Exception(f"Error loading image {image_path}: {e}")
    
    def create_plot_type_matrix(self, plot_type: str, scenarios: Optional[List[str]] = None,
                               max_scenarios_per_row: int = 4, figsize: Tuple[int, int] = (20, 15),
                               crop_legend: bool = True) -> Path:
        """
        Create a matrix comparing one plot type across multiple scenarios.
        
        Args:
            plot_type: Type of plot to compare (e.g., 'additionality')
            scenarios: List of scenarios to include (None = all available)
            max_scenarios_per_row: Maximum number of scenarios per row
            figsize: Figure size (width, height)
            crop_legend: Whether to crop out legends from individual plots
            
        Returns:
            Path to saved matrix image
        """
        if scenarios is None:
            scenarios = self.available_scenarios
        
        if not scenarios:
            raise ValueError("No scenarios available for comparison")
        
        if plot_type not in self.plot_types:
            raise ValueError(f"Plot type '{plot_type}' not found. Available types: {self.plot_types}")
        
        # Filter scenarios that have the requested plot
        valid_scenarios = []
        for scenario in scenarios:
            plots_dir = self._get_scenario_plots_dir(scenario)
            plot_path = plots_dir / f"{plot_type}.png"
            if plot_path.exists():
                valid_scenarios.append(scenario)
        
        if not valid_scenarios:
            raise ValueError(f"No scenarios found with plot type '{plot_type}'")
        
        # Calculate grid dimensions
        n_scenarios = len(valid_scenarios)
        n_cols = min(max_scenarios_per_row, n_scenarios)
        n_rows = (n_scenarios + n_cols - 1) // n_cols  # Ceiling division
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Load and display each plot
        for i, scenario in enumerate(valid_scenarios):
            ax = axes[i]
            
            # Load the plot image
            plots_dir = self._get_scenario_plots_dir(scenario)
            plot_path = plots_dir / f"{plot_type}.png"
            
            try:
                img = self._load_and_crop_image(plot_path, crop_legend)
                ax.imshow(img)
                ax.set_title(f"{self._scenario_name_to_title(scenario)}", 
                           fontsize=12, fontweight='bold', pad=10)
                ax.axis('off')
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading plot:\n{str(e)}", 
                       ha='center', va='center', transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
                ax.set_title(f"{self._scenario_name_to_title(scenario)} (ERROR)", 
                           fontsize=12, fontweight='bold', color='red')
                ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_scenarios, len(axes)):
            axes[i].axis('off')
        
        # Set main title
        plot_title = self._plot_type_to_title(plot_type)
        fig.suptitle(f"{plot_title} - Scenario Comparison Matrix", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save the matrix
        self.output_matrix_dir.mkdir(exist_ok=True)
        output_path = self.output_matrix_dir / f"{plot_type}_matrix_comparison.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Matrix saved: {output_path}")
        return output_path
    
    def create_scenario_matrix(self, scenario: str, plot_types: Optional[List[str]] = None,
                              max_plots_per_row: int = 4, figsize: Tuple[int, int] = (20, 15),
                              crop_legend: bool = True) -> Path:
        """
        Create a matrix showing all plot types for one scenario.
        
        Args:
            scenario: Scenario name
            plot_types: List of plot types to include (None = all available)
            max_plots_per_row: Maximum number of plots per row
            figsize: Figure size (width, height)
            crop_legend: Whether to crop out legends from individual plots
            
        Returns:
            Path to saved matrix image
        """
        if plot_types is None:
            plot_types = self.plot_types
        
        if scenario not in self.available_scenarios:
            raise ValueError(f"Scenario '{scenario}' not found. Available scenarios: {self.available_scenarios}")
        
        # Filter plot types that exist for this scenario
        plots_dir = self._get_scenario_plots_dir(scenario)
        valid_plot_types = []
        for plot_type in plot_types:
            plot_path = plots_dir / f"{plot_type}.png"
            if plot_path.exists():
                valid_plot_types.append(plot_type)
        
        if not valid_plot_types:
            raise ValueError(f"No plots found for scenario '{scenario}'")
        
        # Calculate grid dimensions
        n_plots = len(valid_plot_types)
        n_cols = min(max_plots_per_row, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Load and display each plot
        for i, plot_type in enumerate(valid_plot_types):
            ax = axes[i]
            
            # Load the plot image
            plot_path = plots_dir / f"{plot_type}.png"
            
            try:
                img = self._load_and_crop_image(plot_path, crop_legend)
                ax.imshow(img)
                ax.set_title(f"{self._plot_type_to_title(plot_type)}", 
                           fontsize=12, fontweight='bold', pad=10)
                ax.axis('off')
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading plot:\n{str(e)}", 
                       ha='center', va='center', transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
                ax.set_title(f"{self._plot_type_to_title(plot_type)} (ERROR)", 
                           fontsize=12, fontweight='bold', color='red')
                ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')
        
        # Set main title
        scenario_title = self._scenario_name_to_title(scenario)
        fig.suptitle(f"{scenario_title} - All Plots Matrix", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save the matrix
        self.output_matrix_dir.mkdir(exist_ok=True)
        output_path = self.output_matrix_dir / f"{scenario}_all_plots_matrix.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Matrix saved: {output_path}")
        return output_path
    
    def create_custom_matrix(self, scenarios: List[str], plot_types: List[str],
                           figsize: Tuple[int, int] = (24, 18), crop_legend: bool = True) -> Path:
        """
        Create a custom matrix with specified scenarios and plot types.
        
        Args:
            scenarios: List of scenarios to include
            plot_types: List of plot types to include
            figsize: Figure size (width, height)
            crop_legend: Whether to crop out legends from individual plots
            
        Returns:
            Path to saved matrix image
        """
        # Validate inputs
        invalid_scenarios = [s for s in scenarios if s not in self.available_scenarios]
        if invalid_scenarios:
            raise ValueError(f"Invalid scenarios: {invalid_scenarios}. Available: {self.available_scenarios}")
        
        invalid_plot_types = [p for p in plot_types if p not in self.plot_types]
        if invalid_plot_types:
            raise ValueError(f"Invalid plot types: {invalid_plot_types}. Available: {self.plot_types}")
        
        # Create grid
        n_scenarios = len(scenarios)
        n_plot_types = len(plot_types)
        
        fig, axes = plt.subplots(n_plot_types, n_scenarios, figsize=figsize)
        
        # Handle single row/column cases
        if n_plot_types == 1:
            axes = axes.reshape(1, -1)
        if n_scenarios == 1:
            axes = axes.reshape(-1, 1)
        
        # Fill the matrix
        for i, plot_type in enumerate(plot_types):
            for j, scenario in enumerate(scenarios):
                ax = axes[i, j]
                
                # Load the plot image
                plots_dir = self._get_scenario_plots_dir(scenario)
                plot_path = plots_dir / f"{plot_type}.png"
                
                if plot_path.exists():
                    try:
                        img = self._load_and_crop_image(plot_path, crop_legend)
                        ax.imshow(img)
                        ax.axis('off')
                    except Exception as e:
                        ax.text(0.5, 0.5, f"Error loading plot:\n{str(e)}", 
                               ha='center', va='center', transform=ax.transAxes,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
                        ax.axis('off')
                else:
                    ax.text(0.5, 0.5, "Plot not found", 
                           ha='center', va='center', transform=ax.transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
                    ax.axis('off')
                
                # Add titles
                if i == 0:  # Top row - scenario names
                    ax.set_title(f"{self._scenario_name_to_title(scenario)}", 
                               fontsize=12, fontweight='bold', pad=10)
                if j == 0:  # Left column - plot type names
                    ax.set_ylabel(f"{self._plot_type_to_title(plot_type)}", 
                                fontsize=12, fontweight='bold', rotation=0, ha='right', va='center')
        
        # Set main title
        fig.suptitle(f"Custom Comparison Matrix: {len(plot_types)} Plot Types × {len(scenarios)} Scenarios", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, left=0.15)
        
        # Save the matrix
        self.output_matrix_dir.mkdir(exist_ok=True)
        output_path = self.output_matrix_dir / f"custom_matrix_{len(plot_types)}x{len(scenarios)}.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Matrix saved: {output_path}")
        return output_path
    
    def list_available_options(self):
        """Print available scenarios and plot types."""
        print("Available Scenarios:")
        for i, scenario in enumerate(self.available_scenarios, 1):
            print(f"  {i:2d}. {scenario}")
        
        print(f"\nAvailable Plot Types ({len(self.plot_types)}):")
        for i, plot_type in enumerate(self.plot_types, 1):
            print(f"  {i:2d}. {plot_type}")
        
        print(f"\nTotal scenarios: {len(self.available_scenarios)}")
        print(f"Total plot types: {len(self.plot_types)}")


def main():
    """Command-line interface for the plot matrix generator."""
    parser = argparse.ArgumentParser(description="Generate plot comparison matrices from existing scenario plots")
    
    parser.add_argument('--list', action='store_true', 
                       help='List available scenarios and plot types')
    
    parser.add_argument('--plot-type', type=str, 
                       help='Create matrix for specific plot type across scenarios')
    
    parser.add_argument('--scenario', type=str,
                       help='Create matrix for specific scenario across plot types')
    
    parser.add_argument('--scenarios', nargs='+',
                       help='List of scenarios for custom matrix')
    
    parser.add_argument('--plot-types', nargs='+',
                       help='List of plot types for custom matrix')
    
    parser.add_argument('--max-per-row', type=int, default=4,
                       help='Maximum plots/scenarios per row (default: 4)')
    
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory containing scenarios (default: output)')
    
    parser.add_argument('--keep-legends', action='store_true',
                       help='Keep legends in individual plots (default: crop legends)')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = PlotMatrixGenerator(args.output_dir)
    
    # List options if requested
    if args.list:
        generator.list_available_options()
        return
    
    # Determine legend cropping setting
    crop_legend = not args.keep_legends
    
    # Generate matrices based on arguments
    if args.plot_type:
        # Matrix for one plot type across scenarios
        generator.create_plot_type_matrix(args.plot_type, max_scenarios_per_row=args.max_per_row, crop_legend=crop_legend)
        
    elif args.scenario:
        # Matrix for one scenario across plot types
        generator.create_scenario_matrix(args.scenario, max_plots_per_row=args.max_per_row, crop_legend=crop_legend)
        
    elif args.scenarios and args.plot_types:
        # Custom matrix
        generator.create_custom_matrix(args.scenarios, args.plot_types, crop_legend=crop_legend)
        
    else:
        # Default: create some useful matrices
        print("No specific matrix requested. Creating some useful default matrices...")
        
        # Create matrices for key plot types
        key_plot_types = ['total_carbon_stocks_all_scenarios', 'additionality', 'carbon_pools_comparison']
        
        for plot_type in key_plot_types:
            try:
                generator.create_plot_type_matrix(plot_type, crop_legend=crop_legend)
            except Exception as e:
                print(f"Could not create matrix for {plot_type}: {e}")
        
        # Create matrix for first scenario (if available)
        if generator.available_scenarios:
            try:
                generator.create_scenario_matrix(generator.available_scenarios[0], crop_legend=crop_legend)
            except Exception as e:
                print(f"Could not create scenario matrix: {e}")


if __name__ == "__main__":
    main()
