"""Color management system for Forest Carbon Lite visualizations."""

from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class ColorManager:
    """Manages color schemes for consistent visualization across all plots."""
    
    def __init__(self):
        """Initialize color manager with predefined color schemes."""
        self._initialize_color_palettes()
    
    def _initialize_color_palettes(self):
        """Initialize all color palettes."""
        
        # Scenario colors - consistent across all plots
        self.scenario_colors = {
            'baseline': '#e74c3c',      # Red - represents degraded/damaged state
            'management': '#3498db',     # Blue - represents active/adaptive management
            'reforestation': '#2ecc71'   # Green - represents restoration/growth
        }
        
        # Forest type colors - for multi-forest comparisons
        self.forest_type_colors = {
            'EOF': '#8e44ad',           # Purple - Eucalypt Open Forest
            'ETOF': '#f39c12',          # Orange - Eucalypt Tall Open Forest
            'default': '#34495e'        # Dark gray - fallback
        }
        
        # Extended palette for future forest types and custom configs
        self.extended_palette = [
            '#e74c3c',  # Red
            '#3498db',  # Blue  
            '#2ecc71',  # Green
            '#8e44ad',  # Purple
            '#f39c12',  # Orange
            '#1abc9c',  # Turquoise
            '#e67e22',  # Carrot
            '#9b59b6',  # Amethyst
            '#34495e',  # Wet Asphalt
            '#16a085',  # Green Sea
            '#27ae60',  # Nephritis
            '#2980b9',  # Belize Hole
            '#8e44ad',  # Wisteria
            '#2c3e50',  # Midnight Blue
            '#f1c40f',  # Sun Flower
            '#e67e22',  # Carrot
            '#e74c3c',  # Alizarin
            '#95a5a6',  # Concrete
            '#7f8c8d',  # Asbestos
            '#bdc3c7'   # Silver
        ]
        
        # Carbon pool colors - for detailed breakdowns
        self.carbon_pool_colors = {
            'agb': '#27ae60',           # Green - Above-ground biomass
            'bgb': '#16a085',           # Teal - Below-ground biomass
            'litter': '#f39c12',        # Orange - Litter
            'active_soil': '#8e44ad',   # Purple - Active soil
            'slow_soil': '#34495e',     # Dark gray - Slow soil
            'char': '#2c3e50',          # Charcoal - Charcoal
            'slash': '#e67e22',         # Carrot - Slash
            'hwp_short': '#e74c3c',     # Red - Short-lived HWP
            'hwp_medium': '#3498db',    # Blue - Medium-lived HWP
            'hwp_long': '#2ecc71'       # Green - Long-lived HWP
        }
        
        # Economic colors
        self.economic_colors = {
            'revenue': '#27ae60',       # Green - positive cash flow
            'costs': '#e74c3c',         # Red - negative cash flow
            'net': '#2c3e50',           # Dark - net cash flow
            'npv': '#8e44ad',           # Purple - NPV
            'irr': '#f39c12',           # Orange - IRR
            'carbon_price': '#3498db'   # Blue - carbon pricing
        }
        
        # Line styles for different scenarios
        self.scenario_styles = {
            'baseline': {'linestyle': '-', 'linewidth': 2.5, 'alpha': 0.8},
            'management': {'linestyle': '-', 'linewidth': 2.5, 'alpha': 0.8},
            'reforestation': {'linestyle': '--', 'linewidth': 2.5, 'alpha': 0.8}
        }
        
        # Marker styles for different scenarios
        self.scenario_markers = {
            'baseline': 'o',
            'management': 's', 
            'reforestation': '^'
        }
    
    def get_scenario_color(self, scenario: str) -> str:
        """Get color for a specific scenario."""
        return self.scenario_colors.get(scenario, self.extended_palette[0])
    
    def get_forest_type_color(self, forest_type: str) -> str:
        """Get color for a specific forest type."""
        return self.forest_type_colors.get(forest_type, self.forest_type_colors['default'])
    
    def get_carbon_pool_color(self, pool: str) -> str:
        """Get color for a specific carbon pool."""
        return self.carbon_pool_colors.get(pool, self.extended_palette[0])
    
    def get_economic_color(self, economic_type: str) -> str:
        """Get color for economic analysis elements."""
        return self.economic_colors.get(economic_type, self.extended_palette[0])
    
    def get_scenario_style(self, scenario: str) -> Dict:
        """Get complete style dictionary for a scenario."""
        base_style = self.scenario_styles.get(scenario, {'linestyle': '-', 'linewidth': 2.5, 'alpha': 0.8})
        base_style['color'] = self.get_scenario_color(scenario)
        return base_style
    
    def get_scenario_marker(self, scenario: str) -> str:
        """Get marker style for a scenario."""
        return self.scenario_markers.get(scenario, 'o')
    
    def get_color_sequence(self, n_colors: int, palette: str = 'extended') -> List[str]:
        """Get a sequence of colors for multiple elements.
        
        Args:
            n_colors: Number of colors needed
            palette: Which palette to use ('extended', 'scenario', 'forest_type')
        """
        if palette == 'scenario':
            colors = list(self.scenario_colors.values())
        elif palette == 'forest_type':
            colors = list(self.forest_type_colors.values())
        else:  # extended
            colors = self.extended_palette
        
        # Cycle through colors if we need more than available
        if n_colors <= len(colors):
            return colors[:n_colors]
        else:
            # Repeat colors if needed
            return [colors[i % len(colors)] for i in range(n_colors)]
    
    def get_contrasting_colors(self, n_colors: int) -> List[str]:
        """Get a set of maximally contrasting colors."""
        # Use matplotlib's tab10 palette for good contrast
        cmap = plt.cm.tab10
        return [cmap(i) for i in range(n_colors)]
    
    def add_custom_forest_type(self, forest_type: str, color: str):
        """Add a custom forest type with its color."""
        self.forest_type_colors[forest_type] = color
    
    def add_custom_scenario(self, scenario: str, color: str, style: Optional[Dict] = None):
        """Add a custom scenario with its color and style."""
        self.scenario_colors[scenario] = color
        if style:
            self.scenario_styles[scenario] = style
    
    def get_color_info(self) -> Dict:
        """Get information about all available colors."""
        return {
            'scenario_colors': self.scenario_colors,
            'forest_type_colors': self.forest_type_colors,
            'carbon_pool_colors': self.carbon_pool_colors,
            'economic_colors': self.economic_colors,
            'extended_palette': self.extended_palette,
            'scenario_styles': self.scenario_styles,
            'scenario_markers': self.scenario_markers
        }
    
    def print_color_palette(self):
        """Print a visual representation of the color palette."""
        print("Forest Carbon Lite - Color Palette")
        print("=" * 50)
        
        print("\nScenario Colors:")
        for scenario, color in self.scenario_colors.items():
            print(f"  {scenario:12}: {color}")
        
        print("\nForest Type Colors:")
        for forest_type, color in self.forest_type_colors.items():
            print(f"  {forest_type:12}: {color}")
        
        print("\nCarbon Pool Colors:")
        for pool, color in self.carbon_pool_colors.items():
            print(f"  {pool:12}: {color}")
        
        print("\nEconomic Colors:")
        for econ_type, color in self.economic_colors.items():
            print(f"  {econ_type:12}: {color}")
        
        print(f"\nExtended Palette ({len(self.extended_palette)} colors):")
        for i, color in enumerate(self.extended_palette):
            if i % 5 == 0:
                print()
            print(f"  {color:8}", end="")
        print()

# Global color manager instance
color_manager = ColorManager()