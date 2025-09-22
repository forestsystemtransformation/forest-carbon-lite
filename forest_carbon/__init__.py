"""
Forest Carbon Lite - Dynamic Carbon Sequestration Simulator

A comprehensive Python implementation of a "FullCAM-lite" forest carbon accounting model,
featuring the Tree Yield Formula (TYF) growth engine with dynamic scenario generation
and climate change integration.
"""

__version__ = "8.0.0"
__author__ = "Pia Angelike, Healthy Forests Foundation"
__email__ = "info@healthyforests.org"

# Main imports for easy access
from .core.simulator import ForestCarbonSimulator
from .scenarios.manager import ScenarioManager
from .analysis.comprehensive import ComprehensiveAnalyzer

__all__ = [
    "ForestCarbonSimulator",
    "ScenarioManager", 
    "ComprehensiveAnalyzer"
]
