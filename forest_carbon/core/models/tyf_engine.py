"""Tree Yield Formula (TYF) implementation for biomass growth modeling."""

import numpy as np
from typing import Tuple, Optional, Union, Dict, Any

class TYFEngine:
    """Implements FullCAM Tree Yield Formula for biomass growth."""
    
    def __init__(self, M: float, G: float, y: float = 1.0, FPI_ratio: Union[float, Dict[str, Any]] = 1.0):
        """
        Initialize TYF engine with growth parameters.
        
        Args:
            M: Maximum potential AGB (tonnes/hectare)
            G: Age of maximum growth rate (years)
            y: Type-2 multiplier for planting factors
            FPI_ratio: Forest Productivity Index ratio (can be constant or time-varying config)
        """
        assert M > 0, f"Maximum potential AGB (M) must be positive, got {M}"
        assert G > 0, f"Age of maximum growth rate (G) must be positive, got {G}"
        assert y > 0, f"Type-2 multiplier (y) must be positive, got {y}"
        
        self.M = M
        self.G = G
        self.y = y
        self.FPI_ratio = FPI_ratio
        self.k = 2 * G - 1.25  # Growth curve parameter
    
    def get_fpi_ratio(self, year: int) -> float:
        """
        Get FPI ratio for a specific year, handling both constant and time-varying configurations.
        
        Args:
            year: Simulation year (0-based)
            
        Returns:
            FPI ratio for the given year
        """
        if isinstance(self.FPI_ratio, (int, float)):
            # Constant FPI ratio
            return float(self.FPI_ratio)
        elif isinstance(self.FPI_ratio, dict):
            # Time-varying FPI ratio
            # Check for year-specific value first
            year_key = f"year_{year}"
            if year_key in self.FPI_ratio:
                return float(self.FPI_ratio[year_key])
            # Fall back to default
            return float(self.FPI_ratio.get('default', 1.0))
        else:
            # Fallback to 1.0 if invalid configuration
            return 1.0
    
    def calculate_delta_agb(self, age1: float, age2: float, year: Optional[int] = None) -> float:
        """
        Calculate change in Above-Ground Biomass over time step.
        
        Ensures monotonic accumulation until asymptote and numerical stability at age 0.
        
        Args:
            age1: Age at beginning of time step (years)
            age2: Age at end of time step (years)
            year: Simulation year for time-varying FPI ratio (optional)
            
        Returns:
            Change in AGB (tonnes/hectare)
        """
        assert age1 >= 0, f"Age1 must be non-negative, got {age1}"
        assert age2 >= 0, f"Age2 must be non-negative, got {age2}"
        assert age2 >= age1, f"Age2 must be >= age1, got age1={age1}, age2={age2}"
        
        if age1 < 0 or age2 <= age1:
            return 0.0
        
        # Prevent division by zero for very young stands
        if age2 < 0.1:
            age2 = 0.1
        if age1 < 0.1 and age1 > 0:
            age1 = 0.1
        
        # Get FPI ratio for the current year
        fpi_ratio = self.get_fpi_ratio(year) if year is not None else self.get_fpi_ratio(0)
        
        # TYF equation
        exp_term2 = np.exp(-self.k / age2)
        exp_term1 = np.exp(-self.k / age1) if age1 > 0 else 0
        
        delta_agb = self.y * self.M * (exp_term2 - exp_term1) * fpi_ratio
        
        return max(0, delta_agb)  # Ensure non-negative growth
    
    def calculate_total_agb(self, age: float) -> float:
        """
        Calculate total AGB at given age.
        
        Args:
            age: Stand age (years)
            
        Returns:
            Total AGB (tonnes/hectare)
        """
        assert age >= 0, f"Age must be non-negative, got {age}"
        
        if age <= 0:
            return 0.0
        
        # Calculate from age 0 to current age
        return self.calculate_delta_agb(0, age)
    
    def get_growth_rate(self, age: float) -> float:
        """
        Calculate instantaneous growth rate at given age.
        
        Args:
            age: Stand age (years)
            
        Returns:
            Growth rate (tonnes/hectare/year)
        """
        assert age >= 0, f"Age must be non-negative, got {age}"
        
        if age <= 0:
            return 0.0
        
        # Approximate derivative using small time step
        dt = 0.1
        return self.calculate_delta_agb(age, age + dt) / dt
    
    def get_parameters(self, year: Optional[int] = None) -> dict:
        """
        Return current TYF parameters.
        
        Args:
            year: Simulation year for time-varying FPI ratio (optional)
        
        Returns:
            Dictionary of TYF parameters
        """
        fpi_ratio = self.get_fpi_ratio(year) if year is not None else self.get_fpi_ratio(0)
        return {
            'M': self.M,
            'G': self.G,
            'y': self.y,
            'k': self.k,
            'FPI_ratio': fpi_ratio,
            'FPI_config': self.FPI_ratio
        }
    
    def get_maximum_growth_age(self) -> float:
        """
        Calculate the age at which growth rate is maximum.
        
        Returns:
            Age at maximum growth rate (years)
        """
        return self.G
    
    def get_asymptotic_biomass(self, year: Optional[int] = None) -> float:
        """
        Calculate the asymptotic (maximum) biomass.
        
        Args:
            year: Simulation year for time-varying FPI ratio (optional)
        
        Returns:
            Maximum possible biomass (tonnes/hectare)
        """
        fpi_ratio = self.get_fpi_ratio(year) if year is not None else self.get_fpi_ratio(0)
        return self.y * self.M * fpi_ratio
    
    def get_biomass_at_age(self, age: float) -> float:
        """
        Alias for calculate_total_agb for clarity.
        
        Args:
            age: Stand age (years)
            
        Returns:
            Total AGB (tonnes/hectare)
        """
        assert age >= 0, f"Age must be non-negative, got {age}"
        return self.calculate_total_agb(age)
    
    def validate_monotonicity(self, max_age: float = 200.0, step: float = 1.0) -> bool:
        """
        Validate that biomass accumulation is monotonic (always increasing).
        
        Args:
            max_age: Maximum age to test
            step: Age step for testing
            
        Returns:
            True if monotonic, False otherwise
        """
        assert max_age > 0, f"Max age must be positive, got {max_age}"
        assert step > 0, f"Step must be positive, got {step}"
        
        ages = np.arange(0, max_age + step, step)
        biomass_values = [self.calculate_total_agb(age) for age in ages]
        
        # Check if all values are non-decreasing
        for i in range(1, len(biomass_values)):
            if biomass_values[i] < biomass_values[i-1]:
                return False
        
        return True
