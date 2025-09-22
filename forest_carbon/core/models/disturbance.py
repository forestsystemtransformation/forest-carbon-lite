"""Stochastic disturbance and mortality modeling."""

import numpy as np
from typing import Optional, Tuple

class DisturbanceModel:
    """Models chronic mortality and stochastic disturbance events."""
    
    def __init__(self, chronic_mortality: float = 0.01,
                 disturbance_probability: float = 0.05,
                 disturbance_severity: float = 0.2,
                 seed: Optional[int] = None):
        """
        Initialize disturbance model.
        
        Args:
            chronic_mortality: Annual mortality rate
            disturbance_probability: Annual probability of disturbance
            disturbance_severity: Severity when disturbance occurs (0-1)
            seed: Random seed for reproducibility
        """
        self.chronic_mortality = chronic_mortality
        self.disturbance_probability = disturbance_probability
        self.disturbance_severity = disturbance_severity
        
        # Use numpy's modern RNG for better reproducibility
        self.rng = np.random.default_rng(seed)
    
    def get_annual_mortality(self) -> float:
        """Return chronic mortality rate."""
        return self.chronic_mortality
    
    def check_disturbance(self) -> Tuple[bool, float]:
        """
        Check if disturbance occurs and return severity.
        
        Returns:
            Tuple of (happened: bool, severity: float) where severity is in [0, 1]
        """
        # Check if disturbance occurs using instance RNG
        if self.rng.random() < self.disturbance_probability:
            # Add variability to severity using beta distribution for realistic values
            # Beta distribution ensures values stay in [0, 1] range
            alpha = 2.0  # Shape parameter for moderate variability
            beta = 2.0   # Shape parameter for moderate variability
            
            # Scale the beta distribution to match our expected severity
            base_severity = self.rng.beta(alpha, beta)
            
            # Adjust to match our target severity with some variability
            severity = self.disturbance_severity * (0.5 + base_severity)
            severity = min(1.0, max(0.0, severity))  # Ensure [0, 1] range
            
            return True, severity
        
        return False, 0.0
    
    def simulate_year(self) -> dict:
        """
        Simulate one year of disturbance.
        
        Returns:
            Dictionary with mortality and disturbance info
        """
        disturbance_occurred, severity = self.check_disturbance()
        
        return {
            'chronic_mortality': self.chronic_mortality,
            'disturbance_occurred': disturbance_occurred,
            'disturbance_severity': severity if disturbance_occurred else 0.0,
            'total_mortality': self.chronic_mortality + (severity if disturbance_occurred else 0.0)
        }
    
    def update_parameters(self, **kwargs):
        """Update disturbance parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_parameters(self) -> dict:
        """Return current disturbance parameters."""
        return {
            'chronic_mortality': self.chronic_mortality,
            'disturbance_probability': self.disturbance_probability,
            'disturbance_severity': self.disturbance_severity
        }
    
    def validate_parameters(self) -> bool:
        """
        Validate that all parameters are within reasonable ranges.
        
        Returns:
            True if parameters are valid, False otherwise
        """
        if not (0 <= self.chronic_mortality <= 1):
            return False
        if not (0 <= self.disturbance_probability <= 1):
            return False
        if not (0 <= self.disturbance_severity <= 1):
            return False
        return True
    
    def get_expected_annual_mortality(self) -> float:
        """
        Calculate expected annual mortality including both chronic and disturbance.
        
        Returns:
            Expected annual mortality rate
        """
        # Expected disturbance mortality = P(disturbance) * E(severity)
        expected_disturbance_mortality = (self.disturbance_probability * 
                                        self.disturbance_severity * 0.75)  # 0.75 is mean of beta(2,2)
        
        return self.chronic_mortality + expected_disturbance_mortality
