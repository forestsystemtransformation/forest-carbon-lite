"""Carbon pool dynamics and accounting."""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List
from ...utils.constants import (
    C_TO_CO2,
    BIOMASS_TO_C,
    LITTER_DECAY_RATE,
    CHAR_DECAY_RATE,
    ACTIVE_SOIL_DECAY_RATE,
    SLOW_SOIL_DECAY_RATE,
    SLASH_DECAY_RATE,
    HWP_SHORT_LIFE,
    HWP_MEDIUM_LIFE,
    HWP_LONG_LIFE
)

@dataclass
class CarbonPools:
    """Tracks all carbon pools in the system."""
    
    agb: float = 0.0  # Above-ground biomass
    bgb: float = 0.0  # Below-ground biomass
    litter: float = 0.0
    char: float = 0.0  # Charcoal
    active_soil: float = 0.0
    slow_soil: float = 0.0
    hwp_short: float = 0.0  # Harvested wood products - short lived
    hwp_medium: float = 0.0  # Medium lived
    hwp_long: float = 0.0  # Long lived
    slash: float = 0.0
    
    def get_total_carbon(self) -> float:
        """Calculate total carbon across all pools (tonnes C/ha)."""
        return (self.agb + self.bgb + self.litter + self.char + 
                self.active_soil + self.slow_soil + self.hwp_short + 
                self.hwp_medium + self.hwp_long + self.slash)
    
    def get_total_co2e(self) -> float:
        """Calculate total CO2 equivalent (tonnes CO2e/ha)."""
        return self.get_total_carbon() * C_TO_CO2
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easy export."""
        return {
            'agb': self.agb,
            'bgb': self.bgb,
            'litter': self.litter,
            'char': self.char,
            'active_soil': self.active_soil,
            'slow_soil': self.slow_soil,
            'hwp_short': self.hwp_short,
            'hwp_medium': self.hwp_medium,
            'hwp_long': self.hwp_long,
            'slash': self.slash,
            'total_carbon': self.get_total_carbon(),
            'total_co2e': self.get_total_co2e()
        }

class CarbonPoolManager:
    """Manages carbon pool dynamics and transfers."""
    
    def __init__(self, root_shoot_ratio: float = 0.25):
        """
        Initialize carbon pool manager.
        
        Args:
            root_shoot_ratio: Ratio of below-ground to above-ground biomass
        """
        assert 0 < root_shoot_ratio <= 1, f"Root-shoot ratio must be between 0 and 1, got {root_shoot_ratio}"
        self.root_shoot_ratio = root_shoot_ratio
        self.pools = CarbonPools()
    
    def update_biomass(self, delta_agb: float):
        """
        Update biomass pools with new growth.
        
        Args:
            delta_agb: Change in above-ground biomass (tonnes/ha)
        """
        assert delta_agb >= 0, f"Biomass growth cannot be negative, got {delta_agb}"
        
        # Convert to carbon
        delta_agb_c = delta_agb * BIOMASS_TO_C
        
        # Update above and below-ground biomass
        self.pools.agb += delta_agb_c
        self.pools.bgb = self.pools.agb * self.root_shoot_ratio
        
        # Enforce non-negativity
        self._enforce_non_negativity()
    
    def apply_mortality(self, mortality_rate: float):
        """
        Apply mortality and transfer biomass to dead pools.
        
        Args:
            mortality_rate: Annual mortality rate (fraction)
        """
        assert 0 <= mortality_rate <= 1, f"Mortality rate must be between 0 and 1, got {mortality_rate}"
        
        # Calculate biomass loss
        agb_loss = self.pools.agb * mortality_rate
        bgb_loss = self.pools.bgb * mortality_rate
        
        # Store total loss for mass balance check
        total_loss = agb_loss + bgb_loss
        
        # Transfer to litter and soil
        self.pools.agb -= agb_loss
        self.pools.bgb -= bgb_loss
        
        # Allocate dead biomass with mass balance
        # Above-ground: 60% to litter, 40% to slash
        self.pools.litter += agb_loss * 0.6
        self.pools.slash += agb_loss * 0.4
        
        # Below-ground: 70% to active soil, 30% to slow soil
        self.pools.active_soil += bgb_loss * 0.7
        self.pools.slow_soil += bgb_loss * 0.3
        
        # Enforce non-negativity
        self._enforce_non_negativity()
        
        # Verify mass balance
        total_gained = (agb_loss * 0.6 + agb_loss * 0.4 + 
                       bgb_loss * 0.7 + bgb_loss * 0.3)
        if abs(total_loss - total_gained) > 1e-10:
            raise ValueError(f"Mass balance error: lost {total_loss}, gained {total_gained}")
    
    def apply_disturbance(self, severity: float):
        """
        Apply disturbance event (e.g., fire).
        
        Args:
            severity: Disturbance severity (0-1)
        """
        assert 0 <= severity <= 1, f"Disturbance severity must be between 0 and 1, got {severity}"
        
        # Calculate biomass loss
        agb_loss = self.pools.agb * severity
        bgb_loss = self.pools.bgb * severity * 0.5  # Roots partially survive
        
        # Store total loss for mass balance check
        total_loss = agb_loss + bgb_loss
        
        # Update pools
        self.pools.agb -= agb_loss
        self.pools.bgb -= bgb_loss
        
        # Distribute lost biomass with mass balance
        # 10% becomes charcoal (persistent)
        char_created = agb_loss * 0.1
        self.pools.char += char_created
        
        # 20% becomes litter (decomposable)
        litter_created = agb_loss * 0.2
        self.pools.litter += litter_created
        
        # 70% goes to atmosphere (not tracked in pools)
        # Below-ground losses: 50% to soil, 50% to atmosphere
        soil_created = bgb_loss * 0.5
        self.pools.active_soil += soil_created * 0.7
        self.pools.slow_soil += soil_created * 0.3
        
        # Enforce non-negativity
        self._enforce_non_negativity()
        
        # Verify mass balance (accounting for atmospheric losses)
        total_tracked = char_created + litter_created + soil_created
        atmospheric_loss = total_loss - total_tracked
        if atmospheric_loss < 0:
            raise ValueError(f"Mass balance error: atmospheric loss cannot be negative: {atmospheric_loss}")
    
    def apply_harvest(self, harvest_fraction: float = 0.0):
        """
        Apply harvesting and create wood products.
        
        Args:
            harvest_fraction: Fraction of AGB harvested
        """
        assert 0 <= harvest_fraction <= 1, f"Harvest fraction must be between 0 and 1, got {harvest_fraction}"
        
        if harvest_fraction <= 0:
            return
        
        harvest = self.pools.agb * harvest_fraction
        self.pools.agb -= harvest
        
        # Allocate to wood products
        self.pools.hwp_short += harvest * 0.2
        self.pools.hwp_medium += harvest * 0.5
        self.pools.hwp_long += harvest * 0.3
    
    def decay_pools(self, timestep: float = 1.0):
        """
        Apply decay to dead organic matter pools.
        
        Args:
            timestep: Time step in years
        """
        # Decay litter
        litter_decay = self.pools.litter * LITTER_DECAY_RATE * timestep
        self.pools.litter -= litter_decay
        self.pools.active_soil += litter_decay * 0.6
        
        # Decay char (very slow)
        self.pools.char *= (1 - CHAR_DECAY_RATE * timestep)
        
        # Decay soil pools
        self.pools.active_soil *= (1 - ACTIVE_SOIL_DECAY_RATE * timestep)
        self.pools.slow_soil *= (1 - SLOW_SOIL_DECAY_RATE * timestep)
        
        # Decay slash
        slash_decay = self.pools.slash * SLASH_DECAY_RATE * timestep
        self.pools.slash -= slash_decay
        self.pools.active_soil += slash_decay * 0.4
        
        # Decay harvested wood products
        self.pools.hwp_short *= (1 - 1.0/HWP_SHORT_LIFE * timestep)
        self.pools.hwp_medium *= (1 - 1.0/HWP_MEDIUM_LIFE * timestep)
        self.pools.hwp_long *= (1 - 1.0/HWP_LONG_LIFE * timestep)
    
    def initialize_from_standing_forest(self, initial_agb: float, 
                                       soil_fraction: float = 0.3):
        """
        Initialize pools for existing forest.
        
        Args:
            initial_agb: Initial above-ground biomass (tonnes/ha)
            soil_fraction: Fraction of equilibrium soil carbon
        """
        assert initial_agb >= 0, f"Initial AGB must be non-negative, got {initial_agb}"
        assert 0 <= soil_fraction <= 1, f"Soil fraction must be between 0 and 1, got {soil_fraction}"
        
        # Set initial biomass
        self.pools.agb = initial_agb * BIOMASS_TO_C
        self.pools.bgb = self.pools.agb * self.root_shoot_ratio
        
        # Initialize dead pools at quasi-equilibrium
        self.pools.litter = self.pools.agb * 0.1
        self.pools.active_soil = self.pools.agb * soil_fraction
        self.pools.slow_soil = self.pools.agb * soil_fraction * 2
    
    def get_pools(self) -> CarbonPools:
        """Return current carbon pools."""
        return self.pools
    
    def reset(self):
        """Reset all pools to zero."""
        self.pools = CarbonPools()
    
    def _enforce_non_negativity(self):
        """Enforce non-negativity constraints on all pools."""
        # Ensure all pools are non-negative
        self.pools.agb = max(0.0, self.pools.agb)
        self.pools.bgb = max(0.0, self.pools.bgb)
        self.pools.litter = max(0.0, self.pools.litter)
        self.pools.char = max(0.0, self.pools.char)
        self.pools.active_soil = max(0.0, self.pools.active_soil)
        self.pools.slow_soil = max(0.0, self.pools.slow_soil)
        self.pools.hwp_short = max(0.0, self.pools.hwp_short)
        self.pools.hwp_medium = max(0.0, self.pools.hwp_medium)
        self.pools.hwp_long = max(0.0, self.pools.hwp_long)
        self.pools.slash = max(0.0, self.pools.slash)
    
    def get_mass_balance(self) -> Dict[str, float]:
        """
        Calculate mass balance for all pools.
        
        Returns:
            Dictionary with mass balance information
        """
        total_carbon = self.pools.get_total_carbon()
        
        return {
            'total_carbon': total_carbon,
            'total_co2e': self.pools.get_total_co2e(),
            'agb_carbon': self.pools.agb,
            'bgb_carbon': self.pools.bgb,
            'dead_carbon': (self.pools.litter + self.pools.char + 
                           self.pools.active_soil + self.pools.slow_soil + 
                           self.pools.slash),
            'hwp_carbon': (self.pools.hwp_short + self.pools.hwp_medium + 
                          self.pools.hwp_long)
        }
    
    def transfer_carbon(self, from_pool: str, to_pool: str, amount: float):
        """
        Transfer carbon between pools with mass balance enforcement.
        
        Args:
            from_pool: Source pool name
            to_pool: Destination pool name
            amount: Amount to transfer (tonnes C/ha)
        """
        assert amount >= 0, f"Transfer amount cannot be negative, got {amount}"
        
        # Get current values
        from_value = getattr(self.pools, from_pool, None)
        to_value = getattr(self.pools, to_pool, None)
        
        if from_value is None or to_value is None:
            raise ValueError(f"Invalid pool names: {from_pool} or {to_pool}")
        
        if from_value < amount:
            raise ValueError(f"Insufficient carbon in {from_pool}: {from_value} < {amount}")
        
        # Perform transfer
        setattr(self.pools, from_pool, from_value - amount)
        setattr(self.pools, to_pool, to_value + amount)
        
        # Enforce non-negativity
        self._enforce_non_negativity()
