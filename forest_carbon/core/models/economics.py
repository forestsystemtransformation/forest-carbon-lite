"""Economic analysis for carbon projects."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

class EconomicsModel:
    """Economic analysis including NPV, IRR, and carbon crediting."""
    
    def __init__(self, config: Dict):
        """
        Initialize economics model with configuration.
        
        Args:
            config: Economics configuration dictionary
        """
        assert config, "Economics configuration cannot be empty"
        assert 'carbon' in config, "Economics configuration must contain 'carbon' section"
        assert 'costs' in config, "Economics configuration must contain 'costs' section"
        
        self.carbon_config = config.get('carbon', {})
        self.costs_config = config.get('costs', {})
        
        # Carbon pricing
        self.price_start = self.carbon_config.get('price_start', 35.0)
        self.price_growth = self.carbon_config.get('price_growth', 0.03)
        self.buffer = self.carbon_config.get('buffer', 0.20)
        self.crediting_years = self.carbon_config.get('crediting_years', 30)
        self.discount_rate = self.carbon_config.get('discount_rate', 0.07)
        
        # Validate parameters
        assert self.price_start >= 0, f"Carbon price must be non-negative, got {self.price_start}"
        assert 0 <= self.price_growth <= 1, f"Price growth must be between 0 and 1, got {self.price_growth}"
        assert 0 <= self.buffer <= 1, f"Buffer must be between 0 and 1, got {self.buffer}"
        assert self.crediting_years > 0, f"Crediting years must be positive, got {self.crediting_years}"
        assert 0 <= self.discount_rate <= 1, f"Discount rate must be between 0 and 1, got {self.discount_rate}"
    
    def calculate_carbon_price(self, year: int) -> float:
        """
        Calculate carbon price for given year using continuous compounding.
        
        Uses continuous compounding for more realistic price growth modeling.
        
        Args:
            year: Year of simulation
            
        Returns:
            Carbon price ($/tCO2e)
        """
        # Continuous compounding: P(t) = P0 * exp(r * t)
        return self.price_start * np.exp(self.price_growth * year)
    
    def calculate_credits(self, abatement: float, year: int) -> float:
        """
        Calculate carbon credits after buffer and crediting period limits.
        
        Args:
            abatement: Carbon abatement (tCO2e)
            year: Year of simulation (0-indexed)
            
        Returns:
            Credits after buffer (tCO2e)
        """
        if year >= self.crediting_years:
            return 0.0
        
        if abatement < 0:
            return 0.0  # No credits for negative abatement
        
        # Apply buffer (conservative approach)
        credits = abatement * (1 - self.buffer)
        
        return max(0.0, credits)
    
    def calculate_revenue(self, credits: float, year: int) -> float:
        """
        Calculate revenue from carbon credits.
        
        Args:
            credits: Carbon credits (tCO2e)
            year: Year of simulation
            
        Returns:
            Revenue ($)
        """
        price = self.calculate_carbon_price(year)
        return credits * price
    
    def get_scenario_costs(self, scenario: str) -> Dict[str, float]:
        """
        Get costs for specific scenario.
        
        Args:
            scenario: Scenario name ('management' or 'reforestation')
            
        Returns:
            Dictionary of costs
        """
        if scenario not in self.costs_config:
            return {'capex': 0, 'opex': 0, 'mrv': 0}
        
        return self.costs_config[scenario]
    
    def calculate_cashflow(self, abatement_series: List[float], 
                         scenario: str, area_ha: float = 1.0) -> pd.DataFrame:
        """
        Calculate annual cashflow for project.
        
        Args:
            abatement_series: Annual abatement values (tCO2e/ha)
            scenario: Scenario name
            area_ha: Project area (hectares)
            
        Returns:
            DataFrame with cashflow analysis
        """
        # Validate inputs
        assert abatement_series is not None, "Abatement series cannot be None"
        assert len(abatement_series) > 0, "Abatement series cannot be empty"
        assert scenario in {'management', 'reforestation'}, f"Invalid scenario: {scenario}"
        assert area_ha > 0, f"Project area must be positive, got {area_ha}"
        
        costs = self.get_scenario_costs(scenario)
        years = len(abatement_series)
        
        cashflow_data = []
        
        for year in range(years):
            # Calculate revenues
            abatement = abatement_series[year] * area_ha
            credits = self.calculate_credits(abatement, year)
            revenue = self.calculate_revenue(credits, year)
            
            # Calculate costs
            capex = costs['capex'] * area_ha if year == 0 else 0
            opex = costs['opex'] * area_ha if year <= self.crediting_years else 0
            mrv = costs['mrv'] * area_ha if year <= self.crediting_years else 0
            total_costs = capex + opex + mrv
            
            # Net cashflow
            net_cashflow = revenue - total_costs
            
            # Discounted cashflow using continuous discounting
            # For continuous discounting: DF = exp(-r * t)
            discount_factor = np.exp(-self.discount_rate * year)
            discounted_cashflow = net_cashflow * discount_factor
            
            cashflow_data.append({
                'year': year,
                'abatement_tCO2e': abatement,
                'credits_tCO2e': credits,
                'carbon_price': self.calculate_carbon_price(year),
                'revenue': revenue,
                'capex': capex,
                'opex': opex,
                'mrv': mrv,
                'total_costs': total_costs,
                'net_cashflow': net_cashflow,
                'discount_factor': discount_factor,
                'discounted_cashflow': discounted_cashflow,
                'cumulative_discounted': sum([cf['discounted_cashflow'] 
                                             for cf in cashflow_data])
            })
        
        cashflow_df = pd.DataFrame(cashflow_data)
        assert not cashflow_df.empty, "Cashflow calculation resulted in empty dataframe"
        assert len(cashflow_df) == years, f"Expected {years} years of cashflow data, got {len(cashflow_df)}"
        return cashflow_df
    
    def calculate_npv(self, cashflow_df: pd.DataFrame) -> float:
        """
        Calculate Net Present Value.
        
        Args:
            cashflow_df: Cashflow DataFrame
            
        Returns:
            NPV ($)
        """
        assert cashflow_df is not None, "Cashflow DataFrame cannot be None"
        assert not cashflow_df.empty, "Cashflow DataFrame cannot be empty"
        assert 'discounted_cashflow' in cashflow_df.columns, "Cashflow DataFrame must contain 'discounted_cashflow' column"
        
        return cashflow_df['discounted_cashflow'].sum()
    
    def calculate_irr(self, cashflow_df: pd.DataFrame) -> float:
        """
        Calculate Internal Rate of Return using Newton-Raphson method.
        
        Args:
            cashflow_df: Cashflow DataFrame
            
        Returns:
            IRR (%)
        """
        assert cashflow_df is not None, "Cashflow DataFrame cannot be None"
        assert not cashflow_df.empty, "Cashflow DataFrame cannot be empty"
        assert 'net_cashflow' in cashflow_df.columns, "Cashflow DataFrame must contain 'net_cashflow' column"
        
        cashflows = cashflow_df['net_cashflow'].values
        
        # Use numpy-financial if available
        try:
            from numpy_financial import irr
            irr_value = irr(cashflows) * 100
            return irr_value if not np.isnan(irr_value) else 0.0
        except ImportError:
            # Manual IRR calculation using Newton-Raphson
            return self._calculate_irr_newton_raphson(cashflows)
    
    def _calculate_irr_newton_raphson(self, cashflows: np.ndarray, 
                                    initial_guess: float = 0.1, 
                                    tolerance: float = 1e-6, 
                                    max_iterations: int = 100) -> float:
        """
        Calculate IRR using Newton-Raphson method.
        
        Args:
            cashflows: Array of cashflows
            initial_guess: Initial guess for IRR
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            
        Returns:
            IRR (%)
        """
        if len(cashflows) < 2:
            return 0.0
        
        # Check if all cashflows are positive or negative
        if all(cf >= 0 for cf in cashflows) or all(cf <= 0 for cf in cashflows):
            return 0.0
        
        r = initial_guess
        
        for _ in range(max_iterations):
            # Calculate NPV and its derivative
            npv = 0.0
            npv_derivative = 0.0
            
            for i, cf in enumerate(cashflows):
                if cf != 0:
                    discount_factor = (1 + r) ** i
                    npv += cf / discount_factor
                    npv_derivative -= i * cf / ((1 + r) ** (i + 1))
            
            # Check convergence
            if abs(npv) < tolerance:
                return r * 100
            
            # Update guess
            if abs(npv_derivative) < tolerance:
                break
            
            r = r - npv / npv_derivative
            
            # Keep rate reasonable
            r = max(-0.99, min(r, 10.0))
        
        return r * 100
    
    def calculate_payback_period(self, cashflow_df: pd.DataFrame) -> Optional[int]:
        """
        Calculate payback period.
        
        Args:
            cashflow_df: Cashflow DataFrame
            
        Returns:
            Payback period in years, or None if never
        """
        cumulative = cashflow_df['net_cashflow'].cumsum()
        positive_years = cumulative[cumulative > 0]
        
        if len(positive_years) > 0:
            return positive_years.index[0]
        return None
    
    def generate_summary(self, cashflow_df: pd.DataFrame) -> Dict:
        """
        Generate economic summary metrics.
        
        Args:
            cashflow_df: Cashflow DataFrame
            
        Returns:
            Dictionary of summary metrics
        """
        assert cashflow_df is not None, "Cashflow DataFrame cannot be None"
        assert not cashflow_df.empty, "Cashflow DataFrame cannot be empty"
        required_columns = ['revenue', 'total_costs', 'credits_tCO2e', 'carbon_price']
        for col in required_columns:
            assert col in cashflow_df.columns, f"Cashflow DataFrame must contain '{col}' column"
        
        return {
            'npv': self.calculate_npv(cashflow_df),
            'irr': self.calculate_irr(cashflow_df),
            'payback_period': self.calculate_payback_period(cashflow_df),
            'total_revenue': cashflow_df['revenue'].sum(),
            'total_costs': cashflow_df['total_costs'].sum(),
            'total_credits': cashflow_df['credits_tCO2e'].sum(),
            'avg_carbon_price': cashflow_df['carbon_price'].mean()
        }
