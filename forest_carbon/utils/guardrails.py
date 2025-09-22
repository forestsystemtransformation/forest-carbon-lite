"""Scientifically defensible parameter guardrails based on FullCAM calibration studies.

These guardrails prevent unrealistic parameter values that could lead to incorrect
simulation results. Based on research with 9,300+ plots and observed climate impacts.
"""

from typing import Dict, Tuple, Optional
import warnings

# FPI Guardrails
FPI_MIN = 0.40  # Severe degradation/drought (below this = dead forest)
FPI_MAX = 1.30  # Exceptional conditions (rare, only best sites)
FPI_TYPICAL_RANGE = (0.70, 1.15)  # Most scenarios should fall here

# Y Multiplier Guardrails
Y_MIN = 0.70   # Severely degraded/stressed forest
Y_MAX = 1.50   # Maximum realistic management benefit
Y_TYPICAL_RANGE = (0.90, 1.30)  # Most scenarios should fall here
Y_COMPOUND_MAX = 1.60  # Absolute max if multiple effects compound

# Mortality Rate Guardrails (annual fraction)
M_MIN = 0.005  # Healthy mature forest minimum
M_MAX = 0.150  # Extreme stress before stand collapse
M_TYPICAL_RANGE = (0.010, 0.040)  # Normal forest range
M_CATASTROPHIC = 0.200  # Above this = rapid forest decline

# Disturbance Probability Guardrails
PDIST_MIN = 0.005  # Well-managed, low-risk areas
PDIST_MAX = 0.300  # Extreme fire-prone conditions
PDIST_TYPICAL_RANGE = (0.02, 0.15)  # Most forests

# Disturbance Severity Guardrails
DSEV_MIN = 0.02   # Light surface fire only
DSEV_MAX = 0.95   # Near-complete stand replacement
DSEV_TYPICAL_RANGE = (0.10, 0.40)  # Typical fire severity

# Compound Effect Limits
TOTAL_GROWTH_MODIFIER_MAX = 1.80  # Absolute maximum
TOTAL_GROWTH_MODIFIER_TYPICAL = 1.40  # Typical maximum

# Forest-Type Specific Limits
FOREST_LIMITS = {
    'ETOF': {
        'y_max': 1.45,      # Can respond well to management
        'm_max': 0.100,     # Sensitive to mortality
        'fpi_min': 0.50,    # Vulnerable to climate
    },
    'EOF': {
        'y_max': 1.40,      
        'm_max': 0.120,     # Slightly more resilient
        'fpi_min': 0.55,
    },
    'AFW': {
        'y_max': 1.30,      # Less responsive to management
        'm_max': 0.150,     # Adapted to harsh conditions
        'fpi_min': 0.40,    # Can survive very dry conditions
    },
    'RAINFOREST': {
        'y_max': 1.35,      # Moderate management response
        'm_max': 0.080,     # Low mortality tolerance
        'fpi_min': 0.60,    # Requires good conditions
    },
    'MALLEE': {
        'y_max': 1.25,      # Limited management response
        'm_max': 0.180,     # Adapted to harsh conditions
        'fpi_min': 0.35,    # Can survive very dry conditions
    },
    'SHRUBLAND': {
        'y_max': 1.20,      # Minimal management response
        'm_max': 0.200,     # High mortality tolerance
        'fpi_min': 0.30,    # Very drought tolerant
    },
    'OTHER_FW': {
        'y_max': 1.35,      # Moderate management response
        'm_max': 0.120,     # Moderate mortality tolerance
        'fpi_min': 0.50,    # Moderate climate requirements
    },
    'EW_OW': {
        'y_max': 1.40,      # Good management response
        'm_max': 0.110,     # Moderate mortality tolerance
        'fpi_min': 0.55,    # Moderate climate requirements
    }
}

def validate_fpi(fpi: float, forest_type: Optional[str] = None, 
                scenario: Optional[str] = None) -> Tuple[float, bool]:
    """
    Validate FPI ratio against scientifically defensible guardrails.
    
    Args:
        fpi: Forest Productivity Index ratio
        forest_type: Forest type for specific limits
        scenario: Scenario name for context
        
    Returns:
        Tuple of (validated_fpi, was_adjusted)
    """
    original_fpi = fpi
    was_adjusted = False
    
    # Apply forest-specific minimum if available
    if forest_type and forest_type.upper() in FOREST_LIMITS:
        forest_min = FOREST_LIMITS[forest_type.upper()]['fpi_min']
        if fpi < forest_min:
            warnings.warn(
                f"FPI {fpi:.3f} below forest-specific minimum {forest_min:.3f} "
                f"for {forest_type}. Adjusting to minimum.",
                UserWarning
            )
            fpi = forest_min
            was_adjusted = True
    
    # Apply global minimum
    if fpi < FPI_MIN:
        warnings.warn(
            f"FPI {fpi:.3f} below absolute minimum {FPI_MIN:.3f}. "
            f"Forest would be dead. Adjusting to minimum.",
            UserWarning
        )
        fpi = FPI_MIN
        was_adjusted = True
    
    # Apply global maximum
    if fpi > FPI_MAX:
        warnings.warn(
            f"FPI {fpi:.3f} above absolute maximum {FPI_MAX:.3f}. "
            f"Adjusting to maximum.",
            UserWarning
        )
        fpi = FPI_MAX
        was_adjusted = True
    
    # Check if outside typical range
    if not (FPI_TYPICAL_RANGE[0] <= fpi <= FPI_TYPICAL_RANGE[1]):
        warnings.warn(
            f"FPI {fpi:.3f} outside typical range {FPI_TYPICAL_RANGE}. "
            f"Consider reviewing parameter values.",
            UserWarning
        )
    
    return fpi, was_adjusted

def validate_y_multiplier(y: float, forest_type: Optional[str] = None, 
                         scenario: Optional[str] = None) -> Tuple[float, bool]:
    """
    Validate Y multiplier against scientifically defensible guardrails.
    
    Args:
        y: Y multiplier value
        forest_type: Forest type for specific limits
        scenario: Scenario name for context
        
    Returns:
        Tuple of (validated_y, was_adjusted)
    """
    original_y = y
    was_adjusted = False
    
    # Apply forest-specific maximum if available
    if forest_type and forest_type.upper() in FOREST_LIMITS:
        forest_max = FOREST_LIMITS[forest_type.upper()]['y_max']
        if y > forest_max:
            warnings.warn(
                f"Y multiplier {y:.3f} above forest-specific maximum {forest_max:.3f} "
                f"for {forest_type}. Adjusting to maximum.",
                UserWarning
            )
            y = forest_max
            was_adjusted = True
    
    # Apply global minimum
    if y < Y_MIN:
        warnings.warn(
            f"Y multiplier {y:.3f} below absolute minimum {Y_MIN:.3f}. "
            f"Adjusting to minimum.",
            UserWarning
        )
        y = Y_MIN
        was_adjusted = True
    
    # Apply global maximum
    if y > Y_MAX:
        warnings.warn(
            f"Y multiplier {y:.3f} above absolute maximum {Y_MAX:.3f}. "
            f"Adjusting to maximum.",
            UserWarning
        )
        y = Y_MAX
        was_adjusted = True
    
    # Check if outside typical range
    if not (Y_TYPICAL_RANGE[0] <= y <= Y_TYPICAL_RANGE[1]):
        warnings.warn(
            f"Y multiplier {y:.3f} outside typical range {Y_TYPICAL_RANGE}. "
            f"Consider reviewing parameter values.",
            UserWarning
        )
    
    return y, was_adjusted

def validate_mortality_rate(m: float, forest_type: Optional[str] = None, 
                          scenario: Optional[str] = None) -> Tuple[float, bool]:
    """
    Validate mortality rate against scientifically defensible guardrails.
    
    Args:
        m: Annual mortality rate (fraction)
        forest_type: Forest type for specific limits
        scenario: Scenario name for context
        
    Returns:
        Tuple of (validated_m, was_adjusted)
    """
    original_m = m
    was_adjusted = False
    
    # Apply forest-specific maximum if available
    if forest_type and forest_type.upper() in FOREST_LIMITS:
        forest_max = FOREST_LIMITS[forest_type.upper()]['m_max']
        if m > forest_max:
            warnings.warn(
                f"Mortality rate {m:.3f} above forest-specific maximum {forest_max:.3f} "
                f"for {forest_type}. Adjusting to maximum.",
                UserWarning
            )
            m = forest_max
            was_adjusted = True
    
    # Apply global minimum
    if m < M_MIN:
        warnings.warn(
            f"Mortality rate {m:.3f} below absolute minimum {M_MIN:.3f}. "
            f"Adjusting to minimum.",
            UserWarning
        )
        m = M_MIN
        was_adjusted = True
    
    # Apply global maximum
    if m > M_MAX:
        warnings.warn(
            f"Mortality rate {m:.3f} above absolute maximum {M_MAX:.3f}. "
            f"Adjusting to maximum.",
            UserWarning
        )
        m = M_MAX
        was_adjusted = True
    
    # Check for catastrophic levels
    if m > M_CATASTROPHIC:
        warnings.warn(
            f"Mortality rate {m:.3f} above catastrophic threshold {M_CATASTROPHIC:.3f}. "
            f"Forest would rapidly decline.",
            UserWarning
        )
    
    # Check if outside typical range
    if not (M_TYPICAL_RANGE[0] <= m <= M_TYPICAL_RANGE[1]):
        warnings.warn(
            f"Mortality rate {m:.3f} outside typical range {M_TYPICAL_RANGE}. "
            f"Consider reviewing parameter values.",
            UserWarning
        )
    
    return m, was_adjusted

def validate_disturbance_probability(pdist: float, forest_type: Optional[str] = None, 
                                   scenario: Optional[str] = None) -> Tuple[float, bool]:
    """
    Validate disturbance probability against scientifically defensible guardrails.
    
    Args:
        pdist: Annual disturbance probability (fraction)
        forest_type: Forest type for specific limits
        scenario: Scenario name for context
        
    Returns:
        Tuple of (validated_pdist, was_adjusted)
    """
    original_pdist = pdist
    was_adjusted = False
    
    # Apply global minimum
    if pdist < PDIST_MIN:
        warnings.warn(
            f"Disturbance probability {pdist:.3f} below absolute minimum {PDIST_MIN:.3f}. "
            f"Adjusting to minimum.",
            UserWarning
        )
        pdist = PDIST_MIN
        was_adjusted = True
    
    # Apply global maximum
    if pdist > PDIST_MAX:
        warnings.warn(
            f"Disturbance probability {pdist:.3f} above absolute maximum {PDIST_MAX:.3f}. "
            f"Adjusting to maximum.",
            UserWarning
        )
        pdist = PDIST_MAX
        was_adjusted = True
    
    # Check if outside typical range
    if not (PDIST_TYPICAL_RANGE[0] <= pdist <= PDIST_TYPICAL_RANGE[1]):
        warnings.warn(
            f"Disturbance probability {pdist:.3f} outside typical range {PDIST_TYPICAL_RANGE}. "
            f"Consider reviewing parameter values.",
            UserWarning
        )
    
    return pdist, was_adjusted

def validate_disturbance_severity(dsev: float, forest_type: Optional[str] = None, 
                                scenario: Optional[str] = None) -> Tuple[float, bool]:
    """
    Validate disturbance severity against scientifically defensible guardrails.
    
    Args:
        dsev: Disturbance severity (fraction)
        forest_type: Forest type for specific limits
        scenario: Scenario name for context
        
    Returns:
        Tuple of (validated_dsev, was_adjusted)
    """
    original_dsev = dsev
    was_adjusted = False
    
    # Apply global minimum
    if dsev < DSEV_MIN:
        warnings.warn(
            f"Disturbance severity {dsev:.3f} below absolute minimum {DSEV_MIN:.3f}. "
            f"Adjusting to minimum.",
            UserWarning
        )
        dsev = DSEV_MIN
        was_adjusted = True
    
    # Apply global maximum
    if dsev > DSEV_MAX:
        warnings.warn(
            f"Disturbance severity {dsev:.3f} above absolute maximum {DSEV_MAX:.3f}. "
            f"Adjusting to maximum.",
            UserWarning
        )
        dsev = DSEV_MAX
        was_adjusted = True
    
    # Check if outside typical range
    if not (DSEV_TYPICAL_RANGE[0] <= dsev <= DSEV_TYPICAL_RANGE[1]):
        warnings.warn(
            f"Disturbance severity {dsev:.3f} outside typical range {DSEV_TYPICAL_RANGE}. "
            f"Consider reviewing parameter values.",
            UserWarning
        )
    
    return dsev, was_adjusted

def validate_compound_effects(fpi: float, y: float, forest_type: Optional[str] = None) -> Tuple[float, float, bool]:
    """
    Validate compound effects to prevent unrealistic growth modifiers.
    
    Args:
        fpi: Forest Productivity Index ratio
        y: Y multiplier value
        forest_type: Forest type for context
        
    Returns:
        Tuple of (validated_fpi, validated_y, was_adjusted)
    """
    original_fpi, original_y = fpi, y
    was_adjusted = False
    
    compound = fpi * y
    
    if compound > TOTAL_GROWTH_MODIFIER_MAX:
        warnings.warn(
            f"Unrealistic growth modifier: {compound:.2f} (FPI={fpi:.2f} × Y={y:.2f}). "
            f"Maximum allowed: {TOTAL_GROWTH_MODIFIER_MAX:.2f}. Scaling down.",
            UserWarning
        )
        
        # Scale both parameters proportionally to maintain their relative relationship
        scaling = TOTAL_GROWTH_MODIFIER_MAX / compound
        fpi = fpi * scaling
        y = y * scaling
        was_adjusted = True
    
    elif compound > TOTAL_GROWTH_MODIFIER_TYPICAL:
        warnings.warn(
            f"Growth modifier {compound:.2f} above typical maximum {TOTAL_GROWTH_MODIFIER_TYPICAL:.2f}. "
            f"Consider reviewing parameter values.",
            UserWarning
        )
    
    return fpi, y, was_adjusted

def validate_all_parameters(fpi: float, y: float, m: float, pdist: float, dsev: float,
                          forest_type: Optional[str] = None, scenario: Optional[str] = None) -> Dict[str, float]:
    """
    Validate all parameters against scientifically defensible guardrails.
    
    Args:
        fpi: Forest Productivity Index ratio
        y: Y multiplier value
        m: Annual mortality rate (fraction)
        pdist: Annual disturbance probability (fraction)
        dsev: Disturbance severity (fraction)
        forest_type: Forest type for specific limits
        scenario: Scenario name for context
        
    Returns:
        Dictionary of validated parameters
    """
    # Validate individual parameters
    fpi, _ = validate_fpi(fpi, forest_type, scenario)
    y, _ = validate_y_multiplier(y, forest_type, scenario)
    m, _ = validate_mortality_rate(m, forest_type, scenario)
    pdist, _ = validate_disturbance_probability(pdist, forest_type, scenario)
    dsev, _ = validate_disturbance_severity(dsev, forest_type, scenario)
    
    # Validate compound effects
    fpi, y, _ = validate_compound_effects(fpi, y, forest_type)
    
    return {
        'fpi': fpi,
        'y': y,
        'm': m,
        'pdist': pdist,
        'dsev': dsev
    }

def get_forest_limits(forest_type: str) -> Dict[str, float]:
    """
    Get forest-type specific parameter limits.
    
    Args:
        forest_type: Forest type identifier
        
    Returns:
        Dictionary of forest-specific limits
    """
    return FOREST_LIMITS.get(forest_type.upper(), {
        'y_max': Y_MAX,
        'm_max': M_MAX,
        'fpi_min': FPI_MIN
    })

def check_parameter_realism(params: Dict[str, float], forest_type: Optional[str] = None) -> Dict[str, str]:
    """
    Check parameter realism and return warnings for unrealistic values.
    
    Args:
        params: Dictionary of parameters to check
        forest_type: Forest type for specific limits
        
    Returns:
        Dictionary of warnings for each parameter
    """
    warnings_dict = {}
    
    # Check FPI
    if 'fpi' in params:
        fpi = params['fpi']
        if fpi < FPI_TYPICAL_RANGE[0]:
            warnings_dict['fpi'] = f"Low FPI {fpi:.3f} - forest may be degraded"
        elif fpi > FPI_TYPICAL_RANGE[1]:
            warnings_dict['fpi'] = f"High FPI {fpi:.3f} - exceptional conditions"
    
    # Check Y multiplier
    if 'y' in params:
        y = params['y']
        if y < Y_TYPICAL_RANGE[0]:
            warnings_dict['y'] = f"Low Y {y:.3f} - forest may be stressed"
        elif y > Y_TYPICAL_RANGE[1]:
            warnings_dict['y'] = f"High Y {y:.3f} - exceptional management benefit"
    
    # Check mortality rate
    if 'm' in params:
        m = params['m']
        if m > M_TYPICAL_RANGE[1]:
            warnings_dict['m'] = f"High mortality {m:.3f} - forest under stress"
        if m > M_CATASTROPHIC:
            warnings_dict['m'] = f"Catastrophic mortality {m:.3f} - rapid forest decline"
    
    # Check disturbance probability
    if 'pdist' in params:
        pdist = params['pdist']
        if pdist > PDIST_TYPICAL_RANGE[1]:
            warnings_dict['pdist'] = f"High disturbance risk {pdist:.3f} - fire-prone area"
    
    # Check disturbance severity
    if 'dsev' in params:
        dsev = params['dsev']
        if dsev > DSEV_TYPICAL_RANGE[1]:
            warnings_dict['dsev'] = f"High severity {dsev:.3f} - severe fire conditions"
    
    # Check compound effects
    if 'fpi' in params and 'y' in params:
        compound = params['fpi'] * params['y']
        if compound > TOTAL_GROWTH_MODIFIER_TYPICAL:
            warnings_dict['compound'] = f"High growth modifier {compound:.2f} - exceptional conditions"
    
    return warnings_dict

