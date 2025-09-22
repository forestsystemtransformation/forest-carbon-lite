"""Physical constants and conversion factors for carbon accounting."""

# Carbon conversion factors
C_TO_CO2 = 44.0 / 12.0  # Convert C to CO2
BIOMASS_TO_C = 0.47  # Default carbon fraction in biomass

# Pool decay rates (per year)
LITTER_DECAY_RATE = 0.3
CHAR_DECAY_RATE = 0.01
ACTIVE_SOIL_DECAY_RATE = 0.1
SLOW_SOIL_DECAY_RATE = 0.01

# Harvested wood product lifespans (years)
HWP_SHORT_LIFE = 2
HWP_MEDIUM_LIFE = 20
HWP_LONG_LIFE = 50

# Slash decay rate
SLASH_DECAY_RATE = 0.2

# Default parameters if not in config
DEFAULT_FPI_RATIO = 1.0  # Forest Productivity Index ratio