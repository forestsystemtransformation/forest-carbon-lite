# Forest Plot Data Collection Guide
## Complete Parameter List for Forest Carbon Lite Site Configurations

This guide provides a comprehensive list of all parameters needed to create site-specific configurations for Forest Carbon Lite. Use this to collect real data from forest plots or research databases.

---

## 🌲 **FOREST TYPE & BASIC PARAMETERS**

### **Forest Type Classification**
- **Required**: Forest type identifier (ETOF, EOF, AFW, RAINFOREST, MALLEE, SHRUBLAND, OTHER_FW, EW_OW)
- **Data Source**: Forest inventory databases, vegetation mapping, field surveys
- **Collection Method**: Visual assessment, species composition, canopy structure

### **Maximum Biomass Potential (K_AGB)**
- **Parameter**: `K_AGB` (tonnes/ha)
- **Range**: 10-1000 tonnes/ha
- **Data Source**: 
  - Forest inventory databases (e.g., National Forest Inventory)
  - Allometric equations for maximum biomass
  - Literature values for forest type
- **Collection Method**: 
  - Measure largest trees in the area
  - Use allometric equations: `Biomass = a × DBH^b × Height^c`
  - Research maximum biomass for forest type

### **Age of Maximum Growth Rate (G)**
- **Parameter**: `G` (years)
- **Range**: 1-100 years
- **Data Source**: 
  - Growth ring analysis
  - Forest inventory data
  - Species-specific growth curves
- **Collection Method**: 
  - Dendrochronology (tree ring analysis)
  - Forest inventory age-class data
  - Literature values for species

### **Root-to-Shoot Ratio**
- **Parameter**: `root_shoot` (dimensionless)
- **Range**: 0.05-1.0
- **Data Source**: 
  - Root excavation studies
  - Literature values by species
  - Allometric equations
- **Collection Method**: 
  - Destructive sampling (excavate roots)
  - Literature values for forest type
  - Use species-specific ratios

---

## 🌱 **INITIAL BIOMASS VALUES** (tonnes/ha)

### **Baseline Scenario Biomass**
- **Parameter**: `initial_biomass_baseline`
- **Data Source**: 
  - Forest inventory plots
  - Remote sensing biomass maps
  - Allometric equations
- **Collection Method**: 
  - Measure DBH and height of all trees
  - Apply allometric equations
  - Use forest inventory data

### **Management Scenario Biomass**
- **Parameter**: `initial_biomass_management`
- **Data Source**: Same as baseline
- **Collection Method**: Same as baseline (typically same starting point)

### **Reforestation Scenario Biomass**
- **Parameter**: `initial_biomass_reforestation`
- **Data Source**: 
  - Newly planted areas
  - Regeneration surveys
- **Collection Method**: 
  - Measure young trees/saplings
  - Use regeneration inventory data

---

## 🌍 **INITIAL AGES** (years)

### **Degraded Forest Age**
- **Parameter**: `age_degraded`
- **Range**: 0-500 years
- **Data Source**: 
  - Forest inventory age data
  - Growth ring analysis
  - Historical records
- **Collection Method**: 
  - Dendrochronology
  - Forest inventory age-class data
  - Historical land use records

### **Managed Forest Age**
- **Parameter**: `age_managed`
- **Collection Method**: Same as degraded (typically same starting age)

### **Reforestation Age**
- **Parameter**: `age_reforestation`
- **Range**: 0-100 years
- **Collection Method**: 
  - Planting records
  - Regeneration surveys
  - Typically 0 for new plantings

---

## 🌿 **SOIL CARBON FRACTIONS** (0-1)

### **Degraded Forest Soil Carbon**
- **Parameter**: `S0_deg_frac`
- **Data Source**: 
  - Soil carbon databases
  - Soil surveys
  - Literature values
- **Collection Method**: 
  - Soil sampling (0-30 cm depth)
  - Laboratory analysis for organic carbon
  - Use soil carbon maps

### **Managed Forest Soil Carbon**
- **Parameter**: `S0_man_frac`
- **Collection Method**: Same as degraded (typically same starting point)

### **Reforestation Soil Carbon**
- **Parameter**: `S0_new_frac`
- **Collection Method**: 
  - Soil sampling in cleared areas
  - Typically much lower (0.01-0.1)

---

## 💀 **MORTALITY RATES** (annual fraction)

### **Degraded Forest Mortality**
- **Parameter**: `m_degraded`
- **Range**: 0-0.5 (0-50% annually)
- **Data Source**: 
  - Forest inventory mortality data
  - Long-term monitoring plots
  - Literature values
- **Collection Method**: 
  - Annual mortality surveys
  - Long-term plot monitoring
  - Literature values for forest type

### **Managed Forest Mortality**
- **Parameter**: `m_managed`
- **Collection Method**: Same as degraded (management effects applied separately)

### **Reforestation Mortality**
- **Parameter**: `m_reforestation`
- **Collection Method**: 
  - Young tree mortality surveys
  - Planting success rates
  - Literature values for establishment

---

## 🔥 **DISTURBANCE PARAMETERS**

### **Disturbance Probabilities** (annual fraction)
- **Parameters**: `pdist_degraded`, `pdist_managed`, `pdist_reforestation`
- **Range**: 0-1 (0-100% annually)
- **Data Source**: 
  - Fire history databases
  - Disturbance mapping
  - Historical records
- **Collection Method**: 
  - Fire history analysis
  - Disturbance mapping from satellite data
  - Historical records

### **Disturbance Severity** (fraction of biomass lost)
- **Parameters**: `dsev_degraded`, `dsev_managed`, `dsev_reforestation`
- **Range**: 0-1 (0-100% biomass loss)
- **Data Source**: 
  - Post-fire assessments
  - Disturbance severity studies
  - Literature values
- **Collection Method**: 
  - Post-disturbance biomass surveys
  - Satellite-based severity mapping
  - Literature values for fire severity

---

## 📈 **TYF CALIBRATION PARAMETERS**

### **Maximum Potential Biomass (M)**
- **Parameter**: `M` (tonnes/ha)
- **Range**: 10-1000 tonnes/ha
- **Collection Method**: Same as K_AGB

### **Age of Maximum Growth Rate (G)**
- **Parameter**: `G` (years)
- **Collection Method**: Same as basic G parameter

### **Growth Multiplier (y)**
- **Parameter**: `y` (dimensionless)
- **Range**: 0.1-5.0
- **Data Source**: 
  - Growth rate comparisons
  - Management effect studies
  - Literature values
- **Collection Method**: 
  - Compare growth rates between scenarios
  - Literature values for management effects
  - Typically 1.0 for baseline, >1.0 for management

---

## 🌡️ **CLIMATE PARAMETERS**

### **FPI (Forest Productivity Index) Ratios**
- **Parameters**: `fpi_ratios.baseline`, `fpi_ratios.management`, `fpi_ratios.reforestation`
- **Range**: 0.1-2.0
- **Data Source**: 
  - Climate databases
  - Productivity indices
  - Literature values
- **Collection Method**: 
  - Climate data analysis
  - Productivity mapping
  - Literature values for climate effects

### **Climate Adjustments**
- **Temperature Change**: Additional temperature increase (°C)
- **Rainfall Change**: Rainfall change (mm/year)
- **Data Source**: 
  - Climate projections
  - Historical climate data
  - IPCC scenarios

---

## 💰 **ECONOMIC PARAMETERS**

### **Carbon Pricing**
- **Parameters**: 
  - `price_start`: Starting carbon price ($/tCO2e)
  - `price_growth`: Annual price growth rate
  - `buffer`: Risk buffer (0-1)
  - `crediting_years`: Crediting period (years)
  - `discount_rate`: Discount rate (0-1)

### **Cost Parameters**
- **Management Costs**:
  - `capex`: Capital expenditure ($/ha)
  - `opex`: Operating expenditure ($/ha/year)
  - `mrv`: Monitoring, reporting, verification ($/ha/year)
- **Reforestation Costs**: Same structure as management

---

## 📊 **DATA COLLECTION PRIORITIES**

### **High Priority (Essential)**
1. **Forest Type Classification**
2. **Initial Biomass Values** (all scenarios)
3. **Initial Ages** (all scenarios)
4. **Maximum Biomass Potential (K_AGB)**
5. **Age of Maximum Growth Rate (G)**

### **Medium Priority (Important)**
1. **Mortality Rates** (all scenarios)
2. **Disturbance Probabilities** (all scenarios)
3. **Disturbance Severity** (all scenarios)
4. **Soil Carbon Fractions** (all scenarios)

### **Low Priority (Can use defaults)**
1. **Economic Parameters** (use default values)
2. **Climate Parameters** (use standard scenarios)
3. **TYF Calibration** (use literature values)

---

## 🔍 **RECOMMENDED DATA SOURCES**

### **Australia-Specific**
- **National Forest Inventory** (NFI)
- **Forests of Australia** database
- **TERN AusPlots** network
- **State forest inventory data**
- **Fire history databases**

### **International**
- **Global Forest Watch**
- **FAO Global Forest Resources Assessment**
- **WorldClim** climate data
- **SoilGrids** soil data
- **Allometric equation databases**

### **Literature Sources**
- **Forest Ecology and Management** journal
- **Global Change Biology**
- **Forest Science**
- **Species-specific allometric studies**

---

## 📝 **DATA COLLECTION TEMPLATE**

```
FOREST PLOT DATA SHEET
=====================

Plot ID: ________________
Date: ________________
Location: _______________
Forest Type: ____________

BIOMASS DATA
------------
Total AGB (tonnes/ha): _______
Largest tree DBH (cm): _______
Largest tree height (m): ______
Number of trees/ha: _________

AGE DATA
--------
Average age (years): _________
Age range: _______________
Growth ring count: _________

SOIL DATA
---------
Soil carbon % (0-30cm): ______
Soil depth (cm): ____________
Soil type: _________________

DISTURBANCE HISTORY
-------------------
Last fire year: ____________
Fire severity: _____________
Other disturbances: ________

MORTALITY
---------
Dead trees/ha: _____________
Mortality rate (%/year): ____
```

---

## ⚠️ **IMPORTANT NOTES**

1. **Units**: All biomass values in tonnes/ha, ages in years, rates as fractions
2. **Validation**: Use the built-in guardrails to check parameter realism
3. **Defaults**: Many parameters have sensible defaults - focus on plot-specific data
4. **Literature**: Use literature values for parameters difficult to measure directly
5. **Uncertainty**: Consider uncertainty ranges for critical parameters

This guide should help you collect all necessary data to create accurate, site-specific configurations for Forest Carbon Lite!
