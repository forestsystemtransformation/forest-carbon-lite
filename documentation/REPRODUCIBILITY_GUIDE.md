# 🔬 Reproducibility Guide - Forest Carbon Lite V.8

This guide explains how to achieve reproducible results in Forest Carbon Lite V.8 for scientific research and quality assurance.

## 🎯 Overview

Forest Carbon Lite V.8 implements comprehensive reproducibility features using seeded random number generation (RNG). This ensures that:

- **Same seed = Identical results** across multiple runs
- **Different seeds = Different stochastic outcomes** for sensitivity analysis
- **Scientific rigor** for publication and peer review
- **Quality assurance** for code changes and updates

## 🔧 Technical Implementation

### Seeded Random Number Generation

The system uses `numpy.random.default_rng(seed)` instances instead of the global RNG:

```python
# ✅ CORRECT: Isolated RNG instance
class DisturbanceModel:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
    def check_disturbance(self):
        if self.rng.random() < self.disturbance_probability:
            severity = self.rng.beta(alpha, beta)
            return True, severity
        return False, 0.0

# ❌ AVOID: Global RNG (old approach)
# np.random.seed(seed)
# if np.random.random() < probability:
```

### Components with Reproducible RNG

1. **DisturbanceModel**: Fire, drought, and mortality events
2. **GrowthCarbonUncertainty**: Monte Carlo parameter sampling
3. **Parameter Distributions**: Beta, normal, and uniform distributions
4. **Batch Processing**: Parallel scenario execution
5. **Mock Data Generation**: Test data creation

## 🚀 Usage Examples

### Single Simulation Reproducibility

```bash
# Identical results across runs
python main.py simulate --forest ETOF --years 25 --seed 42
python main.py simulate --forest ETOF --years 25 --seed 42  # Same results

# Different results with different seeds
python main.py simulate --forest ETOF --years 25 --seed 123  # Different results
python main.py simulate --forest ETOF --years 25 --seed 456  # Different results
```

### Scenario Analysis Reproducibility

```bash
# Reproducible multi-scenario analysis
python main.py analyze --forest-types ETOF,EOF --climates current,paris_target --years 25 --seed 42

# Reproducible uncertainty analysis
python main.py analyze --forest-types ETOF --climates current --managements intensive --uncertainty --seed 123
```

### Uncertainty Analysis Reproducibility

```bash
# Monte Carlo uncertainty analysis with seed
python main.py simulate --forest ETOF --years 25 --uncertainty --seed 42

# Same uncertainty results across runs
python main.py simulate --forest ETOF --years 25 --uncertainty --seed 42  # Identical
```

## 🧪 Verification Testing

### Test Reproducibility

```bash
# Test 1: Same seed should produce identical results
python main.py simulate --forest ETOF --years 5 --seed 123 > run1.txt
python main.py simulate --forest ETOF --years 5 --seed 123 > run2.txt
diff run1.txt run2.txt  # Should show no differences

# Test 2: Different seeds should produce different results
python main.py simulate --forest ETOF --years 5 --seed 123 > run1.txt
python main.py simulate --forest ETOF --years 5 --seed 456 > run2.txt
diff run1.txt run2.txt  # Should show differences
```

### Expected Output Differences

With different seeds, you should see differences in:
- **Disturbance Events**: Number and timing of fires/droughts
- **Uncertainty Analysis**: Monte Carlo parameter sampling
- **Stochastic Parameters**: Mortality rates, growth variations
- **Economic Outcomes**: NPV variations due to stochastic events

## 📊 Scientific Applications

### Experiment Replication

For scientific publications, always include:
1. **Seed Value**: Document the exact seed used
2. **Software Version**: Forest Carbon Lite V.8.0.0
3. **Configuration**: All parameter settings
4. **Hardware**: System specifications if relevant

Example:
```
Simulation Parameters:
- Forest Type: ETOF
- Years: 25
- Seed: 42
- Software: Forest Carbon Lite V.8.0.0
- Configuration: configs/base/site_ETOF.yaml
```

### Sensitivity Analysis

Use different seeds to test parameter sensitivity:

```bash
# Test sensitivity to stochastic events
for seed in 1 2 3 4 5; do
    python main.py simulate --forest ETOF --years 25 --seed $seed --uncertainty
done
```

### Quality Assurance

When updating code, verify that core results remain stable:

```bash
# Before code changes
python main.py simulate --forest ETOF --years 25 --seed 42 > baseline.txt

# After code changes  
python main.py simulate --forest ETOF --years 25 --seed 42 > updated.txt

# Compare results (should be identical for deterministic components)
diff baseline.txt updated.txt
```

## 🔍 Troubleshooting

### Common Issues

1. **Results Not Reproducible**
   - Check that you're using the same seed
   - Verify no other processes are using global RNG
   - Ensure all stochastic components use seeded RNG

2. **Results Too Similar**
   - Verify you're using different seeds
   - Check that disturbance events are enabled
   - Ensure uncertainty analysis is running

3. **Parallel Processing Issues**
   - Each worker process gets its own RNG instance
   - Seeds are properly propagated to all components
   - No shared global state between processes

### Debugging Commands

```bash
# Check if disturbance events are occurring
python main.py simulate --forest ETOF --years 25 --seed 42 | grep "disturbance"

# Verify uncertainty analysis is running
python main.py simulate --forest ETOF --years 25 --uncertainty --seed 42 | grep "uncertainty"

# Test with verbose output
python main.py simulate --forest ETOF --years 25 --seed 42 --plot
```

## 📚 Best Practices

### For Researchers

1. **Always use seeds** for reproducible research
2. **Document seed values** in publications
3. **Test multiple seeds** for sensitivity analysis
4. **Verify reproducibility** before publishing results

### For Developers

1. **Use seeded RNG instances** instead of global RNG
2. **Thread seeds** through all stochastic components
3. **Test reproducibility** after code changes
4. **Document RNG usage** in code comments

### For Quality Assurance

1. **Regression testing** with fixed seeds
2. **Sensitivity analysis** with multiple seeds
3. **Performance testing** with seeded runs
4. **Documentation updates** for RNG changes

## 🎯 Summary

Forest Carbon Lite V.8 provides comprehensive reproducibility through:

- ✅ **Seeded RNG instances** in all stochastic components
- ✅ **End-to-end seed propagation** from CLI to all models
- ✅ **Thread-safe parallel processing** with isolated RNG states
- ✅ **Scientific rigor** for publication and peer review
- ✅ **Quality assurance** for code development and testing

Use the `--seed` parameter to achieve reproducible results for your forest carbon research! 🌲🔬
