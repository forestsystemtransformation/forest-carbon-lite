# Changelog

All notable changes to the Forest Carbon Lite Simulator will be documented in this file.

## [Latest] - 2025-01-19

### Fixed
- **Critical Validation Bug**: Fixed scenario builder validation error where FPI ratios were using incorrect key `"baseline degraded"` instead of `"baseline"`
  - **Impact**: All scenarios now run successfully without configuration validation errors
  - **Files Changed**: `scenario_builder.py` (line 348)
  - **Before**: `'baseline degraded': base_fpi * 0.9`
  - **After**: `'baseline': base_fpi * 0.9`

### Improved
- **Documentation**: Added comprehensive troubleshooting section to README.md
- **Scenario Builder Guide**: Added detailed explanation of recent fixes and configuration loading behavior
- **Validation Schema**: Enhanced understanding of how external config files take priority over built-in defaults

### Technical Details
- **Configuration Loading**: Scenario builder now correctly prioritizes external site config files (`config/site_ETOF.yaml`) over internal defaults
- **Validation**: All generated scenarios pass Pydantic schema validation
- **Error Handling**: Better error messages and troubleshooting guidance

### Testing
- ✅ All scenario combinations now run successfully
- ✅ Validation errors eliminated
- ✅ Parallel processing works correctly
- ✅ Output generation and visualization working properly

---

## Previous Versions

### [v1.0] - Initial Release
- FullCAM-lite implementation with Tree Yield Formula
- Dynamic scenario builder system
- Climate change integration
- Economic analysis capabilities
- Professional visualization system
