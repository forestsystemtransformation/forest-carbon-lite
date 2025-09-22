# Forest Carbon Lite - Reports

This directory contains LaTeX reports and documentation for the Forest Carbon Lite project.

## Files

- `forest_carbon_analysis_report.tex` - Main analysis report template
- `build_report.py` - Python script to build LaTeX reports
- `README.md` - This file

## Building Reports

### Prerequisites

1. **LaTeX Distribution**: Install MiKTeX (Windows) or TeX Live (Linux/Mac)
2. **Python**: For the build script (optional)

### Building with Python Script

```bash
cd reports
python build_report.py
```

### Building Manually

```bash
cd reports
pdflatex forest_carbon_analysis_report.tex
pdflatex forest_carbon_analysis_report.tex  # Run twice for references
```

## Report Structure

The main report includes:

1. **Introduction** - Project overview and objectives
2. **Methodology** - Model description and scenario configuration
3. **Results** - Analysis results with figures and tables
4. **Discussion** - Key findings and insights
5. **Conclusions** - Summary of findings
6. **Recommendations** - Actionable recommendations
7. **References** - Literature and data sources
8. **Appendix** - Technical specifications

## Adding Your Analysis Results

To include your actual analysis results:

1. **Run your forest carbon simulations** using the main project tools
2. **Generate plots and data** in the `output/` directory
3. **Update the LaTeX file** to reference your actual results:
   - Replace placeholder figures with actual plot paths
   - Add real data to tables
   - Update findings based on your analysis

## Example Integration

After running a scenario analysis:

```bash
# Run analysis
python main.py analyze --forest-types ETOF --climates current,paris --years 25

# Build report with results
cd reports
python build_report.py
```

The report will automatically reference the generated plots and data from your analysis.

## Customization

You can customize the report by:

- Modifying the LaTeX template
- Adding new sections
- Including additional figures
- Updating the styling and formatting

## Git Integration

- LaTeX source files (`.tex`) are tracked in git
- Generated PDFs and build artifacts are ignored (see `.gitignore`)
- Only commit the source files, not the compiled outputs
