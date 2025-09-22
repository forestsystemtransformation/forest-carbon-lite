#!/usr/bin/env python3
"""
Build script for LaTeX reports in the Forest Carbon Lite project.
This script compiles LaTeX documents and handles dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def build_latex_report(tex_file, output_dir="."):
    """
    Build a LaTeX report using pdflatex.
    
    Args:
        tex_file (str): Path to the .tex file
        output_dir (str): Output directory for the PDF
    """
    tex_path = Path(tex_file)
    
    if not tex_path.exists():
        print(f"Error: LaTeX file {tex_file} not found!")
        return False
    
    # Change to the reports directory
    os.chdir(tex_path.parent)
    
    # Build command
    build_cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-output-directory", output_dir,
        tex_path.name
    ]
    
    try:
        print(f"Building {tex_path.name}...")
        
        # First pass
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Error in first pass:")
            print(result.stdout)
            print(result.stderr)
            return False
        
        # Second pass (for references, etc.)
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Error in second pass:")
            print(result.stdout)
            print(result.stderr)
            return False
        
        print(f"Successfully built {tex_path.stem}.pdf")
        return True
        
    except FileNotFoundError:
        print("Error: pdflatex not found. Please install a LaTeX distribution (e.g., MiKTeX, TeX Live)")
        return False
    except Exception as e:
        print(f"Error building LaTeX document: {e}")
        return False

def main():
    """Main function to build the forest carbon analysis report."""
    
    # Default report file
    report_file = "forest_carbon_analysis_report.tex"
    
    # Check if report file exists
    if not Path(report_file).exists():
        print(f"Error: {report_file} not found in current directory!")
        print("Make sure you're running this script from the reports/ directory")
        sys.exit(1)
    
    # Build the report
    success = build_latex_report(report_file)
    
    if success:
        print("\n✅ Report built successfully!")
        print("📄 Check the generated PDF file")
    else:
        print("\n❌ Report build failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
