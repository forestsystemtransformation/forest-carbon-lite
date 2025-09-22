#!/usr/bin/env python3
"""
Forest Carbon Lite Quick Start Script

This script demonstrates basic usage of Forest Carbon Lite with example scenarios.
"""

import subprocess
import sys
from pathlib import Path

def run_example(command, description):
    """Run an example command."""
    print(f"\n🌲 {description}")
    print(f"Command: {command}")
    print("-" * 60)
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False

def main():
    """Run quick start examples."""
    print("🌲 Forest Carbon Lite - Quick Start Examples")
    print("=" * 60)
    
    # Check if package is installed
    try:
        result = subprocess.run("fcl --version", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Forest Carbon Lite not installed. Please run: pip install -e .")
            return False
    except:
        print("❌ Forest Carbon Lite not installed. Please run: pip install -e .")
        return False
    
    print("✅ Forest Carbon Lite is installed and ready!")
    
    # Example 1: Single simulation
    run_example(
        "fcl simulate --forest ETOF --years 25 --plot --seed 42",
        "Example 1: Single ETOF simulation (25 years, with plots)"
    )
    
    # Example 2: Small scenario analysis
    run_example(
        "fcl analyze --forest-types ETOF --climates current --managements baseline,intensive --years 25 --plots",
        "Example 2: Small scenario analysis (ETOF, current climate, baseline vs intensive)"
    )
    
    # Example 3: Show help
    print("\n🌲 Example 3: Available Commands")
    print("-" * 60)
    run_example("fcl --help", "Show main help")
    
    print("\n🎉 Quick start examples completed!")
    print("\n📖 Next Steps:")
    print("   1. Explore the output/ directory for results")
    print("   2. Try different forest types: --forest AFW,EOF,ETOF")
    print("   3. Try different climates: --climates current,paris_target,paris_overshoot")
    print("   4. Try different management: --managements baseline,adaptive,intensive")
    print("   5. Run longer simulations: --years 100")
    print("   6. Enable uncertainty analysis: --uncertainty")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

