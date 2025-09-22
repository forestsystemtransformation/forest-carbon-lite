#!/usr/bin/env python3
"""
Forest Carbon Lite Installation Script

This script helps users install Forest Carbon Lite with all dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_package():
    """Install the Forest Carbon Lite package."""
    print("🌲 Forest Carbon Lite Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ pyproject.toml not found. Please run this script from the forest-carbon-lite directory.")
        return False
    
    # Install the package
    if not run_command("pip install -e .", "Installing Forest Carbon Lite"):
        return False
    
    # Test installation
    if not run_command("fcl --version", "Testing installation"):
        print("⚠️  Package installed but CLI test failed. You may need to restart your terminal.")
        return False
    
    print("\n🎉 Installation completed successfully!")
    print("\n📖 Quick Start:")
    print("   fcl simulate --forest ETOF --years 25 --plot --seed 42")
    print("   fcl analyze --forest-types ETOF,AFW --climates current --years 50 --plots")
    print("   fcl --help")
    
    return True

if __name__ == "__main__":
    success = install_package()
    sys.exit(0 if success else 1)

