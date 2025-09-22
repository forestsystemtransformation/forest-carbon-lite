#!/usr/bin/env python3
"""
Forest Carbon Lite Installation Test

This script tests that Forest Carbon Lite is properly installed and working.
"""

import subprocess
import sys
from pathlib import Path

def test_command(command, description):
    """Test a command and return success status."""
    print(f"🔄 Testing: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"   Error: {e.stderr}")
        return False

def main():
    """Run installation tests."""
    print("🌲 Forest Carbon Lite - Installation Test")
    print("=" * 50)
    
    tests = [
        ("fcl --version", "CLI version check"),
        ("fcl --help", "CLI help command"),
        ("fcl simulate --help", "Simulate command help"),
        ("fcl analyze --help", "Analyze command help"),
        ("fcl comprehensive --help", "Comprehensive command help"),
    ]
    
    passed = 0
    total = len(tests)
    
    for command, description in tests:
        if test_command(command, description):
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Forest Carbon Lite is properly installed.")
        print("\n📖 Quick Start Examples:")
        print("   fcl simulate --forest ETOF --years 25 --plot --seed 42")
        print("   fcl analyze --forest-types ETOF,AFW --climates current --years 50 --plots")
        return True
    else:
        print("❌ Some tests failed. Please check the installation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
