#!/usr/bin/env python3
"""
Main entry point for Forest Carbon Lite

This is a compatibility wrapper that imports and runs the main CLI.
For development use, this maintains backward compatibility.
For production use, install the package and use the 'fcl' command.
"""

from forest_carbon.cli import main

if __name__ == "__main__":
    main()
