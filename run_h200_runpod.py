#!/usr/bin/env python3
"""H200 entry point for the shared, GPU-detecting Runpod benchmark runner."""
import sys

from run_b200_runpod import cli


if __name__ == "__main__":
    sys.exit(cli())
