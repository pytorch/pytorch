#!/bin/bash -e

# TEST Allows this script to be invoked from any directory:
cd "$(dirname "$0")"

python3 scripts/generate_ci_workflows.py
