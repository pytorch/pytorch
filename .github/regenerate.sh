#!/bin/bash -e

# Allows this script to be invoked from any directory - dummy:
cd "$(dirname "$0")"

python3 scripts/generate_ci_workflows.py
