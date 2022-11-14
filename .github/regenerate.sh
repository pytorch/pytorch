#!/bin/bash -e


# Allows this script to be invoked from any directory:
cd "$(dirname "$0")"

# Require common generate binary build matrix from test-infra repo
curl -o scripts/.tools/generate_binary_build_matrix.py --create-dirs https://raw.githubusercontent.com/pytorch/test-infra/main/tools/scripts/generate_binary_build_matrix.py

python3 scripts/generate_ci_workflows.py
