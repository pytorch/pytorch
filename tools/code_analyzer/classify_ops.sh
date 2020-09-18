#!/bin/bash

# If you're a Facebook employee, chances are you're running on CentOS 8.
# If that's the case, you can install all the dependencies you need with:
#
#   sudo dnf install llvm-devel llvm-static clang ncurses-devel
#
# and then set LLVM_DIR=/usr

set -eux -o pipefail

ROOT="$( cd "$(dirname "$0")" ; pwd -P)/../.."
LLVM_DIR=${LLVM_DIR:-/usr}

cd $ROOT

# Build PyTorch itself to produce Declarations.yaml.
# TODO: directly call codegen script?
python setup.py develop

# Call code analyzer to analyze DispatchStub call graph.
rm -rf build_code_analyzer
ANALYZE_TORCH=1 tools/code_analyzer/build.sh -debug_path=true

# Call master script.
python -m tools.code_analyzer.classify_ops
