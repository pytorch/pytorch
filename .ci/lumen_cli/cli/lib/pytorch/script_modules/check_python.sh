#!/bin/bash
set -eu
: "${PYTHON_VERSION:=3.12}"

ACTUAL=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
if [ "$ACTUAL" != "$PYTHON_VERSION" ]; then
    echo "ERROR: expected Python $PYTHON_VERSION but got $ACTUAL"
    exit 1
fi
echo "Python version OK: $ACTUAL"
