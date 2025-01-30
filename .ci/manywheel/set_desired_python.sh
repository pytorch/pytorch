#!/usr/bin/env bash

# Require only one python installation
if [[ -z "$DESIRED_PYTHON" ]]; then
    echo "Need to set DESIRED_PYTHON env variable"
    exit 1
fi

# If given a python version like 3.6m or 2.7mu, convert this to the format we
# expect. The binary CI jobs pass in python versions like this; they also only
# ever pass one python version, so we assume that DESIRED_PYTHON is not a list
# in this case
if [[ -n "$DESIRED_PYTHON" && $DESIRED_PYTHON =~ ([0-9].[0-9]+)t ]]; then
    python_digits="$(echo $DESIRED_PYTHON | tr -cd [:digit:])"
    py_majmin="${DESIRED_PYTHON}"
    DESIRED_PYTHON="cp${python_digits}-cp${python_digits}t"
elif [[ -n "$DESIRED_PYTHON" && "$DESIRED_PYTHON" != cp* ]]; then
    python_nodot="$(echo $DESIRED_PYTHON | tr -d m.u)"
    DESIRED_PYTHON="cp${python_nodot}-cp${python_nodot}"
    if [[ ${python_nodot} -ge 310 ]]; then
        py_majmin="${DESIRED_PYTHON:2:1}.${DESIRED_PYTHON:3:2}"
    else
        py_majmin="${DESIRED_PYTHON:2:1}.${DESIRED_PYTHON:3:1}"
    fi
fi

pydir="/opt/python/$DESIRED_PYTHON"
export DESIRED_PYTHON_BIN_DIR="${pydir}/bin"
export PATH="$DESIRED_PYTHON_BIN_DIR:$PATH"
echo "Will build for Python version: ${DESIRED_PYTHON}"
