#!/bin/bash
set -ex
if [[ "$BUILD_ENVIRONMENT" == "" ]]; then
  CONDA_PARENT_DIR=$(pwd)
  export CONDA_PARENT_DIR
else
  export CONDA_PARENT_DIR=/c/Jenkins
fi

# Be conservative here when rolling out the new AMI with conda. This will try
# to install conda as before if it couldn't find the conda installation. This
# can be removed eventually after we gain enough confidence in the AMI
if [ ! -d "$CONDA_PARENT_DIR/Miniconda3" ]; then
  export INSTALL_FRESH_CONDA=1
fi

if [[ "$INSTALL_FRESH_CONDA" == "1" ]]; then
  curl --retry 3 --retry-all-errors -k https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe --output "$TMP_DIR/Miniconda3-latest-Windows-x86_64.exe"

  "$TMP_DIR/Miniconda3-latest-Windows-x86_64.exe" /InstallationType=JustMe /RegisterPython=0 /S /AddToPath=0 /D=$CONDA_PARENT_DIR/Miniconda3
fi

# Activate conda so that we can use its commands, i.e. conda, python, pip
# This is a batch file
# shellcheck disable=SC1091
source $CONDA_PARENT_DIR/Miniconda3/Scripts/activate $CONDA_PARENT_DIR/Miniconda3
