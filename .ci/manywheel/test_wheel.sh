#!/usr/bin/env bash
set -e

yum install -y wget git

rm -rf /usr/local/cuda*

# Install Anaconda
if ! ls /py
then
    echo "Miniconda needs to be installed"
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p /py
else
    echo "Miniconda is already installed"
fi

export PATH="/py/bin:$PATH"

# Anaconda token
if ls /remote/token
then
   source /remote/token
fi

conda install -y conda-build anaconda-client
