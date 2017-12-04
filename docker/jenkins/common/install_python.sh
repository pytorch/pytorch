#!/bin/bash

set -ex

if [ -z "$PYTHON_VERSION" ]; then
  echo "Please specify PYTHON_VERSION..."
  exit 1
fi

apt-get update
case "$PYTHON_VERSION" in
  2*)
    apt-get install -y --no-install-recommends \
            python-dev \
            python-setuptools
    PYTHON=python2
  ;;
  3*)
    apt-get install -y --no-install-recommends \
            python3-dev \
            python3-setuptools
    PYTHON=python3
  ;;
  *)
    echo "Invalid PYTHON_VERSION..."
    exit 1
  ;;
esac

# Install pip from source.
# The python-pip package on Ubuntu Trusty is old
# and upon install numpy doesn't use the binary
# distribution, and fails to compile it from source.
curl -O https://pypi.python.org/packages/11/b6/abcb525026a4be042b486df43905d6893fb04f05aac21c32c638e939e447/pip-9.0.1.tar.gz
tar zxf pip-9.0.1.tar.gz
pushd pip-9.0.1
"$PYTHON" setup.py install
popd
rm -rf pip-9.0.1*

# Install pip packages
pip install --no-cache-dir \
    future \
    hypothesis \
    numpy \
    protobuf \
    pytest \
    scipy==0.19.1 \
    scikit-image \
    virtualenv
