#!/bin/bash

set -ex

if [ -z "$PYTHON_VERSION" ]; then
  echo "Please specify PYTHON_VERSION..."
  exit 1
fi

install_ubuntu() {
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

  # Clean up
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

install_centos() {
  source /etc/os-release
  if [ "$ID" != "centos" ]; then
    echo "Unknown ID: $ID"
    exit 1
  fi

  case "$PYTHON_VERSION" in
    2*)
      yum install -y \
          python-devel \
          python-setuptools
      PYTHON=python2
      ;;
    3*)
      yum install -y \
          python34-devel \
          python34-setuptools
      PYTHON=python3
      ;;
    *)
      echo "Invalid PYTHON_VERSION..."
      exit 1
      ;;
  esac

  # Cleanup
  yum clean all
  rm -rf /var/cache/yum
  rm -rf /var/lib/yum/yumdb
  rm -rf /var/lib/yum/history
}

# Install Python packages depending on the base OS
if [ -f /etc/lsb-release ]; then
  install_ubuntu
elif [ -f /etc/os-release ]; then
  install_centos
else
  echo "Unable to determine OS..."
  exit 1
fi

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
    jupyter \
    numpy \
    protobuf \
    pytest \
    scipy==0.19.1 \
    scikit-image \
    virtualenv
