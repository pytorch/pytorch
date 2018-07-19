#!/bin/bash

set -ex

# Needs https transport for apt
apt-get update
apt-get install -y --no-install-recommends apt-transport-https

# Add Intel MKL repository
key="https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB"
curl "${key}" | apt-key add -
echo 'deb http://apt.repos.intel.com/mkl all main' | \
  tee /etc/apt/sources.list.d/intel-mkl.list
apt-get update

# Multiple candidates for intel-mkl-64bit, so have to be specific
apt-get install -y --no-install-recommends intel-mkl-64bit-2018.1-038
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Ensure loader can find MKL path
echo '/opt/intel/mkl/lib/intel64' | tee /etc/ld.so.conf.d/intel-mkl.conf
ldconfig
