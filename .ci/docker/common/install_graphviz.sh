#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

if [ -n "${UBUNTU_VERSION}" ]; then
    apt update
    apt-get install -y graphviz
elif [ -n "${CENTOS_VERSION}" ]; then
    dnf update
    dnf install -y graphviz
else
    echo "Unsupported Linux distribution"
    exit 1
fi
