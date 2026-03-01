#!/bin/bash

set -ex

apt-get update
apt-get install -y --no-install-recommends \
        libopencv-dev

# Cleanup
apt-get autoclean && apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Cache vision models used by the test
source "$(dirname "${BASH_SOURCE[0]}")/cache_vision_models.sh"
