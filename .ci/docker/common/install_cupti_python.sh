#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

# The Cuspy tests need libcupti >= 13.5 (TEST_CUPTI_V13_5). torch loads libcupti from
# the nvidia-cuda-cupti wheel, so pin it together with the matching cupti-python.
pip_install \
  cupti-python==13.5.0 \
  nvidia-cuda-cupti==13.5.25
