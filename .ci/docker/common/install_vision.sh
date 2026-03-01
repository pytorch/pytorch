#!/bin/bash

set -ex

install_ubuntu() {
  apt-get update
  apt-get install -y --no-install-recommends \
          libopencv-dev

  # Cleanup
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

# Install base packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    install_ubuntu
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac

# Cache vision models used by the test
source "$(dirname "${BASH_SOURCE[0]}")/cache_vision_models.sh"
