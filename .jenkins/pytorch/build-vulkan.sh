#!/bin/bash

set -ex

# Do some vulkan-specific set up before calling the regular build script.
# shellcheck disable=SC1091
source /var/lib/jenkins/vulkansdk/setup-env.sh

exec "$(dirname "${BASH_SOURCE[0]}")/build.sh" "$@"

