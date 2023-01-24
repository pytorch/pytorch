#!/bin/bash

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

docker build -t pytorch .
