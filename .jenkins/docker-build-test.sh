#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

docker build -t pytorch .
