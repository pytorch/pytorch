#!/usr/bin/env bash

set -ex
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
source $SCRIPTPATH/build_caffe2.sh
source $SCRIPTPATH/build_libtorch.sh
