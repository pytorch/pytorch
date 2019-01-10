#!/usr/bin/env bash

set -ex
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
source $SCRIPTPATH/build_aten.sh
source $SCRIPTPATH/build_nanopb.sh
source $SCRIPTPATH/build_libtorch.sh
