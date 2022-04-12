#!/bin/bash

set -eux -o pipefail

# This step runs on multiple executors with different envfile locations
if [[ "$(uname)" == Darwin ]]; then
  envfile="/Users/distiller/project/env"
elif [[ -d "/home/circleci/project" ]]; then
  # machine executor (binary tests)
  envfile="/home/circleci/project/env"
else
  # docker executor (binary builds)
  envfile="/env"
fi

# TODO this is super hacky and ugly. Basically, the binary_update_html job does
# not have an env file, since it does not call binary_populate_env.sh, since it
# does not have a BUILD_ENVIRONMENT. So for this one case, which we detect by a
# lack of an env file, we manually export the environment variables that we
# need to install miniconda
if [[ ! -f "$envfile" ]]; then
  MINICONDA_ROOT="/home/circleci/project/miniconda"
  workdir="/home/circleci/project"
  retry () {
      $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
  }
  export -f retry
else
  source "$envfile"
fi

conda_sh="$workdir/install_miniconda.sh"
if [[ "$(uname)" == Darwin ]]; then
  curl --retry 3 -o "$conda_sh" https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
else
  curl --retry 3 -o "$conda_sh" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
fi
chmod +x "$conda_sh"
"$conda_sh" -b -p "$MINICONDA_ROOT"
rm -f "$conda_sh"

# We can't actually add miniconda to the PATH in the envfile, because that
# breaks 'unbuffer' in Mac jobs. This is probably because conda comes with
# a tclsh, which then gets inserted before the tclsh needed in /usr/bin
