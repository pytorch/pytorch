#!/bin/bash

set -eux -o pipefail

# This step runs on multiple executors with different envfile locations
if [[ "$(uname)" == Darwin ]]; then
  source "/Users/distiller/project/env"
elif [[ -d "/home/circleci/project" ]]; then
  # machine executor (binary tests)
  source "/home/circleci/project/env"
else
  # docker executor (binary builds)
  source "/env"
fi

# MINICONDA_ROOT is populated in binary_populate_env.sh , but update_htmls does
# not source that script since it does not have a BUILD_ENVIRONMENT. It could
# make a fake BUILD_ENVIRONMENT and call that script anyway, but that seems
# more hacky than this
if [[ -z "${MINICONDA_ROOT:-}" ]]; then
  # TODO get rid of this. Might need to separate binary_populate_env into two
  # steps, one for every job and one for build jobs
  MINICONDA_ROOT="/home/circleci/project/miniconda"
  workdir="/home/circleci/project"
  retry () {
      $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
  }
  export -f retry
fi

conda_sh="$workdir/install_miniconda.sh"
if [[ "$(uname)" == Darwin ]]; then
  retry curl -o "$conda_sh" https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
else
  retry curl -o "$conda_sh" https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
fi
chmod +x "$conda_sh"
"$conda_sh" -b -p "$MINICONDA_ROOT"
rm -f "$conda_sh"

# We can't actually add miniconda to the PATH in the envfile, because that
# breaks 'unbuffer' in Mac jobs. This is probably because conda comes with
# a tclsh, which then gets inserted before the tclsh needed in /usr/bin
