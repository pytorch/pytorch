#!/bin/bash

set -ex
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

conda_sh="$workdir/install_miniconda.sh"
if [[ "$(uname)" == Darwin ]]; then
  retry curl -o "$conda_sh" https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
else
  retry curl -o "$conda_sh" https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
fi
chmod +x "$conda_sh"
"$conda_sh" -b -p "$MINICONDA_ROOT"
rm -f "$conda_sh"

# TODO we can probably remove the next two lines
export PATH="$MINICONDA_ROOT/bin:$PATH"
source "$MINICONDA_ROOT/bin/activate"

# We can't actually add miniconda to the PATH in the envfile, because that
# breaks 'unbuffer' in Mac jobs. This is probably because conda comes with
# a tclsh, which then gets inserted before the tclsh needed in /usr/bin
