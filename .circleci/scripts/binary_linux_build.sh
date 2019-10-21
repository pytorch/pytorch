#!/bin/bash

echo "RUNNING ON $(uname -a) WITH $(nproc) CPUS AND $(free -m)"
set -eux -o pipefail
source /env

# Defaults here so they can be changed in one place
export MAX_JOBS=12

# Parse the parameters
if [[ "$PACKAGE_TYPE" == 'conda' ]]; then
  build_script='conda/build_pytorch.sh'
elif [[ "$DESIRED_CUDA" == cpu ]]; then
  build_script='manywheel/build_cpu.sh'
else
  build_script='manywheel/build.sh'
fi

# We want to call unbuffer, which calls tclsh which finds the expect
# package. The expect was installed by yum into /usr/bin so we want to
# find /usr/bin/tclsh, but this is shadowed by /opt/conda/bin/tclsh in
# the conda docker images, so we prepend it to the path here.
if [[ "$PACKAGE_TYPE" == 'conda' ]]; then
  mkdir /just_tclsh_bin
  ln -s /usr/bin/tclsh /just_tclsh_bin/tclsh
  export PATH=/just_tclsh_bin:$PATH
fi

# Build the package
SKIP_ALL_TESTS=1 unbuffer "/builder/$build_script" | ts
