#!/bin/bash

source /home/circleci/project/env
cat >/home/circleci/project/ci_test_script.sh <<EOL
# =================== The following code will be executed inside Docker container ===================
set -eux -o pipefail

python_nodot="\$(echo $DESIRED_PYTHON | tr -d m.u)"

# There was a bug that was introduced in conda-package-handling >= 1.6.1 that makes archives
# above a certain size fail out when attempting to extract
# see: https://github.com/conda/conda-package-handling/issues/71
conda install -y conda-package-handling=1.6.0

# Set up Python
if [[ "$PACKAGE_TYPE" == conda ]]; then
  retry conda create -qyn testenv python="$DESIRED_PYTHON"
  source activate testenv >/dev/null
elif [[ "$PACKAGE_TYPE" != libtorch ]]; then
  python_path="/opt/python/cp\$python_nodot-cp\${python_nodot}"
  # Prior to Python 3.8 paths were suffixed with an 'm'
  if [[ -d  "\${python_path}/bin" ]]; then
    export PATH="\${python_path}/bin:\$PATH"
  elif [[ -d "\${python_path}m/bin" ]]; then
    export PATH="\${python_path}m/bin:\$PATH"
  fi
fi

EXTRA_CONDA_FLAGS=""
if [[ "\$python_nodot" = *39* ]]; then
  EXTRA_CONDA_FLAGS="-c=conda-forge"
fi

# Install the package
# These network calls should not have 'retry's because they are installing
# locally and aren't actually network calls
# TODO there is duplicated and inconsistent test-python-env setup across this
#   file, builder/smoke_test.sh, and builder/run_tests.sh, and also in the
#   conda build scripts themselves. These should really be consolidated
pkg="/final_pkgs/\$(ls /final_pkgs)"
if [[ "$PACKAGE_TYPE" == conda ]]; then
  conda install \${EXTRA_CONDA_FLAGS} -y "\$pkg" --offline
  if [[ "$DESIRED_CUDA" == 'cpu' ]]; then
    retry conda install \${EXTRA_CONDA_FLAGS} -y cpuonly -c pytorch
  fi
  retry conda install \${EXTRA_CONDA_FLAGS} -yq future numpy protobuf six
  if [[ "$DESIRED_CUDA" != 'cpu' ]]; then
    # DESIRED_CUDA is in format cu90 or cu102
    if [[ "${#DESIRED_CUDA}" == 4 ]]; then
      cu_ver="${DESIRED_CUDA:2:1}.${DESIRED_CUDA:3}"
    else
      cu_ver="${DESIRED_CUDA:2:2}.${DESIRED_CUDA:4}"
    fi
    retry conda install \${EXTRA_CONDA_FLAGS} -yq -c nvidia -c pytorch "cudatoolkit=\${cu_ver}"
  fi
elif [[ "$PACKAGE_TYPE" != libtorch ]]; then
  pip install "\$pkg"
  retry pip install -q future numpy protobuf six
fi
if [[ "$PACKAGE_TYPE" == libtorch ]]; then
  pkg="\$(ls /final_pkgs/*-latest.zip)"
  unzip "\$pkg" -d /tmp
  cd /tmp/libtorch
fi

# Test the package
/builder/check_binary.sh

# =================== The above code will be executed inside Docker container ===================
EOL
echo
echo
echo "The script that will run in the next step is:"
cat /home/circleci/project/ci_test_script.sh
