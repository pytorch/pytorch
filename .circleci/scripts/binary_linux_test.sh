#!/bin/bash

source /home/circleci/project/env
cat >/home/circleci/project/ci_test_script.sh <<EOL
# =================== The following code will be executed inside Docker container ===================
set -eux -o pipefail

# Set up Python
if [[ "$PACKAGE_TYPE" == conda ]]; then
  retry conda create -qyn testenv python="$DESIRED_PYTHON"
  source activate testenv >/dev/null
elif [[ "$DESIRED_PYTHON" == 2.7mu ]]; then
  export PATH="/opt/python/cp27-cp27mu/bin:\$PATH"
else
  python_nodot="\$(echo $DESIRED_PYTHON | tr -d m.u)"
  export PATH="/opt/python/cp\$python_nodot-cp\${python_nodot}m/bin:\$PATH"
fi

# Install the package
# These network calls should not have 'retry's because they are installing
# locally and aren't actually network calls
# TODO there is duplicated and inconsistent test-python-env setup across this
#   file, builder/smoke_test.sh, and builder/run_tests.sh, and also in the
#   conda build scripts themselves. These should really be consolidated
pkg="/final_pkgs/\$(ls /final_pkgs)"
if [[ "$PACKAGE_TYPE" == conda ]]; then
  conda install -y "\$pkg" --offline
  if [[ "$DESIRED_CUDA" == 'cpu' ]]; then
    conda install -y cpu-only -c pytorch
  fi
  retry conda install -yq future numpy protobuf six
  if [[ "$DESIRED_CUDA" != 'cpu' ]]; then
    # DESIRED_CUDA is in format cu90 or cu100
    if [[ "${#DESIRED_CUDA}" == 4 ]]; then
      cu_ver="${DESIRED_CUDA:2:1}.${DESIRED_CUDA:3}"
    else
      cu_ver="${DESIRED_CUDA:2:2}.${DESIRED_CUDA:4}"
    fi
    retry conda install -yq -c pytorch "cudatoolkit=\${cu_ver}"
  fi
else
  pip install "\$pkg"
  retry pip install -q future numpy protobuf six
fi

# Test the package
/builder/check_binary.sh
# =================== The above code will be executed inside Docker container ===================
EOL
echo
echo
echo "The script that will run in the next step is:"
cat /home/circleci/project/ci_test_script.sh
