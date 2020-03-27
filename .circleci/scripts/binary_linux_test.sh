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
elif [[ "$PACKAGE_TYPE" != libtorch ]]; then
  python_nodot="\$(echo $DESIRED_PYTHON | tr -d m.u)"
  python_path="/opt/python/cp\$python_nodot-cp\${python_nodot}"
  # Prior to Python 3.8 paths were suffixed with an 'm'
  if [[ -d  "\${python_path}/bin" ]]; then
    export PATH="\${python_path}/bin:\$PATH"
  elif [[ -d "\${python_path}m/bin" ]]; then
    export PATH="\${python_path}m/bin:\$PATH"
  fi
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
    retry conda install -y cpuonly -c pytorch
  fi
  retry conda install -yq future numpy protobuf six
  if [[ "$DESIRED_CUDA" != 'cpu' ]]; then
    # DESIRED_CUDA is in format cu90 or cu102
    if [[ "${#DESIRED_CUDA}" == 4 ]]; then
      cu_ver="${DESIRED_CUDA:2:1}.${DESIRED_CUDA:3}"
    else
      cu_ver="${DESIRED_CUDA:2:2}.${DESIRED_CUDA:4}"
    fi
    retry conda install -yq -c pytorch "cudatoolkit=\${cu_ver}"
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
