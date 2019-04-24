#!/bin/bash

source /home/circleci/project/env
cat >/home/circleci/project/ci_test_script.sh <<EOL
# =================== The following code will be executed inside Docker container ===================
set -ex

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
# locally
pkg="/final_pkgs/\$(ls /final_pkgs)"
if [[ "$PACKAGE_TYPE" == conda ]]; then
  conda install -y "\$pkg" --offline
else
  pip install "\$pkg"
fi

# Test the package
pushd /pytorch
/builder/run_tests.sh "$PACKAGE_TYPE" "$DESIRED_PYTHON" "$DESIRED_CUDA"
# =================== The above code will be executed inside Docker container ===================
EOL
echo "Prepared script to run in next step"
cat /home/circleci/project/ci_test_script.sh
