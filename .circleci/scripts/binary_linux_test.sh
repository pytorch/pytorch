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

# Clone the Pytorch branch
git clone https://github.com/pytorch/pytorch.git /pytorch
pushd /pytorch
if [[ -n "$CIRCLE_PR_NUMBER" ]]; then
  # "smoke" binary build on PRs
  git fetch --force origin "pull/${CIRCLE_PR_NUMBER}/head:remotes/origin/pull/${CIRCLE_PR_NUMBER}"
  git reset --hard "$CIRCLE_SHA1"
  git checkout -q -B "$CIRCLE_BRANCH"
  git reset --hard "$CIRCLE_SHA1"
fi
git submodule update --init --recursive
popd

# Clone the Builder master repo
git clone -q https://github.com/pytorch/builder.git /builder

# Install the package
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
