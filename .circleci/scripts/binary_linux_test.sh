#!/bin/bash

OUTPUT_SCRIPT=${OUTPUT_SCRIPT:-/home/circleci/project/ci_test_script.sh}

# only source if file exists
if [[ -f /home/circleci/project/env ]]; then
  source /home/circleci/project/env
fi
cat >"${OUTPUT_SCRIPT}" <<EOL
# =================== The following code will be executed inside Docker container ===================
set -eux -o pipefail

retry () {
    "\$@"  || (sleep 1 && "\$@") || (sleep 2 && "\$@")
}

# Source binary env file here if exists
if [[ -e "${BINARY_ENV_FILE:-/nofile}" ]]; then
  source "${BINARY_ENV_FILE:-/nofile}"
fi

python_nodot="\$(echo $DESIRED_PYTHON | tr -d m.u)"

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
NUMPY_PIN=""
PROTOBUF_PACKAGE="defaults::protobuf"

if [[ "\$python_nodot" = *310* ]]; then
  # There's an issue with conda channel priority where it'll randomly pick 1.19 over 1.20
  # we set a lower boundary here just to be safe
  NUMPY_PIN=">=1.21.2"
  PROTOBUF_PACKAGE="protobuf>=3.19.0"
fi

if [[ "\$python_nodot" = *39*  ]]; then
  # There's an issue with conda channel priority where it'll randomly pick 1.19 over 1.20
  # we set a lower boundary here just to be safe
  NUMPY_PIN=">=1.20"
fi



# Move debug wheels out of the package dir so they don't get installed
mkdir -p /tmp/debug_final_pkgs
mv /final_pkgs/debug-*.zip /tmp/debug_final_pkgs || echo "no debug packages to move"

# Install the package
# These network calls should not have 'retry's because they are installing
# locally and aren't actually network calls
# TODO there is duplicated and inconsistent test-python-env setup across this
#   file, builder/smoke_test.sh, and builder/run_tests.sh, and also in the
#   conda build scripts themselves. These should really be consolidated
# Pick only one package of multiple available (which happens as result of workflow re-runs)
pkg="/final_pkgs/\$(ls -1 /final_pkgs|sort|tail -1)"
if [[ "\$PYTORCH_BUILD_VERSION" == *dev* ]]; then
    CHANNEL="nightly"
else
    CHANNEL="test"
fi

if [[ "$PACKAGE_TYPE" == conda ]]; then
  (
    # For some reason conda likes to re-activate the conda environment when attempting this install
    # which means that a deactivate is run and some variables might not exist when that happens,
    # namely CONDA_MKL_INTERFACE_LAYER_BACKUP from libblas so let's just ignore unbound variables when
    # it comes to the conda installation commands
    set +u
    retry conda install \${EXTRA_CONDA_FLAGS} -yq \
      "numpy\${NUMPY_PIN}" \
      mkl>=2018 \
      ninja \
      sympy \
      typing-extensions \
      ${PROTOBUF_PACKAGE}
    if [[ "$DESIRED_CUDA" == 'cpu' ]]; then
      retry conda install -c pytorch -y cpuonly
    else
      cu_ver="${DESIRED_CUDA:2:2}.${DESIRED_CUDA:4}"
      CUDA_PACKAGE="pytorch-cuda"
      retry conda install \${EXTRA_CONDA_FLAGS} -yq -c nvidia -c "pytorch-\${CHANNEL}" "pytorch-cuda=\${cu_ver}"
    fi
    conda install \${EXTRA_CONDA_FLAGS} -y "\$pkg" --offline
  )
elif [[ "$PACKAGE_TYPE" != libtorch ]]; then
  if [[ "\$BUILD_ENVIRONMENT" != *s390x* ]]; then
    if [[ "$USE_SPLIT_BUILD" == "true" ]]; then
      pkg_no_python="$(ls -1 /final_pkgs/torch_no_python* | sort |tail -1)"
      pkg_torch="$(ls -1 /final_pkgs/torch-* | sort |tail -1)"
      # todo: after folder is populated use the pypi_pkg channel instead
      pip install "\$pkg_no_python" "\$pkg_torch" --index-url "https://download.pytorch.org/whl/\${CHANNEL}/${DESIRED_CUDA}_pypi_pkg"
      retry pip install -q numpy protobuf typing-extensions
    else
      pip install "\$pkg" --index-url "https://download.pytorch.org/whl/\${CHANNEL}/${DESIRED_CUDA}"
      retry pip install -q numpy protobuf typing-extensions
    fi
  else
    pip install "\$pkg"
    retry pip install -q numpy protobuf typing-extensions
  fi
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
cat "${OUTPUT_SCRIPT}"
