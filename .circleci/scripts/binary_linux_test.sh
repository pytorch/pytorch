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
if [[ "$PACKAGE_TYPE" != libtorch ]]; then
  python_path="/opt/python/cp\$python_nodot-cp\${python_nodot}"
  if [[ "\$python_nodot" = *t ]]; then
    python_digits="\$(echo $DESIRED_PYTHON | tr -cd [:digit:])"
    python_path="/opt/python/cp\$python_digits-cp\${python_digits}t"
  fi
  export PATH="\${python_path}/bin:\$PATH"
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

if [[ "\$python_nodot" = *39* ]]; then
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
# Pick only one package of multiple available (which happens as result of workflow re-runs)
pkg="/final_pkgs/\$(ls -1 /final_pkgs|sort|tail -1)"
if [[ "\$PYTORCH_BUILD_VERSION" == *dev* ]]; then
    CHANNEL="nightly"
else
    CHANNEL="test"
fi

if [[ "$PACKAGE_TYPE" != libtorch ]]; then
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
/pytorch/.ci/pytorch/check_binary.sh

if [[ "\$GPU_ARCH_TYPE" != *s390x* && "\$GPU_ARCH_TYPE" != *xpu* && "\$GPU_ARCH_TYPE" != *rocm*  && "$PACKAGE_TYPE" != libtorch ]]; then
  # Exclude s390, xpu, rocm and libtorch builds from smoke testing
  python /pytorch/.ci/pytorch/smoke_test/smoke_test.py --package=torchonly --torch-compile-check disabled
fi


# =================== The above code will be executed inside Docker container ===================
EOL
echo
echo
echo "The script that will run in the next step is:"
cat "${OUTPUT_SCRIPT}"
