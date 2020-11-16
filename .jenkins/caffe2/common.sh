set -ex

LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$LOCAL_DIR"/../.. && pwd)
TEST_DIR="$ROOT_DIR/test"
gtest_reports_dir="${TEST_DIR}/test-reports/cpp"
pytest_reports_dir="${TEST_DIR}/test-reports/python"

# Figure out which Python to use
PYTHON="$(which python)"
if [[ "${BUILD_ENVIRONMENT}" =~ py((2|3)\.?[0-9]?\.?[0-9]?) ]]; then
  PYTHON=$(which "python${BASH_REMATCH[1]}")
fi

if [[ "${BUILD_ENVIRONMENT}" == *rocm* ]]; then
    # HIP_PLATFORM is auto-detected by hipcc; unset to avoid build errors
    unset HIP_PLATFORM
    if which sccache > /dev/null; then
        # Save sccache logs to file
        sccache --stop-server || true
        rm ~/sccache_error.log || true
        SCCACHE_ERROR_LOG=~/sccache_error.log SCCACHE_IDLE_TIMEOUT=0 sccache --start-server

        # Report sccache stats for easier debugging
        sccache --zero-stats
    fi
fi

# /usr/local/caffe2 is where the cpp bits are installed to in in cmake-only
# builds. In +python builds the cpp tests are copied to /usr/local/caffe2 so
# that the test code in .jenkins/test.sh is the same
INSTALL_PREFIX="/usr/local/caffe2"

mkdir -p "$gtest_reports_dir" || true
mkdir -p "$pytest_reports_dir" || true
mkdir -p "$INSTALL_PREFIX" || true

# Use conda cmake in some CI build. Conda cmake will be newer than our supported
# min version (3.5 for xenial and 3.10 for bionic),
# so we only do it in four builds that we know should use conda.
# Linux bionic cannot find conda mkl with cmake 3.10, so we need a cmake from conda.
# Alternatively we could point cmake to the right place
# export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
if [[ "$BUILD_ENVIRONMENT" == *pytorch-xla-linux-bionic* ]] || \
   [[ "$BUILD_ENVIRONMENT" == *pytorch-linux-xenial-cuda9-cudnn7-py2* ]] || \
   [[ "$BUILD_ENVIRONMENT" == *pytorch-linux-xenial-cuda10.1-cudnn7-py3* ]] || \
   [[ "$BUILD_ENVIRONMENT" == *pytorch-*centos* ]] || \
   [[ "$BUILD_ENVIRONMENT" == *pytorch-linux-bionic* ]]; then
  if ! which conda; then
    echo "Expected ${BUILD_ENVIRONMENT} to use conda, but 'which conda' returns empty"
    exit 1
  else
    conda install -q -y cmake
  fi
fi
