set -ex

LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$LOCAL_DIR"/../.. && pwd)
TEST_DIR="$ROOT_DIR/caffe2_tests"
gtest_reports_dir="${TEST_DIR}/cpp"
pytest_reports_dir="${TEST_DIR}/python"

# This is needed to work around ROCm using old docker images until
# the transition to new images is complete.
if [[ $BUILD_ENVIRONMENT == py3.6-devtoolset7-rocmrpm-centos* ]]; then
  # this file is sourced multiple times, only install conda the first time
  if [[ ! -d /opt/conda ]]; then
    ANACONDA_PYTHON_VERSION=3.6
    $ROOT_DIR/.circleci/docker/common/install_conda.sh
  fi
  export PATH="/opt/conda/bin:${PATH}"
fi

# Figure out which Python to use
PYTHON="$(which python)"
if [[ "${BUILD_ENVIRONMENT}" =~ py((2|3)\.?[0-9]?\.?[0-9]?) ]]; then
  PYTHON=$(which "python${BASH_REMATCH[1]}")
fi

# /usr/local/caffe2 is where the cpp bits are installed to in in cmake-only
# builds. In +python builds the cpp tests are copied to /usr/local/caffe2 so
# that the test code in .jenkins/test.sh is the same
INSTALL_PREFIX="/usr/local/caffe2"

mkdir -p "$gtest_reports_dir" || true
mkdir -p "$pytest_reports_dir" || true
mkdir -p "$INSTALL_PREFIX" || true
