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
        rm -f ~/sccache_error.log || true
        SCCACHE_ERROR_LOG=~/sccache_error.log SCCACHE_IDLE_TIMEOUT=0 sccache --start-server

        # Report sccache stats for easier debugging
        sccache --zero-stats
    fi
fi

# /usr/local/caffe2 is where the cpp bits are installed to in cmake-only
# builds. In +python builds the cpp tests are copied to /usr/local/caffe2 so
# that the test code in .ci/test.sh is the same
INSTALL_PREFIX="/usr/local/caffe2"

mkdir -p "$gtest_reports_dir" || true
mkdir -p "$pytest_reports_dir" || true
mkdir -p "$INSTALL_PREFIX" || true
