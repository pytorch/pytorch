set -ex

LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$LOCAL_DIR"/../.. && pwd)
TEST_DIR="$ROOT_DIR/test"
pytest_reports_dir="${TEST_DIR}/test-reports/python"

# Figure out which Python to use
PYTHON="$(which python)"
if [[ "${BUILD_ENVIRONMENT}" =~ py((2|3)\.?[0-9]?\.?[0-9]?) ]]; then
  PYTHON=$(which "python${BASH_REMATCH[1]}")
fi

if [[ "${BUILD_ENVIRONMENT}" == *rocm* ]]; then
    # HIP_PLATFORM is auto-detected by hipcc; unset to avoid build errors
    unset HIP_PLATFORM
fi

mkdir -p "$pytest_reports_dir" || true
