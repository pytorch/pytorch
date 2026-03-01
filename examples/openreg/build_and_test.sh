#!/bin/bash
# Bash script to build and run OpenReg tests on Linux/macOS
# Usage: ./build_and_test.sh [test_name]
# Example: ./build_and_test.sh test_device.py
#          ./build_and_test.sh test_device.py::TestDevice::test_device_count

set -e

TEST_PATH="${1:=test_device.py}"
REBUILD="${REBUILD:=false}"
VERBOSE="${VERBOSE:=false}"

echo "OpenReg Build and Test Helper (Linux/macOS)"

# Get the repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OPENREG_DIR="$REPO_ROOT/test/cpp_extensions/open_registration_extension"
EXT_DIR="$OPENREG_DIR/torch_openreg"

echo "Repo root: $REPO_ROOT"
echo "OpenReg dir: $OPENREG_DIR"

# Check prerequisites
echo ""
echo "[1/4] Checking prerequisites..."

for cmd in cmake ninja python; do
    if command -v $cmd &> /dev/null; then
        version=$($cmd --version 2>&1 | head -1)
        echo "✓ $cmd found: $version"
    else
        echo "✗ $cmd NOT found"
        echo "  Install and add to PATH, then retry"
        exit 1
    fi
done

# Build the extension
echo ""
echo "[2/4] Building OpenReg extension..."

if [ "$REBUILD" = "true" ] || [ ! -d "$EXT_DIR/lib" ]; then
    echo "  Building (this may take a few minutes)..."
    cd "$OPENREG_DIR"
    python setup.py build_ext --inplace 2>&1 | tail -5
    if [ $? -ne 0 ]; then
        echo "✗ Build failed. See build.log for details."
        exit 1
    fi
    echo "✓ Build successful"
else
    echo "  (Using cached build, set REBUILD=true to force rebuild)"
fi

# Verify import
echo ""
echo "[3/4] Verifying OpenReg import..."

python3 << 'EOF'
import torch
import sys
sys.path.insert(0, '$OPENREG_DIR')
try:
    import torch_openreg
    print(f'✓ torch_openreg imported from: {torch_openreg.__file__}')
    count = torch.accelerator.device_count()
    print(f'✓ Device count: {count}')
except ImportError as e:
    print(f'✗ Import failed: {e}')
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "✗ Import failed"
    exit 1
fi

# Run tests
echo ""
echo "[4/4] Running tests..."
echo "  Test path: $TEST_PATH"

VERBOSE_FLAG="-v"
if [ "$VERBOSE" = "true" ]; then
    VERBOSE_FLAG="-vv"
fi

TEST_DIR="$OPENREG_DIR/torch_openreg/tests"

# Convert friendly test path to full path if needed
if [[ ! "$TEST_PATH" =~ "/" ]]; then
    TEST_PATH="$TEST_DIR/$TEST_PATH"
fi

echo "  Running: pytest $TEST_PATH $VERBOSE_FLAG"
echo ""

python -m pytest "$TEST_PATH" "$VERBOSE_FLAG" --tb=short

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Tests passed!"
else
    echo ""
    echo "✗ Some tests failed"
    echo "  For debugging, see failure_interpretation.md in docs/openreg/"
fi
