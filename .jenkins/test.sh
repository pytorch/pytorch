#!/bin/bash

set -ex

LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$LOCAL_DIR"/.. && pwd)

# Figure out which Python to use
PYTHON="python"
if [ -n "$BUILD_ENVIRONMENT" ]; then
  if [[ "$BUILD_ENVIRONMENT" == py2* ]]; then
    PYTHON="python2"
  elif [[ "$BUILD_ENVIRONMENT" == py3* ]]; then
    PYTHON="python3"
  fi
fi

# The prefix must mirror the setting from build.sh
INSTALL_PREFIX="/usr/local/caffe2"

# Anaconda builds have a special install prefix and python
if [[ "$BUILD_ENVIRONMENT" == conda* ]]; then
  # This path comes from install_anaconda.sh which installs Anaconda into the
  # docker image
  PYTHON="/opt/conda/bin/python"
  INSTALL_PREFIX="/opt/conda/"

  # Testing requires separate packages
  if [[ $BUILD_ENVIRONMENT == *gcc4* ]]; then
    # These are from conda-forge
    conda install -yc conda-forge hypothesis tabulate pydot networkx==2.0 click pytest scipy
    # These packages are from the default channels
    conda install -y opencv=3.1.0=np112py27_1 pil=1.1.7=py27_2
  else
    conda install -y hypothesis tabulate pydot
  fi

  # This build will be tested against onnx tests, which needs onnx installed.
  # Onnx should be built against the same protobuf that Caffe2 uses, which is
  # only installed in the conda environment when Caffe2 is.
  # This path comes from install_anaconda.sh which installs Anaconda into the
  # docker image
  PROTOBUF_INCDIR=/opt/conda/include pip install "${ROOT_DIR}/third_party/onnx"
fi

# Add the site-packages in the caffe2 install prefix to the PYTHONPATH
SITE_DIR=$($PYTHON -c "from distutils import sysconfig; print(sysconfig.get_python_lib(prefix=''))")
INSTALL_SITE_DIR="${INSTALL_PREFIX}/${SITE_DIR}"

# Skip tests in environments where they are not built/applicable
if [[ "${BUILD_ENVIRONMENT}" == *-android* ]]; then
  echo 'Skipping tests'
  exit 0
fi

# Set PYTHONPATH and LD_LIBRARY_PATH so that python can find the installed
# Caffe2. This shouldn't be done on Anaconda, as Anaconda should handle this.
if [[ "$BUILD_ENVIRONMENT" != conda* ]]; then
  export PYTHONPATH="${PYTHONPATH}:$INSTALL_SITE_DIR"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${INSTALL_PREFIX}/lib"
fi

exit_code=0

cd "$ROOT_DIR"/caffe2/python/tutorials
python tutorials_to_script_converter.py
git status
if git diff --quiet HEAD; then
  echo "Source tree is clean."
else
  echo "After running a tutorial -> script sync there are changes. This probably means you edited an ipython notebook without a proper sync to a script. Please see caffe2/python/tutorials/README.md for more information"
  if [ "$exit_code" -eq 0 ]; then
    exit_code=1
  fi
fi

cd "$ROOT_DIR"

if [ -d ./test ]; then
  echo "Directory ./test already exists; please remove it..."
  exit 1
fi

mkdir -p ./test/{cpp,python}
TEST_DIR="$PWD/test"

cd ${INSTALL_PREFIX}

# Commands below may exit with non-zero status
set +e

# C++ tests
echo "Running C++ tests.."
for test in ./test/*; do
  # Skip tests we know are hanging or bad
  case "$(basename "$test")" in
    mkl_utils_test)
      continue
      ;;
    # TODO investigate conv_op_test failures when using MKL
    conv_op_test)
      continue
      ;;
  esac

  "$test" --gtest_output=xml:"$TEST_DIR"/cpp/$(basename "$test").xml
  tmp_exit_code="$?"
  if [ "$exit_code" -eq 0 ]; then
    exit_code="$tmp_exit_code"
  fi
done

# Get the relative path to where the caffe2 python module was installed
CAFFE2_PYPATH="$INSTALL_SITE_DIR/caffe2"

# Collect additional tests to run (outside caffe2/python)
EXTRA_TESTS=()

# CUDA builds always include NCCL support
if [[ "$BUILD_ENVIRONMENT" == *-cuda* ]]; then
  EXTRA_TESTS+=("$CAFFE2_PYPATH/contrib/nccl")
fi

# Python tests
echo "Running Python tests.."
"$PYTHON" \
  -m pytest \
  -x \
  -v \
  --junit-xml="$TEST_DIR/python/result.xml" \
  --ignore "$CAFFE2_PYPATH/python/test/executor_test.py" \
  --ignore "$CAFFE2_PYPATH/python/operator_test/matmul_op_test.py" \
  --ignore "$CAFFE2_PYPATH/python/operator_test/pack_ops_test.py" \
  --ignore "$CAFFE2_PYPATH/python/mkl/mkl_sbn_speed_test.py" \
  "$CAFFE2_PYPATH/python" \
  "${EXTRA_TESTS[@]}"

tmp_exit_code="$?"
if [ "$exit_code" -eq 0 ]; then
  exit_code="$tmp_exit_code"
fi

# Exit with the first non-zero status we got
exit "$exit_code"
