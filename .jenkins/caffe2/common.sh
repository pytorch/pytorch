set -ex

LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$LOCAL_DIR"/../.. && pwd)
TEST_DIR="$ROOT_DIR/caffe2_tests"
gtest_reports_dir="${TEST_DIR}/cpp"
pytest_reports_dir="${TEST_DIR}/python"

# This is needed to work around ROCm using old docker images until
# the transition to new images is complete.
# TODO: Remove once ROCm CI is using new images.
if [[ $BUILD_ENVIRONMENT == py3.6-devtoolset7-rocmrpm-centos* ]]; then
  # This file is sourced multiple times, only install conda the first time.
  # We must install conda where we have write access.
  CONDA_DIR="$ROOT_DIR/conda"
  if [[ ! -d $CONDA_DIR ]]; then
    ANACONDA_PYTHON_VERSION=3.6
    BASE_URL="https://repo.anaconda.com/miniconda"
    CONDA_FILE="Miniconda3-latest-Linux-x86_64.sh"
    mkdir $CONDA_DIR
    pushd /tmp
    wget -q "${BASE_URL}/${CONDA_FILE}"
    chmod +x "${CONDA_FILE}"
    ./"${CONDA_FILE}" -b -f -p "$CONDA_DIR"
    popd
    export PATH="$CONDA_DIR/bin:$PATH"
    # Ensure we run conda in a directory that jenkins has write access to
    pushd $CONDA_DIR
    # Track latest conda update
    conda update -n base conda
    # Install correct Python version
    conda install python="$ANACONDA_PYTHON_VERSION"

    conda_install() {
      # Ensure that the install command don't upgrade/downgrade Python
      # This should be called as
      #   conda_install pkg1 pkg2 ... [-c channel]
      conda install -q -y python="$ANACONDA_PYTHON_VERSION" $*
    }

    # Install PyTorch conda deps, as per https://github.com/pytorch/pytorch README
    conda_install numpy pyyaml mkl mkl-include setuptools cffi typing future six

    # TODO: This isn't working atm
    conda_install nnpack -c killeent

    # Install some other packages

    # Need networkx 2.0 because bellmand_ford was moved in 2.1 . Scikit-image by
    # defaults installs the most recent networkx version, so we install this lower
    # version explicitly before scikit-image pulls it in as a dependency
    pip install networkx==2.0

    # TODO: Why is scipy pinned
    # numba & llvmlite is pinned because of https://github.com/numba/numba/issues/4368
    # scikit-learn is pinned because of
    # https://github.com/scikit-learn/scikit-learn/issues/14485 (affects gcc 5.5
    # only)
    pip install --progress-bar off pytest scipy==1.1.0 scikit-learn==0.20.3 scikit-image librosa>=0.6.2 psutil numba==0.46.0 llvmlite==0.30.0

    # click - onnx
    # hypothesis - tests
    # jupyter - for tutorials
    pip install --progress-bar off click hypothesis jupyter protobuf tabulate virtualenv mock typing-extensions

    popd
  else
    export PATH="$CONDA_DIR/bin:$PATH"
  fi
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
