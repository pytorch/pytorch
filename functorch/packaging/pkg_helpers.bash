# A set of useful bash functions for common functionality we need to do in
# many build scripts


# Setup CUDA environment variables, based on CU_VERSION
#
# Inputs:
#   CU_VERSION (cpu, cu92, cu100)
#   NO_CUDA_PACKAGE (bool)
#   BUILD_TYPE (conda, wheel)
#
# Outputs:
#   VERSION_SUFFIX (e.g., "")
#   PYTORCH_VERSION_SUFFIX (e.g., +cpu)
#   WHEEL_DIR (e.g., cu100/)
#   CUDA_HOME (e.g., /usr/local/cuda-9.2, respected by torch.utils.cpp_extension)
#   FORCE_CUDA (respected by torchvision setup.py)
#   NVCC_FLAGS (respected by torchvision setup.py)
#
# Precondition: CUDA versions are installed in their conventional locations in
# /usr/local/cuda-*
#
# NOTE: Why VERSION_SUFFIX versus PYTORCH_VERSION_SUFFIX?  If you're building
# a package with CUDA on a platform we support CUDA on, VERSION_SUFFIX ==
# PYTORCH_VERSION_SUFFIX and everyone is happy.  However, if you are building a
# package with only CPU bits (e.g., torchaudio), then VERSION_SUFFIX is always
# empty, but PYTORCH_VERSION_SUFFIX is +cpu (because that's how you get a CPU
# version of a Python package.  But that doesn't apply if you're on OS X,
# since the default CU_VERSION on OS X is cpu.
setup_cuda() {

  # First, compute version suffixes.  By default, assume no version suffixes
  export VERSION_SUFFIX=""
  export PYTORCH_VERSION_SUFFIX=""
  export WHEEL_DIR=""
  # Wheel builds need suffixes (but not if they're on OS X, which never has suffix)
  if [[ "$BUILD_TYPE" == "wheel" ]] && [[ "$(uname)" != Darwin ]]; then
    export PYTORCH_VERSION_SUFFIX="+$CU_VERSION"
    # Match the suffix scheme of pytorch, unless this package does not have
    # CUDA builds (in which case, use default)
    if [[ -z "$NO_CUDA_PACKAGE" ]]; then
      export VERSION_SUFFIX="$PYTORCH_VERSION_SUFFIX"
      export WHEEL_DIR="$CU_VERSION/"
    fi
  fi

  # Now work out the CUDA settings
  case "$CU_VERSION" in
    cu115)
      if [[ "$OSTYPE" == "msys" ]]; then
        export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5"
      else
        export CUDA_HOME=/usr/local/cuda-11.5/
      fi
      export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
      ;;
    cu113)
      if [[ "$OSTYPE" == "msys" ]]; then
        export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.3"
      else
        export CUDA_HOME=/usr/local/cuda-11.3/
      fi
      export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
      ;;
    cu112)
      if [[ "$OSTYPE" == "msys" ]]; then
        export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2"
      else
        export CUDA_HOME=/usr/local/cuda-11.2/
      fi
      export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
      ;;
    cu111)
      if [[ "$OSTYPE" == "msys" ]]; then
        export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1"
      else
        export CUDA_HOME=/usr/local/cuda-11.1/
      fi
      export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
      ;;
    cu110)
      if [[ "$OSTYPE" == "msys" ]]; then
        export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0"
      else
        export CUDA_HOME=/usr/local/cuda-11.0/
      fi
      export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0"
      ;;
    cu102)
      if [[ "$OSTYPE" == "msys" ]]; then
        export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2"
      else
        export CUDA_HOME=/usr/local/cuda-10.2/
      fi
      export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5"
      ;;
    cu101)
      if [[ "$OSTYPE" == "msys" ]]; then
        export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1"
      else
        export CUDA_HOME=/usr/local/cuda-10.1/
      fi
      export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5"
      ;;
    cu100)
      if [[ "$OSTYPE" == "msys" ]]; then
        export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0"
      else
        export CUDA_HOME=/usr/local/cuda-10.0/
      fi
      export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5"
      ;;
    cu92)
      if [[ "$OSTYPE" == "msys" ]]; then
        export CUDA_HOME="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.2"
      else
        export CUDA_HOME=/usr/local/cuda-9.2/
      fi
      export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0"
      ;;
    cpu)
      ;;
    rocm*)
      export FORCE_CUDA=1
      ;;
    *)
      echo "Unrecognized CU_VERSION=$CU_VERSION"
      exit 1
      ;;
  esac
  if [[ -n "$CUDA_HOME" ]]; then
    # Adds nvcc binary to the search path so that CMake's `find_package(CUDA)` will pick the right one
    export PATH="$CUDA_HOME/bin:$PATH"
    export FORCE_CUDA=1
  fi
}

# Populate build version if necessary, and add version suffix
#
# Inputs:
#   BUILD_VERSION (e.g., 0.2.0 or empty)
#   VERSION_SUFFIX (e.g., +cpu)
#
# Outputs:
#   BUILD_VERSION (e.g., 0.2.0.dev20190807+cpu)
#
# Fill BUILD_VERSION if it doesn't exist already with a nightly string
# Usage: setup_build_version 0.2.0
setup_build_version() {
  if [[ -z "$BUILD_VERSION" ]]; then
    export BUILD_VERSION="$1.dev$(date "+%Y%m%d")$VERSION_SUFFIX"
  else
    export BUILD_VERSION="$BUILD_VERSION$VERSION_SUFFIX"
  fi

  # Set build version based on tag if on tag
  if [[ -n "${CIRCLE_TAG}" ]]; then
    # Strip tag
    export BUILD_VERSION="$(echo "${CIRCLE_TAG}" | sed -e 's/^v//' -e 's/-.*$//')${VERSION_SUFFIX}"
  fi
}

# Set some useful variables for OS X, if applicable
setup_macos() {
  if [[ "$(uname)" == Darwin ]]; then
    export MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++
  fi
}


# Top-level entry point for things every package will need to do
#
# Usage: setup_env 0.2.0
setup_env() {
  setup_cuda
  setup_build_version "$1"
  setup_macos
}

# Function to retry functions that sometimes timeout or have flaky failures
retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

# Inputs:
#   PYTHON_VERSION (3.7, 3.8, 3.9)
#   UNICODE_ABI (bool)
#
# Outputs:
#   PATH modified to put correct Python version in PATH
#
# Precondition: If Linux, you are in a soumith/manylinux-cuda* Docker image
setup_wheel_python() {
  if [[ "$(uname)" == Darwin || "$OSTYPE" == "msys" ]]; then
    eval "$(conda shell.bash hook)"
    conda env remove -n "env$PYTHON_VERSION" || true
    conda create ${CONDA_CHANNEL_FLAGS} -yn "env$PYTHON_VERSION" python="$PYTHON_VERSION"
    conda activate "env$PYTHON_VERSION"
    # Install libpng from Anaconda (defaults)
    conda install ${CONDA_CHANNEL_FLAGS} libpng "jpeg<=9b" -y
  else
    # Install native CentOS libJPEG, freetype and GnuTLS
    yum install -y libjpeg-turbo-devel freetype gnutls
    case "$PYTHON_VERSION" in
      3.7) python_abi=cp37-cp37m ;;
      3.8) python_abi=cp38-cp38 ;;
      3.9) python_abi=cp39-cp39 ;;
      3.10) python_abi=cp310-cp310 ;;
      *)
        echo "Unrecognized PYTHON_VERSION=$PYTHON_VERSION"
        exit 1
        ;;
    esac
    # Download all the dependencies required to compile image and video_reader
    # extensions

    mkdir -p ext_libraries
    pushd ext_libraries
    popd
    export PATH="/opt/python/$python_abi/bin:$(pwd)/ext_libraries/bin:$PATH"
  fi
}

# Install with pip a bit more robustly than the default
pip_install() {
  retry pip install --progress-bar off "$@"
}

# Install torch with pip, respecting PYTORCH_VERSION, and record the installed
# version into PYTORCH_VERSION, if applicable
setup_pip_pytorch_version() {
  if [[ -z "$PYTORCH_VERSION" ]]; then
    # Install latest prerelease version of torch, per our nightlies, consistent
    # with the requested cuda version
    pip_install --pre torch -f "https://download.pytorch.org/whl/nightly/${WHEEL_DIR}torch_nightly.html"
    if [[ "$CUDA_VERSION" == "cpu" ]]; then
      # CUDA and CPU are ABI compatible on the CPU-only parts, so strip
      # in this case
      export PYTORCH_VERSION="$(pip show torch | grep ^Version: | sed 's/Version:  *//' | sed 's/+.\+//')"
    else
      export PYTORCH_VERSION="$(pip show torch | grep ^Version: | sed 's/Version:  *//')"
    fi
  else
    pip_install "torch==$PYTORCH_VERSION$PYTORCH_VERSION_SUFFIX" \
      -f "https://download.pytorch.org/whl/${CU_VERSION}/torch_stable.html" \
      -f "https://download.pytorch.org/whl/${UPLOAD_CHANNEL}/${CU_VERSION}/torch_${UPLOAD_CHANNEL}.html"
  fi
}

# Fill PYTORCH_VERSION with the latest conda nightly version, and
# CONDA_CHANNEL_FLAGS with appropriate flags to retrieve these versions
#
# You MUST have populated PYTORCH_VERSION_SUFFIX before hand.
setup_conda_pytorch_constraint() {
  if [[ -z "$PYTORCH_VERSION" ]]; then
    export CONDA_CHANNEL_FLAGS="${CONDA_CHANNEL_FLAGS} -c pytorch-nightly -c pytorch"
    export PYTORCH_VERSION="$(conda search --json 'pytorch[channel=pytorch-nightly]' | \
                              python -c "import os, sys, json, re; cuver = os.environ.get('CU_VERSION'); \
                               cuver_1 = cuver.replace('cu', 'cuda') if cuver != 'cpu' else cuver; \
                               cuver_2 = (cuver[:-1] + '.' + cuver[-1]).replace('cu', 'cuda') if cuver != 'cpu' else cuver; \
                               print(re.sub(r'\\+.*$', '', \
                                [x['version'] for x in json.load(sys.stdin)['pytorch'] \
                                  if (x['platform'] == 'darwin' or cuver_1 in x['fn'] or cuver_2 in x['fn']) \
                                    and 'py' + os.environ['PYTHON_VERSION'] in x['fn']][-1]))")"
    if [[ -z "$PYTORCH_VERSION" ]]; then
      echo "PyTorch version auto detection failed"
      echo "No package found for CU_VERSION=$CU_VERSION and PYTHON_VERSION=$PYTHON_VERSION"
      exit 1
    fi
  else
    export CONDA_CHANNEL_FLAGS="${CONDA_CHANNEL_FLAGS} -c pytorch -c pytorch-${UPLOAD_CHANNEL}"
  fi
  if [[ "$CU_VERSION" == cpu ]]; then
    export CONDA_PYTORCH_BUILD_CONSTRAINT="- pytorch==$PYTORCH_VERSION${PYTORCH_VERSION_SUFFIX}"
    export CONDA_PYTORCH_CONSTRAINT="- pytorch==$PYTORCH_VERSION"
  else
    export CONDA_PYTORCH_BUILD_CONSTRAINT="- pytorch==${PYTORCH_VERSION}${PYTORCH_VERSION_SUFFIX}"
    export CONDA_PYTORCH_CONSTRAINT="- pytorch==${PYTORCH_VERSION}${PYTORCH_VERSION_SUFFIX}"
  fi
  if [[ "$OSTYPE" == msys && "$CU_VERSION" == cu92 ]]; then
    export CONDA_CHANNEL_FLAGS="${CONDA_CHANNEL_FLAGS} -c defaults -c numba/label/dev"
  fi
}

# Translate CUDA_VERSION into CUDA_CUDATOOLKIT_CONSTRAINT
setup_conda_cudatoolkit_constraint() {
  export CONDA_BUILD_VARIANT="cuda"
  if [[ "$(uname)" == Darwin ]]; then
    export CONDA_BUILD_VARIANT="cpu"
  else
    case "$CU_VERSION" in
      cu115)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=11.5,<11.6 # [not osx]"
        ;;
      cu113)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=11.3,<11.4 # [not osx]"
        ;;
      cu112)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=11.2,<11.3 # [not osx]"
        ;;
      cu111)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=11.1,<11.2 # [not osx]"
        ;;
      cu110)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=11.0,<11.1 # [not osx]"
        ;;
      cu102)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=10.2,<10.3 # [not osx]"
        ;;
      cu101)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=10.1,<10.2 # [not osx]"
        ;;
      cu100)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=10.0,<10.1 # [not osx]"
        ;;
      cu92)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=9.2,<9.3 # [not osx]"
        ;;
      cpu)
        export CONDA_CUDATOOLKIT_CONSTRAINT=""
        export CONDA_BUILD_VARIANT="cpu"
        ;;
      *)
        echo "Unrecognized CU_VERSION=$CU_VERSION"
        exit 1
        ;;
    esac
  fi
}

setup_conda_cudatoolkit_plain_constraint() {
  export CONDA_BUILD_VARIANT="cuda"
  export CMAKE_USE_CUDA=1
  if [[ "$(uname)" == Darwin ]]; then
    export CONDA_BUILD_VARIANT="cpu"
    export CMAKE_USE_CUDA=0
  else
    case "$CU_VERSION" in
      cu115)
        export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit=11.5"
        ;;
      cu113)
        export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit=11.3"
        ;;
      cu112)
        export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit=11.2"
        ;;
      cu111)
        export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit=11.1"
        ;;
      cu102)
        export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit=10.2"
        ;;
      cu101)
        export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit=10.1"
        ;;
      cu100)
        export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit=10.0"
        ;;
      cu92)
        export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit=9.2"
        ;;
      cpu)
        export CONDA_CUDATOOLKIT_CONSTRAINT=""
        export CONDA_BUILD_VARIANT="cpu"
        export CMAKE_USE_CUDA=0
        ;;
      *)
        echo "Unrecognized CU_VERSION=$CU_VERSION"
        exit 1
        ;;
    esac
  fi
}

# Build the proper compiler package before building the final package
setup_visual_studio_constraint() {
  if [[ "$OSTYPE" == "msys" ]]; then
      export VSTOOLCHAIN_PACKAGE=vs$VC_YEAR
      conda build $CONDA_CHANNEL_FLAGS --no-anaconda-upload packaging/$VSTOOLCHAIN_PACKAGE
      cp packaging/$VSTOOLCHAIN_PACKAGE/conda_build_config.yaml packaging/torchvision/conda_build_config.yaml
  fi
}

setup_junit_results_folder() {
  if [[ "$CI" == "true" ]]; then
    export CONDA_PYTORCH_BUILD_RESULTS_DIRECTORY="${SOURCE_ROOT_DIR}/build_results/results.xml"
  fi
}


download_copy_ffmpeg() {
  if [[ "$OSTYPE" == "msys" ]]; then
    # conda install -yq ffmpeg=4.2 -c pytorch
    # curl -L -q https://anaconda.org/pytorch/ffmpeg/4.3/download/win-64/ffmpeg-4.3-ha925a31_0.tar.bz2 --output ffmpeg-4.3-ha925a31_0.tar.bz2
    # bzip2 --decompress --stdout ffmpeg-4.3-ha925a31_0.tar.bz2 | tar -x --file=-
    # cp Library/bin/*.dll ../torchvision
    echo "FFmpeg is disabled currently on Windows"
  else
    if [[ "$(uname)" == Darwin ]]; then
      conda install -yq ffmpeg=4.2 -c pytorch
      conda install -yq wget
    else
      # pushd ext_libraries
      # wget -q https://anaconda.org/pytorch/ffmpeg/4.2/download/linux-64/ffmpeg-4.2-hf484d3e_0.tar.bz2
      # tar -xjvf ffmpeg-4.2-hf484d3e_0.tar.bz2
      # rm -rf ffmpeg-4.2-hf484d3e_0.tar.bz2
      # ldconfig
      # which ffmpeg
      # popd
      echo "FFmpeg is disabled currently on Linux"
    fi
  fi
}
