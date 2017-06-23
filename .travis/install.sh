#!/bin/bash
set -e
set -x

APT_INSTALL_CMD='sudo apt-get install -y --no-install-recommends'

if [ "$TRAVIS_OS_NAME" = 'linux' ]; then
    ####################
    # apt dependencies #
    ####################
    sudo apt-get update
    $APT_INSTALL_CMD \
        asciidoc \
        autoconf \
        automake \
        build-essential \
        ca-certificates \
        ccache \
        docbook-xml \
        docbook-xsl \
        git \
        gperf \
        libatlas-base-dev \
        libgoogle-glog-dev \
        libiomp-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libpthread-stubs0-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-pip \
        software-properties-common \
        xsltproc

    # Install ccache symlink wrappers
    pushd /usr/local/bin
    sudo ln -sf "$(which ccache)" gcc
    sudo ln -sf "$(which ccache)" g++
    popd

    if [ "$BUILD_GCC5" = 'true' ]; then
        ################
        # Install GCC5 #
        ################
        sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
        sudo apt-get update
        $APT_INSTALL_CMD g++-5
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 60 \
            --slave /usr/bin/g++ g++ /usr/bin/g++-5
    fi

    if [ "$BUILD_CUDA" = 'true' ]; then
        ##################
        # Install ccache #
        ##################
        # Needs specific branch to work with nvcc (ccache/ccache#145)
        if [ -e "${BUILD_CCACHE_DIR}/ccache" ]; then
            echo "Using cached ccache build at \"$BUILD_CCACHE_DIR\" ..."
        else
            git clone https://github.com/colesbury/ccache -b ccbin "$BUILD_CCACHE_DIR"
            pushd "$BUILD_CCACHE_DIR"
            ./autogen.sh
            ./configure
            make "-j$(nproc)"
            popd
        fi

        # Overwrite ccache symlink wrappers
        pushd /usr/local/bin
        sudo ln -sf "${BUILD_CCACHE_DIR}/ccache" gcc
        sudo ln -sf "${BUILD_CCACHE_DIR}/ccache" g++
        sudo ln -sf "${BUILD_CCACHE_DIR}/ccache" nvcc
        popd

        #################
        # Install CMake #
        #################
        # Newer version required to get cmake+ccache+nvcc to work
        _cmake_installer=/tmp/cmake.sh
        wget -O "$_cmake_installer" https://cmake.org/files/v3.8/cmake-3.8.2-Linux-x86_64.sh
        sudo bash "$_cmake_installer" --prefix=/usr/local --skip-license
        rm -rf "$_cmake_installer"

        ################
        # Install CUDA #
        ################
        CUDA_REPO_PKG='cuda-repo-ubuntu1404_8.0.44-1_amd64.deb'
        CUDA_PKG_VERSION='8-0'
        CUDA_VERSION='8.0'
        wget "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/${CUDA_REPO_PKG}"
        sudo dpkg -i "$CUDA_REPO_PKG"
        rm -f "$CUDA_REPO_PKG"
        sudo apt-get update
        $APT_INSTALL_CMD \
            "cuda-core-${CUDA_PKG_VERSION}" \
            "cuda-cublas-dev-${CUDA_PKG_VERSION}" \
            "cuda-cudart-dev-${CUDA_PKG_VERSION}" \
            "cuda-curand-dev-${CUDA_PKG_VERSION}" \
            "cuda-driver-dev-${CUDA_PKG_VERSION}" \
            "cuda-nvrtc-dev-${CUDA_PKG_VERSION}"
        # Manually create CUDA symlink
        sudo ln -sf /usr/local/cuda-$CUDA_VERSION /usr/local/cuda

        #################
        # Install cuDNN #
        #################
        CUDNN_REPO_PKG='nvidia-machine-learning-repo-ubuntu1404_4.0-2_amd64.deb'
        CUDNN_PKG_VERSION='6.0.20-1+cuda8.0'
        wget "https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/${CUDNN_REPO_PKG}"
        sudo dpkg -i "$CUDNN_REPO_PKG"
        rm -f "$CUDNN_REPO_PKG"
        sudo apt-get update
        $APT_INSTALL_CMD \
            "libcudnn6=${CUDNN_PKG_VERSION}" \
            "libcudnn6-dev=${CUDNN_PKG_VERSION}"
    fi

    if [ "$BUILD_MKL" = 'true' ]; then
        ###############
        # Install MKL #
        ###############
        _mkl_key=/tmp/mkl.pub
        wget -O "$_mkl_key" http://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
        sudo apt-key add "$_mkl_key"
        rm -f "$_mkl_key"
        echo 'deb http://apt.repos.intel.com/mkl all main' | sudo tee /etc/apt/sources.list.d/intel-mkl.list
        sudo apt-get update
        $APT_INSTALL_CMD intel-mkl-64bit-2017.3-056
    fi
elif [ "$TRAVIS_OS_NAME" = 'osx' ]; then
    #####################
    # brew dependencies #
    #####################
    pip uninstall -y numpy  # use brew version (opencv dependency)
    brew tap homebrew/science  # for OpenCV
    brew install \
        ccache \
        glog \
        leveldb \
        lmdb \
        opencv \
        protobuf

    # Install ccache symlink wrappers
    pushd /usr/local/bin
    sudo ln -sf "$(which ccache)" clang
    sudo ln -sf "$(which ccache)" clang++
    popd
else
    echo "OS \"$TRAVIS_OS_NAME\" is unknown"
    exit 1
fi

####################
# pip dependencies #
####################
pip install numpy

if [ "$BUILD_ANDROID" = 'true' ]; then
    #######################
    # Install Android NDK #
    #######################
    _ndk_zip=/tmp/ndk.zip
    if [ "$TRAVIS_OS_NAME" = 'linux' ]; then
        $APT_INSTALL_CMD autotools-dev autoconf
        wget -O "$_ndk_zip" https://dl.google.com/android/repository/android-ndk-r13b-linux-x86_64.zip
    elif [ "$TRAVIS_OS_NAME" = 'osx' ]; then
        brew install libtool
        wget -O "$_ndk_zip" https://dl.google.com/android/repository/android-ndk-r13b-darwin-x86_64.zip
    else
        echo "OS \"$TRAVIS_OS_NAME\" is unknown"
        exit 1
    fi
    _ndk_dir=/opt/android_ndk
    sudo mkdir -p "$_ndk_dir"
    sudo chmod a+rwx "$_ndk_dir"
    unzip -qo "$_ndk_zip" -d "$_ndk_dir"
    rm -f "$_ndk_zip"
    _versioned_dir=$(find $_ndk_dir/ -mindepth 1 -maxdepth 1 -type d)
    mv "$_versioned_dir"/* "$_ndk_dir"/
    rmdir "$_versioned_dir"
fi

if [ "$BUILD_NNPACK" = 'true' ]; then
    #################
    # Install ninja #
    #################
    if [ "$TRAVIS_OS_NAME" = 'linux' ]; then
        # NNPACK needs a recent version
        if [ -e "${BUILD_NINJA_DIR}/ninja" ]; then
            echo "Using cached ninja build at \"$BUILD_NINJA_DIR\" ..."
        else
            git clone https://github.com/ninja-build/ninja.git -b release "$BUILD_NINJA_DIR"
            pushd "$BUILD_NINJA_DIR"
            python configure.py --bootstrap
            popd
        fi
        sudo install -m 755 "${BUILD_NINJA_DIR}/ninja" /usr/local/bin/ninja
    elif [ "$TRAVIS_OS_NAME" = 'osx' ]; then
        brew install ninja
    else
        echo "OS \"$TRAVIS_OS_NAME\" is unknown"
        exit 1
    fi
    pip install git+https://github.com/Maratyszcza/PeachPy
    pip install git+https://github.com/Maratyszcza/confu
fi
