# Caffe2

Caffe2 is a deep learning framework made with expression, speed, and modularity in mind. It is an experimental refactoring of Caffe, and allows a more flexible way to organize computation.

## License and Citation

Caffe2 is released under the [BSD 2-Clause license](https://github.com/Yangqing/caffe2/blob/master/LICENSE).

## Building Caffe2

[![Build Status](https://travis-ci.org/caffe2/caffe2.svg?branch=master)](https://travis-ci.org/caffe2/caffe2)

    git clone --recursive https://github.com/caffe2/caffe2.git
    cd caffe2
    
#### OS X
    
    brew install automake protobuf
    mkdir build && cd build
    cmake ..
    make

#### Ubuntu

    sudo apt-get install libprotobuf-dev protobuf-compiler libatlas-base-dev libgoogle-glog-dev libgtest-dev liblmdb-dev libleveldb-dev libsnappy-dev python-dev python-pip libiomp-dev libopencv-dev libpthread-stubs0-dev cmake python-protobuf
    sudo pip install numpy
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.44-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1404_8.0.44-1_amd64.deb
    sudo apt-get update
    sudo apt-get install cuda
    sudo apt-get install git

    CUDNN_URL="http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz" &&
    curl -fsSL ${CUDNN_URL} -O &&
    sudo tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local &&
    rm cudnn-8.0-linux-x64-v5.1.tgz &&
    sudo ldconfig
    
    mkdir build && cd build
    cmake ..
    make

#### Android and iOS

We use CMake's Android and iOS ports to build native binaries that you can then integrate into your Android or XCode projects. See scripts/build_android.sh and scripts/build_ios.sh for more details.

For Android, one can also use gradle to build Caffe2 directly with Android Studio. An example project can be found [here](https://github.com/bwasti/AICamera). Note that you may need to configure Android Studio so that it has the right SDK and NDK versions to build the code.

#### Raspberry Pi

For Raspbian, run scripts/build_raspbian.sh on the Raspberry Pi.

#### Tegra X1

To install Caffe2 on NVidia's Tegra X1 platform, simply install the latest system with the NVidia JetPack installer, and then run scripts/build_tegra_x1.sh on the Tegra device.
    
## Python support

To run the tutorials you'll need ipython-notebooks and matplotlib, which can be installed on OS X with:
    
    brew install matplotlib --with-python3
    pip install ipython notebook

## Build status (known working)

Ubuntu 14.04 (GCC)
- [x] Default CPU build
- [x] Default GPU build

OS X (Clang)
- [x] Default CPU build
- [x] Default GPU build

Options (both Clang and GCC)
- [ ] Nervana GPU
- [ ] ZMQ
- [ ] RocksDB
- [ ] MPI
- [ ] OpenMP
- [x] No LMDB
- [x] No LevelDB
- [x] No OpenCV

BLAS
- [x] OpenBLAS
- [x] ATLAS
- [ ] MKL

Other
- [x] CMake 2.8 support
- [x] List of dependencies for Ubuntu 14.04
- [x] List of dependencies for OS X
