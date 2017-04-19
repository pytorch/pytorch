# Caffe2

Caffe2 is a deep learning framework made with expression, speed, and modularity in mind. It is an experimental refactoring of Caffe, and allows a more flexible way to organize computation.

## License and Citation

Caffe2 is released under the [BSD 2-Clause license](https://github.com/Yangqing/caffe2/blob/master/LICENSE).

## Building Caffe2

[![Travis Build Status](https://travis-ci.org/caffe2/caffe2.svg?branch=master)](https://travis-ci.org/caffe2/caffe2)

[![Windows Build status](https://ci.appveyor.com/api/projects/status/kec4ta779stuyb83?svg=true)](https://ci.appveyor.com/project/Yangqing/caffe2)


Detailed build matrix (hit refresh if you see icons not showing up due to heroku):

| Target      | Status |
|-------------|----|
| Linux       | [![Build Linux](https://travis-matrix-badges.herokuapp.com/repos/caffe2/caffe2/branches/master/1)](https://travis-ci.org/caffe2/caffe2) |
| Android     | [![Build Android](https://travis-matrix-badges.herokuapp.com/repos/caffe2/caffe2/branches/master/3)](https://travis-ci.org/caffe2/caffe2) |
| iOS         | [![Build iOS](https://travis-matrix-badges.herokuapp.com/repos/caffe2/caffe2/branches/master/5)](https://travis-ci.org/caffe2/caffe2) |
| Linux + MKL | [![Build LinuxMKL](https://travis-matrix-badges.herokuapp.com/repos/caffe2/caffe2/branches/master/6)](https://travis-ci.org/caffe2/caffe2) |


    git clone --recursive https://github.com/caffe2/caffe2.git
    cd caffe2

### OS X

    brew install automake protobuf
    mkdir build && cd build
    cmake ..
    make

### Ubuntu

This build is confirmed for:

* Ubuntu 14.04
* Ubuntu 16.04

#### Required Dependencies

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      libgoogle-glog-dev \
      libprotobuf-dev \
      protobuf-compiler \
      python-dev \
      python-pip                          
sudo pip install numpy protobuf
```

#### Optional GPU Support

If you plan to use GPU instead of CPU only, then you should install NVIDIA CUDA and cuDNN, a GPU-accelerated library of primitives for deep neural networks.
[NVIDIA's detailed instructions](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation) or if you're feeling lucky try the quick install set of commands below.

**Update your graphics card drivers first!** Otherwise you may suffer from a wide range of difficult to diagnose errors.

**For Ubuntu 14.04**

```bash
sudo apt-get update && sudo apt-get install wget -y --no-install-recommends
wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.61-1_amd64.deb"
sudo dpkg -i cuda-repo-ubuntu1404_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
```

**For Ubuntu 16.04**

```bash
sudo apt-get update && sudo apt-get install wget -y --no-install-recommends
wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb"
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
```

#### Install cuDNN (all Ubuntu versions)

```
CUDNN_URL="http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz"
wget ${CUDNN_URL}
sudo tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local
rm cudnn-8.0-linux-x64-v5.1.tgz && sudo ldconfig
```

#### Optional Dependencies

> Note `libgflags2` is for Ubuntu 14.04. `libgflags-dev` is for Ubuntu 16.04.

```bash
# for Ubuntu 14.04
sudo apt-get install -y --no-install-recommends libgflags2
```

```bash
# for Ubuntu 16.04
sudo apt-get install -y --no-install-recommends libgflags-dev
```

```bash
# for both Ubuntu 14.04 and 16.04
sudo apt-get install -y --no-install-recommends \
      libgtest-dev \
      libiomp-dev \
      libleveldb-dev \
      liblmdb-dev \
      libopencv-dev \
      libopenmpi-dev \
      libsnappy-dev \
      openmpi-bin \
      openmpi-doc \
      python-pydot
```

Check the Python section below and install optional packages before you build.

    mkdir build && cd build
    cmake ..
    make

### Android and iOS

We use CMake's Android and iOS ports to build native binaries that you can then integrate into your Android or XCode projects. See scripts/build_android.sh and scripts/build_ios.sh for more details.

For Android, one can also use gradle to build Caffe2 directly with Android Studio. An example project can be found [here](https://github.com/bwasti/AICamera). Note that you may need to configure Android Studio so that it has the right SDK and NDK versions to build the code.

### Raspberry Pi

For Raspbian, run scripts/build_raspbian.sh on the Raspberry Pi.

### Tegra X1

To install Caffe2 on NVidia's Tegra X1 platform, simply install the latest system with the NVidia JetPack installer, and then run scripts/build_tegra_x1.sh on the Tegra device.

## Python support

To run the tutorials you'll need ipython-notebooks and matplotlib, which can be installed on OS X with:

```    
brew install matplotlib --with-python3
pip install ipython notebook
```

You may also find these required for specific tutorials and examples, so you can run this to get all of the prerequisites at once:

```
sudo pip install \
      flask \
      graphviz \
      hypothesis \
      jupyter \
      matplotlib \
      pydot python-nvd3 \
      pyyaml \
      requests \
      scikit-image \
      scipy \
      setuptools \
      tornado
```

## Build status (known working)

Ubuntu 14.04 (GCC)
- [x] Default CPU build
- [x] Default GPU build

OS X (Clang)
- [x] Default CPU build
- [x] Default GPU build

Options (both Clang and GCC)
- [x] Nervana GPU
- [ ] ZMQ
- [x] RocksDB
- [x] MPI
- [x] OpenMP
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
- [x] List of dependencies for Ubuntu 16.04
- [x] List of dependencies for OS X
