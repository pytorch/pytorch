#!/bin/bash

set -ex

install_ubuntu() {
  # NVIDIA dockers for RC releases use tag names like `11.0-cudnn8-devel-ubuntu18.04-rc`,
  # for this case we will set UBUNTU_VERSION to `18.04-rc` so that the Dockerfile could
  # find the correct image. As a result, here we have to check for
  #   "$UBUNTU_VERSION" == "18.04"*
  # instead of
  #   "$UBUNTU_VERSION" == "18.04"
  if [[ "$UBUNTU_VERSION" == "18.04"* ]]; then
    cmake3="cmake=3.10*"
    maybe_libiomp_dev="libiomp-dev"
  elif [[ "$UBUNTU_VERSION" == "20.04"* ]]; then
    cmake3="cmake=3.16*"
    maybe_libiomp_dev=""
  elif [[ "$UBUNTU_VERSION" == "22.04"* ]]; then
    cmake3="cmake=3.22*"
    maybe_libiomp_dev=""
  else
    cmake3="cmake=3.5*"
    maybe_libiomp_dev="libiomp-dev"
  fi

  if [[ "$CLANG_VERSION" == 12 ]]; then
    maybe_libomp_dev="libomp-12-dev"
  elif [[ "$CLANG_VERSION" == 10 ]]; then
    maybe_libomp_dev="libomp-10-dev"
  else
    maybe_libomp_dev=""
  fi

  # TODO: Remove this once nvidia package repos are back online
  # Comment out nvidia repositories to prevent them from getting apt-get updated, see https://github.com/pytorch/pytorch/issues/74968
  # shellcheck disable=SC2046
  sed -i 's/.*nvidia.*/# &/' $(find /etc/apt/ -type f -name "*.list")

  # Install common dependencies
  apt-get update
  # TODO: Some of these may not be necessary
  ccache_deps="asciidoc docbook-xml docbook-xsl xsltproc"
  deploy_deps="libffi-dev libbz2-dev libreadline-dev libncurses5-dev libncursesw5-dev libgdbm-dev libsqlite3-dev uuid-dev tk-dev"
  numpy_deps="gfortran"
  apt-get install -y --no-install-recommends \
    $ccache_deps \
    $numpy_deps \
    ${deploy_deps} \
    ${cmake3} \
    apt-transport-https \
    autoconf \
    automake \
    build-essential \
    ca-certificates \
    curl \
    git \
    libatlas-base-dev \
    libc6-dbg \
    ${maybe_libiomp_dev} \
    libyaml-dev \
    libz-dev \
    libjpeg-dev \
    libasound2-dev \
    libsndfile-dev \
    ${maybe_libomp_dev} \
    software-properties-common \
    wget \
    sudo \
    vim \
    jq \
    libtool \
    vim \
    unzip \
    gdb \
    libxml2-dev \
    libxslt-dev

  # Should resolve issues related to various apt package repository cert issues
  # see: https://github.com/pytorch/pytorch/issues/65931
  apt-get install -y libgnutls30

  # cuda-toolkit does not work with gcc-11.2.0 which is default in Ubunutu 22.04
  # see: https://github.com/NVlabs/instant-ngp/issues/119
  if [[ "$UBUNTU_VERSION" == "22.04"* ]]; then
    apt-get install -y g++-10
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 30
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 30
    update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-10 30

    # https://www.spinics.net/lists/libreoffice/msg07549.html
    sudo rm -rf /usr/lib/gcc/x86_64-linux-gnu/11
    wget https://github.com/gcc-mirror/gcc/commit/2b2d97fc545635a0f6aa9c9ee3b017394bc494bf.patch -O noexecpt.patch
    sudo patch  /usr/include/c++/10/bits/range_access.h noexecpt.patch
  fi

  # Cleanup package manager
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

install_centos() {
  # Need EPEL for many packages we depend on.
  # See http://fedoraproject.org/wiki/EPEL
  yum --enablerepo=extras install -y epel-release

  ccache_deps="asciidoc docbook-dtds docbook-style-xsl libxslt"
  numpy_deps="gcc-gfortran"
  # Note: protobuf-c-{compiler,devel} on CentOS are too old to be used
  # for Caffe2. That said, we still install them to make sure the build
  # system opts to build/use protoc and libprotobuf from third-party.
  yum install -y \
    $ccache_deps \
    $numpy_deps \
    autoconf \
    automake \
    bzip2 \
    cmake \
    cmake3 \
    curl \
    gcc \
    gcc-c++ \
    gflags-devel \
    git \
    glibc-devel \
    glibc-headers \
    glog-devel \
    hiredis-devel \
    libstdc++-devel \
    libsndfile-devel \
    make \
    opencv-devel \
    sudo \
    wget \
    vim \
    unzip \
    gdb

  # Cleanup
  yum clean all
  rm -rf /var/cache/yum
  rm -rf /var/lib/yum/yumdb
  rm -rf /var/lib/yum/history
}

# Install base packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    install_ubuntu
    ;;
  centos)
    install_centos
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac

# Install Valgrind separately since the apt-get version is too old.
mkdir valgrind_build && cd valgrind_build
VALGRIND_VERSION=3.20.0
wget https://ossci-linux.s3.amazonaws.com/valgrind-${VALGRIND_VERSION}.tar.bz2
tar -xjf valgrind-${VALGRIND_VERSION}.tar.bz2
cd valgrind-${VALGRIND_VERSION}
./configure --prefix=/usr/local
make -j6
sudo make install
cd ../../
rm -rf valgrind_build
alias valgrind="/usr/local/bin/valgrind"
