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
    gdb

  # Should resolve issues related to various apt package repository cert issues
  # see: https://github.com/pytorch/pytorch/issues/65931
  apt-get install -y libgnutls30

  # Required to install the fortran after gcc update
  if [[ "$UBUNTU_VERSION" == "22.04"* ]]; then
    apt autoremove -y gfortran
    apt-get update -y
    apt-get install -y gfortran libopenblas-dev
  fi

  # Cleanup package manager
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

build_libpng() {
  # install few packages
  yum install -y zlib zlib-devel

  LIBPNG_VERSION=1.6.37

  mkdir -p libpng
  pushd libpng

  wget http://download.sourceforge.net/libpng/libpng-$LIBPNG_VERSION.tar.gz
  tar -xvzf libpng-$LIBPNG_VERSION.tar.gz

  pushd libpng-$LIBPNG_VERSION

  ./configure
  make
  make install

  popd

  popd
  rm -rf libpng
}

install_centos() {
  # Need EPEL for many packages we depend on.
  # See http://fedoraproject.org/wiki/EPEL
  # extras repo is not there for CentOS 9 and epel-release is already part of repo list
  if [[ $OS_VERSION == 9 ]]; then
      yum install -y epel-release
      ALLOW_ERASE="--allowerasing"
  else
      yum --enablerepo=extras install -y epel-release
      ALLOW_ERASE=""
  fi

  ccache_deps="asciidoc docbook-dtds docbook-style-xsl libxslt"
  numpy_deps="gcc-gfortran"
  # Note: protobuf-c-{compiler,devel} on CentOS are too old to be used
  # for Caffe2. That said, we still install them to make sure the build
  # system opts to build/use protoc and libprotobuf from third-party.
  yum install -y $ALLOW_ERASE \
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
    make \
    sudo \
    wget \
    vim \
    unzip \
    gdb

  if [[ $OS_VERSION == 9 ]]
  then
	  dnf --enablerepo=crb -y install libsndfile-devel
  else
	  yum install -y \
            opencv-devel \
	    libsndfile-devel
  fi

  # CentOS7 doesnt have support for higher version of libpng,
  # so it is built from source.
  # Libpng is required for torchvision build.
  build_libpng

  # Cleanup
  yum clean all
  rm -rf /var/cache/yum
  rm -rf /var/lib/yum/yumdb
  rm -rf /var/lib/yum/history
}

ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
OS_VERSION=$(grep -oP '(?<=^VERSION_ID=).+' /etc/os-release | tr -d '"')

# Install base packages depending on the base OS
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
