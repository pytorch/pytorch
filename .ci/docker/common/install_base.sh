#!/bin/bash

set -ex

install_ubuntu() {
  # NVIDIA dockers for RC releases use tag names like `11.0-cudnn9-devel-ubuntu18.04-rc`,
  # for this case we will set UBUNTU_VERSION to `18.04-rc` so that the Dockerfile could
  # find the correct image. As a result, here we have to check for
  #   "$UBUNTU_VERSION" == "18.04"*
  # instead of
  #   "$UBUNTU_VERSION" == "18.04"
  if [[ "$UBUNTU_VERSION" == "20.04"* ]]; then
    cmake3="cmake=3.16*"
  elif [[ "$UBUNTU_VERSION" == "22.04"* ]]; then
    cmake3="cmake=3.22*"
  elif [[ "$UBUNTU_VERSION" == "24.04"* ]]; then
    cmake3="cmake=3.28*"
  else
    echo "Unknown Ubuntu version $UBUNTU_VERSION"
    exit 1
  fi

  # Install common dependencies
  apt-get update
  # Install prerequisites for add-apt-repository (needs gpg-agent for PPA key import)
  apt-get install -y --no-install-recommends software-properties-common gpg-agent
  # Add git-core PPA for a newer version of git
  add-apt-repository ppa:git-core/ppa -y
  apt-get update
  # TODO: Some of these may not be necessary
  deploy_deps="libffi-dev libbz2-dev libreadline-dev libncurses5-dev libncursesw5-dev libgdbm-dev libsqlite3-dev uuid-dev tk-dev"
  numpy_deps="gfortran"
  apt-get install -y --no-install-recommends \
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
    libyaml-dev \
    libz-dev \
    libjemalloc2 \
    libgl1 \
    libjpeg-dev \
    libasound2-dev \
    libsndfile-dev \
    software-properties-common \
    wget \
    sudo \
    vim \
    jq \
    libtool \
    vim \
    unzip \
    gpg-agent \
    gdb \
    bc \
    zip \
    valgrind

  # Should resolve issues related to various apt package repository cert issues
  # see: https://github.com/pytorch/pytorch/issues/65931
  apt-get install -y libgnutls30

  # Cleanup package manager
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

# Install base packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    install_ubuntu
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac
