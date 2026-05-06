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
  # Prerequisites for fetching the git-core PPA signing key and serving it via signed-by
  apt-get install -y --no-install-recommends software-properties-common gpg-agent gnupg ca-certificates
  # Add git-core PPA for a newer version of git. Bypass `add-apt-repository`, which
  # talks to api.launchpad.net to fetch the signing key and is unreliable when that
  # endpoint is degraded. Pull the key from keyserver.ubuntu.com instead and pin it
  # to the repo via signed-by. Treat any failure here as fatal so we don't silently
  # fall back to the distro's older git, which then breaks downstream steps that
  # depend on newer git features (e.g. `git submodule update --filter=tree:0`).
  mkdir -p /etc/apt/keyrings
  gpg --no-default-keyring --keyring /etc/apt/keyrings/git-core.gpg \
      --keyserver hkps://keyserver.ubuntu.com \
      --recv-keys E1DD270288B4E6030699E45FA1715D88E1DF1F24
  UBUNTU_CODENAME=$(. /etc/os-release && echo "$UBUNTU_CODENAME")
  echo "deb [signed-by=/etc/apt/keyrings/git-core.gpg] https://ppa.launchpadcontent.net/git-core/ppa/ubuntu/ ${UBUNTU_CODENAME} main" \
      > /etc/apt/sources.list.d/git-core.list
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
    libssl-dev \
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

  # Hard-fail if git is missing or older than 2.36 (the version required for
  # `git submodule update --filter=tree:0`, which actions/checkout invokes).
  # This catches the case where the git-core PPA setup silently regressed and
  # apt fell back to the distro's older git.
  if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git was not installed" >&2
    exit 1
  fi
  GIT_VERSION=$(git --version | awk '{print $3}')
  GIT_MAJOR=${GIT_VERSION%%.*}
  GIT_MINOR=${GIT_VERSION#*.}; GIT_MINOR=${GIT_MINOR%%.*}
  if (( GIT_MAJOR < 2 || (GIT_MAJOR == 2 && GIT_MINOR < 36) )); then
    echo "ERROR: git ${GIT_VERSION} is too old; need >= 2.36 (git-core PPA)" >&2
    exit 1
  fi

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
