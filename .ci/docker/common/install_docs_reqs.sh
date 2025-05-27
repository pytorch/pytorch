#!/usr/bin/env bash
set -ex

###############################################################################
# Functions that hold distro-specific steps
###############################################################################
install_ubuntu() {
  # Original APT-based path (bumped Node to 18.x for current LTS)
  apt-get update
  apt-get install -y gpg-agent || :              # ignore if absent (16.04 quirk)

  curl --retry 3 -sL https://deb.nodesource.com/setup_18.x | sudo -E bash -
  sudo apt-get install -y nodejs

  curl --retry 3 -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
  echo "deb https://dl.yarnpkg.com/debian/ stable main" \
    | sudo tee /etc/apt/sources.list.d/yarn.list

  apt-get update
  apt-get install -y --no-install-recommends yarn
  yarn global add katex --prefix /usr/local

  sudo apt-get -y install doxygen

  apt-get autoclean -y && apt-get clean -y
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

install_centos() {
  # CentOS 9 / AlmaLinux 9 / Rocky 9 (dnf-based)
  dnf -y update

  curl --retry 3 -sL https://rpm.nodesource.com/setup_18.x | bash -
  dnf -y install nodejs

  curl --retry 3 -sL https://dl.yarnpkg.com/rpm/yarn.repo \
    | tee /etc/yum.repos.d/yarn.repo
  dnf -y install yarn

  yarn global add katex --prefix /usr/local
  dnf -y install doxygen

  dnf clean all
  rm -rf /var/cache/dnf/* /tmp/* /var/tmp/*
}

###############################################################################
# Main — only run if the KATEX flag is set
###############################################################################
if [[ -n "${KATEX:-}" ]]; then
  ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')

  case "$ID" in
    ubuntu|debian)
      install_ubuntu
      ;;
    centos|rocky|almalinux|rhel)
      install_centos
      ;;
    *)
      echo "Unable to determine OS for KaTeX install (ID='$ID')."
      exit 1
      ;;
  esac
fi
