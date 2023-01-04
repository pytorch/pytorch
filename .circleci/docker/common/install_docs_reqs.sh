#!/bin/bash

set -ex

if [ -n "$KATEX" ]; then
  apt-get update
  # Ignore error if gpg-agent doesn't exist (for Ubuntu 16.04)
  apt-get install -y gpg-agent || :

  curl --retry 3 -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
  sudo apt-get install -y nodejs

  curl --retry 3 -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
  echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list

  apt-get update
  apt-get install -y --no-install-recommends yarn
  yarn global add katex --prefix /usr/local

  sudo apt-get -y install doxygen

  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

fi
