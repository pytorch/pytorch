#!/bin/bash

set -ex

if [ -n "$KATEX" ]; then
  # Should resolve issues related to deb.nodesource.com cert issues
  # see: https://github.com/pytorch/pytorch/issues/65931
  apt-get update
  apt-get install -y libgnutls30

  curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
  sudo apt-get install -y nodejs

  curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
  echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list

  apt-get update
  apt-get install -y --no-install-recommends yarn
  yarn global add katex --prefix /usr/local

  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

fi
