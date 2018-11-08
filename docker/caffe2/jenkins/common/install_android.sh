#!/bin/bash

set -ex

apt-get update
apt-get install -y --no-install-recommends autotools-dev autoconf unzip
apt-get autoclean && apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

pushd /tmp
curl -Os https://dl.google.com/android/repository/android-ndk-r13b-linux-x86_64.zip
popd
_ndk_dir=/opt/ndk
mkdir -p "$_ndk_dir"
unzip -qo /tmp/android*.zip -d "$_ndk_dir"
_versioned_dir=$(find "$_ndk_dir/" -mindepth 1 -maxdepth 1 -type d)
mv "$_versioned_dir"/* "$_ndk_dir"/
rmdir "$_versioned_dir"
rm -rf /tmp/*
