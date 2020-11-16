#!/bin/bash

set -ex

[ -n "${VULKAN_SDK_VERSION}" ]

retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

_https_amazon_aws=https://ossci-android.s3.amazonaws.com

_vulkansdk_dir=/var/lib/jenkins/vulkansdk
mkdir -p $_vulkansdk_dir
_tmp_vulkansdk_targz=/tmp/vulkansdk.tar.gz
curl --silent --show-error --location --fail --retry 3 \
  --output "$_tmp_vulkansdk_targz" "$_https_amazon_aws/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz"

tar -C "$_vulkansdk_dir" -xzf "$_tmp_vulkansdk_targz" --strip-components 1

export VULKAN_SDK="$_vulkansdk_dir/"

rm "$_tmp_vulkansdk_targz"
