#!/bin/bash

set -ex

[ -n "${VULKAN_SDK_VERSION}" ]

retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

_vulkansdk_dir=/var/lib/jenkins/vulkansdk
_tmp_vulkansdk_targz=/tmp/vulkansdk.tar.gz

curl \
  --silent \
  --show-error \
  --location \
  --fail \
  --retry 3 \
  --output "${_tmp_vulkansdk_targz}" "https://ossci-android.s3.amazonaws.com/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz"

mkdir -p "${_vulkansdk_dir}"
tar -C "${_vulkansdk_dir}" -xzf "${_tmp_vulkansdk_targz}" --strip-components 1
rm -rf "${_tmp_vulkansdk_targz}"
