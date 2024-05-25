#!/bin/bash

set -ex

[ -n "${SWIFTSHADER}" ]

retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

_https_amazon_aws=https://ossci-android.s3.amazonaws.com

# SwiftShader
_swiftshader_dir=/var/lib/jenkins/swiftshader
_swiftshader_file_targz=swiftshader-abe07b943-prebuilt.tar.gz
mkdir -p $_swiftshader_dir
_tmp_swiftshader_targz="/tmp/${_swiftshader_file_targz}"

curl --silent --show-error --location --fail --retry 3 \
  --output "${_tmp_swiftshader_targz}" "$_https_amazon_aws/${_swiftshader_file_targz}"

tar -C "${_swiftshader_dir}" -xzf "${_tmp_swiftshader_targz}"

export VK_ICD_FILENAMES="${_swiftshader_dir}/build/Linux/vk_swiftshader_icd.json"
