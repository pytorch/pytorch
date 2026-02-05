#!/usr/bin/env bash

set -eux

torch_version=$(unzip -p torch-* '**/METADATA' | grep '^Version: ' | cut -d' ' -f2)
nightly=$(echo ${torch_version} | cut -d'.' -f4)

# Copied from .ci/manywheel/build_common.sh
make_wheel_record() {
  fpath=$1
  if echo $fpath | grep RECORD >/dev/null 2>&1; then
    echo "$fpath,,"
  else
    fhash=$(openssl dgst -sha256 -binary $fpath | openssl base64 | sed -e 's/+/-/g' | sed -e 's/\//_/g' | sed -e 's/=//g')
    fsize=$(ls -nl $fpath | awk '{print $5}')
    echo "$fpath,sha256=$fhash,$fsize"
  fi
}

change_wheel_version() {
  local package=$1
  local wheel=$2
  local f_version=$3
  local t_version=$4

  # Extract the wheel
  ${PYTHON_EXECUTABLE} -mwheel unpack $wheel

  mv "${package}-${f_version}" "${package}-${t_version}"
  # Change the version from f_version to t_version in the dist-info dir
  pushd "${package}-${t_version}"
  mv "${package}-${f_version}.dist-info" "${package}-${t_version}.dist-info"

  pushd "${package}-${t_version}.dist-info"
  sed -i "s/${package}-${f_version}.dist-info/${package}-${t_version}.dist-info/g" RECORD

  # Update the version in METADATA and its SHA256 hash
  sed -i "s/Version: ${f_version}/Version: ${t_version}/g" METADATA
  # then add PyTorch nightly dependency of vLLM
  if [[ "${package}" == vllm ]]; then
    sed -i "/License-File/a\Requires-Dist: torch==${torch_version}" METADATA
  fi
  sed -i '/METADATA,sha256/d' RECORD
  popd

  make_wheel_record "${package}-${t_version}.dist-info/METADATA" >> "${package}-${t_version}.dist-info/RECORD"
  popd

  # Repack the wheel
  ${PYTHON_EXECUTABLE} -mwheel pack "${package}-${t_version}"

  # Clean up
  rm -rf "${package}-${t_version}"
}

repackage_wheel() {
  local package=$1
  pushd $package

  local orig_wheel=$(find . -name *${package//-/_}*)
  local orig_version=$(unzip -p $orig_wheel '**/METADATA' | grep '^Version: ' | cut -d' ' -f2)

  local version=""
  if [[ "${package}" == vllm ]]; then
    # Copied from vllm/.buildkite/scripts/upload-wheels.sh
    version=1.0.0
  else
    version=$(echo $orig_version | tr '.+' '.' | cut -d'.' -f1-3)
  fi
  local nightly_version=$version.$nightly

  # Use nightly version
  change_wheel_version ${package//-/_} $orig_wheel $orig_version $nightly_version
  # Clean up
  rm "${orig_wheel}"

  auditwheel repair --plat $PLATFORM *.whl \
    --exclude libc10* --exclude libtorch* --exclude libcu* --exclude libnv*
  local repair_wheel=$(find wheelhouse -name *${PLATFORM}*)
  local repair_wheel=$(basename ${repair_wheel})
  popd

  cp ${package}/wheelhouse/${repair_wheel} .
  rm -rf $package
}

# Require to re-package the wheel
${PYTHON_EXECUTABLE} -mpip install wheel==0.45.1

pushd externals/vllm/wheels
for package in vllm; do
  repackage_wheel $package
done
popd
