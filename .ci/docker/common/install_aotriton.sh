#!/bin/bash

set -ex

ver() {
    printf "%3d%03d%03d%03d" $(echo "$1" | tr '.' ' ');
}

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

# This read command alwasy returns with exit code 1
read -d "\n" VER MANYLINUX ROCMLISTSTR PINNED_COMMIT SHA256LISTSTR Z < aotriton_version.txt || true
IFS=';' read -ra ROCMLIST <<< "$ROCMLISTSTR"
IFS=';' read -ra SHA256LIST <<< "$SHA256LISTSTR"
ROCM_LOW=$(echo "${ROCMLIST[0]}"| cut -b 5- -)
ROCM_HIGH=$(echo "${ROCMLIST[-1]}"| cut -b 5- -)
if [[ $(ver $ROCM_VERSION) -le $(ver $ROCM_LOW) ]]; then
    ROCMBASE=${ROCM_LOW}
elif [[ $(ver $ROCM_VERSION) -ge $(ver $ROCM_HIGH) ]]; then
    ROCMBASE=${ROCM_HIGH}
else
    ROCMBASE=${ROCM_VERSION}
fi
for i in "${!ROCMLIST[@]}"; do
   if [[ "${ROCMLIST[$i]}" = "rocm${ROCMBASE}" ]]; then
       SHA256="${SHA256LIST[$i]}"
   fi
done

ARCH=$(uname -m)
AOTRITON_INSTALL_PREFIX="$1"
AOTRITON_URL="https://github.com/ROCm/aotriton/releases/download/${VER}/aotriton-${VER}-${MANYLINUX}_${ARCH}-${ROCMBASE}-shared.tar.${Z}"
TARBALL="aotriton.tar.${Z}"
echo "Using AOTriton from ${AOTRITON_URL} expect sha256 ${SHA256}"

cd "${AOTRITON_INSTALL_PREFIX}"
# Must use -L to follow redirects
curl -L --retry 3 -o "${TARBALL}" "${AOTRITON_URL}"
ACTUAL_SHA256=$(sha256sum "${TARBALL}" | cut -d " " -f 1)
if [ "${SHA256}" != "${ACTUAL_SHA256}" ]; then
  echo -n "Error: The SHA256 of downloaded tarball is ${ACTUAL_SHA256},"
  echo " which does not match the expected value ${SHA256}."
  exit
fi
tar xf "${TARBALL}" && rm -rf "${TARBALL}"
