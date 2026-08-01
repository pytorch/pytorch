#!/bin/bash

set -ex

arch_path=''
targetarch=${TARGETARCH:-$(uname -m)}
if [ ${targetarch} = 'amd64' ] || [ "${targetarch}" = 'x86_64' ]; then
  arch_path='x86_64'
else
  arch_path='sbsa'
fi

NVSHMEM_VERSION=3.4.5
CUDA_CUPTI_VERSION=13.3.75

function install_cuda {
  version=$1
  runfile=$2
  major_minor=${version%.*}
  rm -rf /usr/local/cuda-${major_minor} /usr/local/cuda
  if [[ ${arch_path} == 'sbsa' ]]; then
      runfile="${runfile}_sbsa"
  fi
  runfile="${runfile}.run"
  wget -q https://developer.download.nvidia.com/compute/cuda/${version}/local_installers/${runfile} -O ${runfile}
  chmod +x ${runfile}
  ./${runfile} --toolkit --silent
  rm -f ${runfile}
  rm -f /usr/local/cuda && ln -s /usr/local/cuda-${major_minor} /usr/local/cuda
}

function install_cudnn {
  cuda_major_version=$1
  cudnn_version=$2
  mkdir tmp_cudnn && cd tmp_cudnn
  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  filepath="cudnn-linux-${arch_path}-${cudnn_version}_cuda${cuda_major_version}-archive"
  wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-${arch_path}/${filepath}.tar.xz
  tar xf ${filepath}.tar.xz
  cp -a ${filepath}/include/* /usr/local/cuda/include/
  cp -a ${filepath}/lib/* /usr/local/cuda/lib64/
  cd ..
  rm -rf tmp_cudnn
}

function install_nvshmem {
  cuda_major_version=$1      # e.g. "12"
  nvshmem_version=$2         # e.g. "3.3.9"

  case "${arch_path}" in
    sbsa)
      dl_arch="aarch64"
      ;;
    x86_64)
      dl_arch="x64"
      ;;
    *)
      dl_arch="${arch}"
      ;;
  esac

  tmpdir="tmp_nvshmem"
  mkdir -p "${tmpdir}" && cd "${tmpdir}"

  # nvSHMEM license: https://docs.nvidia.com/nvshmem/api/sla.html
  # This pattern is a lie as it is not consistent across versions, for 3.3.9 it was cuda_ver-arch-nvshhem-ver
  filename="libnvshmem-linux-${arch_path}-${nvshmem_version}_cuda${cuda_major_version}-archive"
  suffix=".tar.xz"
  url="https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-${arch_path}/${filename}${suffix}"

  # download, unpack, install
  wget -q "${url}"
  tar xf "${filename}${suffix}"
  cp -a "${filename}/include/"* /usr/local/cuda/include/
  cp -a "${filename}/lib/"*     /usr/local/cuda/lib64/

  # cleanup
  cd ..
  rm -rf "${tmpdir}"

  echo "nvSHMEM ${nvshmem_version} for CUDA ${cuda_major_version} (${arch_path}) installed."
}

function install_cupti_headers {
  cupti_version=$1                  # e.g. "13.3.75"
  major_minor=${cupti_version%.*}   # e.g. "13.3"
  target_dir="/usr/local/cupti-headers-${major_minor}"

  # The CUDA toolkit runfile ships an older CUPTI than the standalone redist
  # archive, so stage the newer headers in a non-default location where they are
  # available for inspection without poisoning the build's include search path.
  # Staged for every CUDA version so the binary-build Dockerfiles can COPY the
  # directory unconditionally. The headers are architecture independent, so
  # always grab the x86_64 archive.
  redist_url="https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/linux-x86_64"
  archive="cuda_cupti-linux-x86_64-${cupti_version}-archive"

  tmp_dir=$(mktemp -d)
  pushd "${tmp_dir}"
  wget -q "${redist_url}/${archive}.tar.xz"
  tar xf "${archive}.tar.xz"
  mkdir -p "${target_dir}"
  cp -a "${archive}/include/"* "${target_dir}/"
  popd

  rm -rf "${tmp_dir}"
  echo "CUPTI ${cupti_version} headers installed to ${target_dir}."
}

function install_124 {
  CUDNN_VERSION=9.1.0.70
  CUSPARSELT_VERSION=0.6.2.3
  echo "Installing CUDA 12.4.1 and cuDNN ${CUDNN_VERSION} and NCCL and cuSparseLt-${CUSPARSELT_VERSION}"
  install_cuda 12.4.1 cuda_12.4.1_550.54.15_linux

  install_cudnn 12 $CUDNN_VERSION

  CUDA_VERSION=12.4 bash install_nccl.sh

  CUDA_VERSION=12.4 bash install_cusparselt.sh $CUSPARSELT_VERSION

  ldconfig
}

function install_126 {
  CUDNN_VERSION=9.10.2.21
  CUSPARSELT_VERSION=0.7.1.0
  echo "Installing CUDA 12.6.3 and cuDNN ${CUDNN_VERSION} and NVSHMEM and NCCL and cuSparseLt-${CUSPARSELT_VERSION}"
  install_cuda 12.6.3 cuda_12.6.3_560.35.05_linux

  install_cudnn 12 $CUDNN_VERSION

  install_nvshmem 12 $NVSHMEM_VERSION

  CUDA_VERSION=12.6 bash install_nccl.sh

  CUDA_VERSION=12.6 bash install_cusparselt.sh $CUSPARSELT_VERSION

  ldconfig
}

function install_129 {
  CUDNN_VERSION=9.24.0.43
  CUSPARSELT_VERSION=0.8.1.1
  echo "Installing CUDA 12.9.1 and cuDNN ${CUDNN_VERSION} and NVSHMEM and NCCL and cuSparseLt-${CUSPARSELT_VERSION}"
  # install CUDA 12.9.1 in the same container
  install_cuda 12.9.1 cuda_12.9.1_575.57.08_linux

  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  install_cudnn 12 $CUDNN_VERSION

  install_nvshmem 12 $NVSHMEM_VERSION

  CUDA_VERSION=12.9 bash install_nccl.sh

  CUDA_VERSION=12.9 bash install_cusparselt.sh $CUSPARSELT_VERSION

  ldconfig
}

function install_128 {
  CUDNN_VERSION=9.24.0.43
  CUSPARSELT_VERSION=0.7.1.0
  echo "Installing CUDA 12.8.1 and cuDNN ${CUDNN_VERSION} and NVSHMEM and NCCL and cuSparseLt-${CUSPARSELT_VERSION}"
  # install CUDA 12.8.1 in the same container
  install_cuda 12.8.1 cuda_12.8.1_570.124.06_linux

  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  install_cudnn 12 $CUDNN_VERSION

  install_nvshmem 12 $NVSHMEM_VERSION

  CUDA_VERSION=12.8 bash install_nccl.sh

  CUDA_VERSION=12.8 bash install_cusparselt.sh $CUSPARSELT_VERSION

  ldconfig
}

function install_130 {
  CUDNN_VERSION=9.24.0.43
  CUSPARSELT_VERSION=0.8.1.1
  echo "Installing CUDA 13.0 and cuDNN ${CUDNN_VERSION} and NVSHMEM and NCCL and cuSparseLt-${CUSPARSELT_VERSION}"
  # install CUDA 13.0 in the same container
  install_cuda 13.0.2 cuda_13.0.2_580.95.05_linux

  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  install_cudnn 13 $CUDNN_VERSION

  install_nvshmem 13 $NVSHMEM_VERSION

  CUDA_VERSION=13.0 bash install_nccl.sh

  CUDA_VERSION=13.0 bash install_cusparselt.sh $CUSPARSELT_VERSION

  ldconfig
}

function install_132 {
  CUDNN_VERSION=9.24.0.43
  CUSPARSELT_VERSION=0.8.1.1
  echo "Installing CUDA 13.2 and cuDNN ${CUDNN_VERSION} and NVSHMEM and NCCL and cuSparseLt-${CUSPARSELT_VERSION}"
  # install CUDA 13.2 in the same container
  install_cuda 13.2.1 cuda_13.2.1_595.58.03_linux

  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  install_cudnn 13 $CUDNN_VERSION

  install_nvshmem 13 $NVSHMEM_VERSION

  CUDA_VERSION=13.2 bash install_nccl.sh

  CUDA_VERSION=13.2 bash install_cusparselt.sh $CUSPARSELT_VERSION

  ldconfig
}

function install_134 {
  CUDNN_VERSION=9.24.0.43
  CUSPARSELT_VERSION=0.8.1.1
  echo "Installing CUDA 13.4 and cuDNN ${CUDNN_VERSION} and NVSHMEM and NCCL and cuSparseLt-${CUSPARSELT_VERSION}"
  # CUDA 13.4 ships no runfile-local installer yet, so install the toolkit from
  # the NVIDIA preview network repo (https://packages.nvidia.com).
  ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
  case "$ID" in
    ubuntu)
      codename=$(grep -oP '(?<=^VERSION_CODENAME=).+' /etc/os-release | tr -d '"')
      wget -q https://packages.nvidia.com/${codename}/nvidia-preview-keyring.deb
      dpkg -i nvidia-preview-keyring.deb
      apt-get update
      apt-get -y install cuda-toolkit-13-4
      rm -f nvidia-preview-keyring.deb
      ;;
    almalinux|rhel|centos)
      wget -q https://packages.nvidia.com/el8/nvidia-preview-keyring.rpm
      rpm -i nvidia-preview-keyring.rpm
      dnf clean all
      dnf -y install cuda-toolkit-13-4
      rm -f nvidia-preview-keyring.rpm
      ;;
    *) echo "install_134: unsupported OS '$ID'"; exit 1 ;;
  esac

  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  install_cudnn 13 $CUDNN_VERSION

  install_nvshmem 13 $NVSHMEM_VERSION

  CUDA_VERSION=13.4 bash install_nccl.sh

  CUDA_VERSION=13.4 bash install_cusparselt.sh $CUSPARSELT_VERSION

  ldconfig
}

# idiomatic parameter and option handling in sh
while test $# -gt 0
do
    case "$1" in
    12.4) install_124;
        ;;
    12.6|12.6.*) install_126;
        ;;
    12.8|12.8.*) install_128;
        ;;
    12.9|12.9.*) install_129;
        ;;
    13.0|13.0.*) install_130;
        ;;
    13.2|13.2.*) install_132;
        ;;
    13.4|13.4.*) install_134;
        ;;
    *) echo "bad argument $1"; exit 1
        ;;
    esac
    install_cupti_headers $CUDA_CUPTI_VERSION
    shift
done
