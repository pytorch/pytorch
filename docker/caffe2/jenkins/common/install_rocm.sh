set -ex

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y wget

DEB_ROCM_REPO=http://repo.radeon.com/rocm/apt/debian
wget -qO - $DEB_ROCM_REPO/rocm.gpg.key | apt-key add -

apt-get update
apt-get install -y \
  rocm-dev \
  rocm-libs \
  rocm-utils \
  rocfft \
  miopen-hip \
  miopengemm \
  rocblas \
  hipblas \
  rocrand \
  rocm-profiler \
  cxlactivitylogger

pushd $HOME
curl https://s3.amazonaws.com/ossci-linux/hcrng-master-a8c6a0b-Linux.deb -o hcrng.deb
dpkg -i hcrng.deb
rm hcrng.deb
curl https://s3.amazonaws.com/ossci-linux/hcsparse-master-907a505-Linux.deb -o hcsparse.deb
dpkg -i hcsparse.deb
rm hcsparse.deb
popd
