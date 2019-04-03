# Set up NVIDIA docker repo
curl -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
echo "deb https://nvidia.github.io/libnvidia-container/ubuntu16.04/amd64 /" | sudo tee -a /etc/apt/sources.list.d/nvidia-docker.list
echo "deb https://nvidia.github.io/nvidia-container-runtime/ubuntu16.04/amd64 /" | sudo tee -a /etc/apt/sources.list.d/nvidia-docker.list
echo "deb https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64 /" | sudo tee -a /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get -y update
sudo apt-get -y remove linux-image-generic linux-headers-generic linux-generic docker-ce
# WARNING: Docker version is hardcoded here; you must update the
# version number below for docker-ce and nvidia-docker2 to get newer
# versions of Docker.  We hardcode these numbers because we kept
# getting broken CI when Docker would update their docker version,
# and nvidia-docker2 would be out of date for a day until they
# released a newer version of their package.
#
# How to figure out what the correct versions of these packages are?
# My preferred method is to start a Docker instance of the correct
# Ubuntu version (e.g., docker run -it ubuntu:16.04) and then ask
# apt what the packages you need are.  Note that the CircleCI image
# comes with Docker.
sudo apt-get -y install \
  linux-headers-$(uname -r) \
  linux-image-generic \
  moreutils \
  docker-ce=5:18.09.4~3-0~ubuntu-xenial \
  nvidia-docker2=2.0.3+docker18.09.4-1 \
  expect-dev

sudo pkill -SIGHUP dockerd

sudo pip -q install awscli==1.16.35

if [ -n "${USE_CUDA_DOCKER_RUNTIME}" ]; then
  DRIVER_FN="NVIDIA-Linux-x86_64-410.104.run"
  wget "https://s3.amazonaws.com/ossci-linux/nvidia_driver/$DRIVER_FN"
  sudo /bin/bash "$DRIVER_FN" -s --no-drm || (sudo cat /var/log/nvidia-installer.log && false)
  nvidia-smi
fi
