# Set up NVIDIA docker repo
curl -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
echo "deb https://nvidia.github.io/libnvidia-container/ubuntu14.04/amd64 /" | sudo tee -a /etc/apt/sources.list.d/nvidia-docker.list
echo "deb https://nvidia.github.io/nvidia-container-runtime/ubuntu14.04/amd64 /" | sudo tee -a /etc/apt/sources.list.d/nvidia-docker.list
echo "deb https://nvidia.github.io/nvidia-docker/ubuntu14.04/amd64 /" | sudo tee -a /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get -q -y update
sudo apt-get -q -y remove linux-image-generic linux-headers-generic linux-generic docker-ce
# WARNING: Docker version is hardcoded here; you must update the
# version number below for docker-ce and nvidia-docker2 to get newer
# versions of Docker.  We hardcode these numbers because we kept
# getting broken CI when Docker would update their docker version,
# and nvidia-docker2 would be out of date for a day until they
# released a newer version of their package.
sudo apt-get -q -y install \
  linux-headers-$(uname -r) \
  linux-image-generic \
  moreutils \
  docker-ce=18.06.2~ce~3-0~ubuntu \
  nvidia-docker2=2.0.3+docker18.06.2-1 \
  expect-dev
sudo pkill -SIGHUP dockerd
sudo pip -q install awscli==1.16.35
if [ -n "${USE_CUDA_DOCKER_RUNTIME}" ]; then
  wget 'https://s3.amazonaws.com/ossci-linux/nvidia_driver/NVIDIA-Linux-x86_64-410.79.run'
  sudo /bin/bash ./NVIDIA-Linux-x86_64-410.79.run -s --no-drm
  nvidia-smi
fi
