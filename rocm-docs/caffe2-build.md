# rocm-caffe2: Building From Source

## Intro
This instruction provides a starting point to build caffe2 on AMD GPUs (Caffe2 ROCm port) from source.
*Note*: it is recommended to start with a clean Ubuntu 16.04 system

## Install ROCm

Install ROCm stack following steps at [link](https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md) if your machine doesn't have ROCm already.

## Build caffe2 inside docker

 If your machine doesn't have docker installed, follow the steps [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce) to install docker.

### Pull the docker image
```
docker pull rohith612/caffe2:rocm1.8.2
```
This docker image has all the dependencies for caffe2 pre-installed.

### Pull the latest caffe2 source:
* Using https 
```
git clone --recursive https://github.com/pytorch/pytorch.git
```
* Using ssh
```
git clone --recursive git@github.com:pytorch/pytorch.git
```
Navigate to repo directory
```
cd pytorch
```

### Launch the docker container
```	
docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video -v $PWD:/pytorch rohith612/caffe2:rocm1.8.2
``` 
Navigate to pytorch directory `cd /pytorch` inside the container.

### Build caffe2 Project from Src

* Run the command  

	`.jenkins/caffe2/build.sh`

	
* Test the rocm-caffe2 Installation 
	Before running the tests, make sure that the required environment variables are set:
	``` 
	export LD_LIBRARY_PATH=/pytorch/build_caffe2/lib:$LD_LIBRARY_PATH
	```

	Run the binaries under `/pytorch/build_caffe2/bin`
