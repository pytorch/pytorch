# Caffe2: Building From Source on ROCm Platform

## Intro
This instruction provides a starting point to build caffe2 on AMD GPUs (Caffe2 ROCm port) from source.
*Note*: it is recommended to start with a clean Ubuntu 16.04 system

## Install docker

 If your machine doesn't have docker installed, follow the steps [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce) to install docker.

## Install ROCm

Install ROCm stack following steps at [link](https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md) if your machine doesn't have ROCm already.

Once the machine is ready with ROCm stack, there are two ways to use caffe2 
* Run the docker container with caffe2 installed in it.

* Build caffe2 from source inside a docker with all the dependencies.

## Launch docker container with caffe2 pre-installed
```
docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video rocm/caffe2:rocm1.8.2-v2
```

To run benchmarks, skip directly to benchmarks section of the document.

## Build Caffe2 from source
### Pull the docker image
```
docker pull rocm/caffe2:unbuilt-rocm1.8.2-v2
```
This docker image has all the dependencies for caffe2 pre-installed.

### Pull the latest caffe2 source:

Checkout the `caffe2_specific` branch of this repository.
* Using https 
```
git clone --recurse-submodules -b caffe2_specific https://github.com/ROCmSoftwarePlatform/pytorch.git
```
* Using ssh
```
git clone --recurse-submodules -b caffe2_specific git@github.com:ROCmSoftwarePlatform/pytorch.git
```
Navigate to repo directory
```
cd pytorch
```

### Launch the docker container
```	
docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video -v $PWD:/pytorch rocm/caffe2:unbuilt-rocm1.8.2-v2
``` 
Navigate to pytorch directory `cd /pytorch` inside the container.

### Build caffe2 Project from source

* Run the command  

	`.jenkins/caffe2/amd/build_amd.sh`

	
* Test the rocm-caffe2 Installation 

	Before running the tests, make sure that the required environment variables are set:
	```
	cd build_caffe2 && python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
	```
Make sure the the following environment variables are set.

```
LD_lIBRARY_PATH=/usr/local/caffe2/lib
PYTHONPATH=/usr/local/caffe2/lib/python2.7/dist-packages
```

## Run benchmarks

Navigate to build directory, `cd /pytorch/build_caffe2` to run benchmarks.

Caffe2 benchmarking script supports the following networks.
1. MLP
2. AlexNet
3. OverFeat
4. VGGA
5. Inception
6. Inception_v2
7. Resnet50 

*Special case:* Inception_v2 and Resnet50 will need their corresponding protobuf files to run the benchmarks. Protobufs can be downloaded from caffe2 model zoo using the below command. Substitute model_name with `inception_v2` or `resnet50`

```
python caffe2/python/models/download.py <model_name>
```
This will download the protobufs to current working directory.

To run benchmarks for networks MLP, AlexNet, OverFeat, VGGA, Inception, run the command replacing `<name_of_the_netwrok>` with one of the networks. 

```
python caffe2/python/convnet_benchmarks.py --batch_size 64 --model <name_of_the_network> --engine MIOPEN --layer_wise_benchmark True --net_type simple

```
To run Inception_v2 or Resnet50, please add additional argument `--model_path` to the above command which should point to the model directories downloaded above.

```
python caffe2/python/convnet_benchmarks.py --batch_size 64 --model <name_of_the_network> --engine MIOPEN --layer_wise_benchmark True --net_type simple --model_path <path_to_model_protobufs>

```

