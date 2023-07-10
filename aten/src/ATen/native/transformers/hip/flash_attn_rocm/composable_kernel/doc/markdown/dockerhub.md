## CK docker hub

[Docker hub](https://hub.docker.com/r/rocm/composable_kernel)

## Why do I need this?

To make our lives easier and bring Composable Kernel dependencies together, we recommend using docker images.

## So what is Composable Kernel?

Composable Kernel (CK) library aims to provide a programming model for writing performance critical kernels for machine learning workloads across multiple architectures including GPUs, CPUs, etc, through general purpose kernel languages, like HIP C++.

To get the CK library

```
git clone https://github.com/ROCmSoftwarePlatform/composable_kernel.git
```

run a docker container 

```
docker run                                                            \
-it                                                                   \
--privileged                                                          \
--group-add sudo                                                      \
-w /root/workspace                                                    \
-v ${PATH_TO_LOCAL_WORKSPACE}:/root/workspace                         \
rocm/composable_kernel:ck_ub20.04_rocm5.3_release                     \
/bin/bash
```

and build the CK

```
mkdir build && cd build

# Need to specify target ID, example below is for gfx908 and gfx90a
cmake                                                                                             \
-D CMAKE_PREFIX_PATH=/opt/rocm                                                                    \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                                                         \
-D CMAKE_CXX_FLAGS="-O3"                                                                          \
-D CMAKE_BUILD_TYPE=Release                                                                       \
-D GPU_TARGETS="gfx908;gfx90a"                                                                    \
..
```

and 

```
make -j examples tests
```

To run all the test cases including tests and examples run

```
make test
```

We can also run specific examples or tests like

```
./bin/example_gemm_xdl_fp16
./bin/test_gemm_fp16
```

For more details visit [CK github repo](https://github.com/ROCmSoftwarePlatform/composable_kernel), [CK examples](https://github.com/ROCmSoftwarePlatform/composable_kernel/tree/develop/example), [even more CK examples](https://github.com/ROCmSoftwarePlatform/composable_kernel/tree/develop/client_example).

## And what is inside?

The docker images have everything you need for running CK including:

* [ROCm](https://www.amd.com/en/graphics/servers-solutions-rocm)
* [CMake](https://cmake.org/)
* [Compiler](https://github.com/RadeonOpenCompute/llvm-project)

## Which image is right for me?

Let's take a look at the image naming, for example "ck_ub20.04_rocm5.4_release". The image specs are:

* "ck" - made for running Composable Kernel
* "ub20.04" - based on Ubuntu 20.04
* "rocm5.4" - ROCm platform version 5.4
* "release" - compiler version is release

So just pick the right image for your project dependencies and you're all set.

## DIY starts here

If you need to customize a docker image or just can't stop tinkering, feel free to adjust the [Dockerfile](https://github.com/ROCmSoftwarePlatform/composable_kernel/blob/develop/Dockerfile) for your needs.

## License

CK is released under the MIT [license](https://github.com/ROCmSoftwarePlatform/composable_kernel/blob/develop/LICENSE).
