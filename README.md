**IMPORTANT NOTE**

**NCCL1 is no longer maintained/updated and has been replaced by NCCL2, available at**

**http://developer.nvidia.com/nccl.**

# NCCL

Optimized primitives for collective multi-GPU communication.

## Introduction

NCCL (pronounced "Nickel") is a stand-alone library of standard collective communication routines, such as all-gather, reduce, broadcast, etc., that have been optimized to achieve high bandwidth over PCIe. NCCL supports an arbitrary number of GPUs installed in a single node and can be used in either single- or multi-process (e.g., MPI) applications.
[This blog post](https://devblogs.nvidia.com/parallelforall/fast-multi-gpu-collectives-nccl/) provides details on NCCL functionality, goals, and performance.

## What's inside

At present, the library implements the following collectives:
- all-reduce
- all-gather
- reduce-scatter
- reduce
- broadcast

These collectives are implemented using ring algorithms and have been optimized primarily for throughput. For best performance, small collectives should be batched into larger operations whenever possible. Small test binaries demonstrating how to use each of the above collectives are also provided.

## Requirements

NCCL requires at least CUDA 7.0 and Kepler or newer GPUs. Best performance is achieved when all GPUs are located on a common PCIe root complex, but multi-socket configurations are also supported.

Note: NCCL may also work with CUDA 6.5, but this is an untested configuration.

## Build & run

To build the library and tests.

```shell
$ cd nccl
$ make CUDA_HOME=<cuda install path> test
```

Test binaries are located in the subdirectories nccl/build/test/{single,mpi}.

```shell
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib
$ ./build/test/single/all_reduce_test
Error: must specify at least data size in bytes!

Tests nccl AllReduce with user supplied arguments.
    Usage: all_reduce_test <data size in bytes> [number of GPUs] [GPU 0] [GPU 1] ...

$ ./build/test/single/all_reduce_test 10000000
# Using devices
#   Device  0 ->  0 [0x0a] GeForce GTX TITAN X
#   Device  1 ->  1 [0x09] GeForce GTX TITAN X
#   Device  2 ->  2 [0x06] GeForce GTX TITAN X
#   Device  3 ->  3 [0x05] GeForce GTX TITAN X

#                                                 out-of-place                    in-place
#      bytes             N    type      op     time  algbw  busbw      res     time  algbw  busbw      res
    10000000      10000000    char     sum    1.628   6.14   9.21    0e+00    1.932   5.18   7.77    0e+00
    10000000      10000000    char    prod    1.629   6.14   9.21    0e+00    1.643   6.09   9.13    0e+00
    10000000      10000000    char     max    1.621   6.17   9.25    0e+00    1.634   6.12   9.18    0e+00
    10000000      10000000    char     min    1.633   6.12   9.19    0e+00    1.637   6.11   9.17    0e+00
    10000000       2500000     int     sum    1.611   6.21   9.31    0e+00    1.626   6.15   9.23    0e+00
    10000000       2500000     int    prod    1.613   6.20   9.30    0e+00    1.629   6.14   9.21    0e+00
    10000000       2500000     int     max    1.619   6.18   9.26    0e+00    1.627   6.15   9.22    0e+00
    10000000       2500000     int     min    1.619   6.18   9.27    0e+00    1.624   6.16   9.24    0e+00
    10000000       5000000    half     sum    1.617   6.18   9.28    4e-03    1.636   6.11   9.17    4e-03
    10000000       5000000    half    prod    1.618   6.18   9.27    1e-03    1.657   6.03   9.05    1e-03
    10000000       5000000    half     max    1.608   6.22   9.33    0e+00    1.621   6.17   9.25    0e+00
    10000000       5000000    half     min    1.610   6.21   9.32    0e+00    1.627   6.15   9.22    0e+00
    10000000       2500000   float     sum    1.618   6.18   9.27    5e-07    1.622   6.17   9.25    5e-07
    10000000       2500000   float    prod    1.614   6.20   9.29    1e-07    1.628   6.14   9.21    1e-07
    10000000       2500000   float     max    1.616   6.19   9.28    0e+00    1.633   6.12   9.19    0e+00
    10000000       2500000   float     min    1.613   6.20   9.30    0e+00    1.628   6.14   9.21    0e+00
    10000000       1250000  double     sum    1.629   6.14   9.21    0e+00    1.628   6.14   9.21    0e+00
    10000000       1250000  double    prod    1.619   6.18   9.26    2e-16    1.628   6.14   9.21    2e-16
    10000000       1250000  double     max    1.613   6.20   9.30    0e+00    1.630   6.13   9.20    0e+00
    10000000       1250000  double     min    1.622   6.16   9.25    0e+00    1.623   6.16   9.24    0e+00
```

To install, run `make PREFIX=<install dir> install` and add `<instal dir>/lib` to your `LD_LIBRARY_PATH`.

## Usage

NCCL follows the MPI collectives API fairly closely. Before any collectives can be called, a communicator object must be initialized on each GPU. On a single-process machine, all GPUs can be conveniently initialized using `ncclCommInitAll`. For multi-process applications (e.g., with MPI), `ncclCommInitRank` must be called for each GPU. Internally `ncclCommInitRank` invokes a synchronization among all GPUs, so these calls must be invoked in different host threads (or processes) for each GPU. A brief single-process example follows, for an MPI example see test/mpi/mpi_test.cu. For details about the API see nccl.h.

```c
#include <nccl.h>

typedef struct {
  double* sendBuff;
  double* recvBuff;
  int size;
  cudaStream_t stream;
} PerThreadData;

int main(int argc, char* argv[])
{
  int nGPUs;
  cudaGetDeviceCount(&nGPUs);
  ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t)*nGPUs);
  ncclCommInitAll(comms, nGPUs); // initialize communicator
                                // One communicator per process

  PerThreadData* data;

  ... // Allocate data and issue work to each GPU's
      // perDevStream to populate the sendBuffs.

  for(int i=0; i<nGPUs; ++i) {
    cudaSetDevice(i); // Correct device must be set
                      // prior to each collective call.
    ncclAllReduce(data[i].sendBuff, data[i].recvBuff, size,
        ncclDouble, ncclSum, comms[i], data[i].stream);
  }

  ... // Issue work into data[*].stream to consume buffers, etc.
}
```

## Copyright and License

NCCL is provided under the [BSD licence](LICENSE.txt). All source code and
accompanying documentation is copyright (c) 2015-2016, NVIDIA CORPORATION. All
rights reserved.

