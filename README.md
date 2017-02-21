# Gloo
Gloo is a collective communications library. It comes with a number of
collective algorithms useful for machine learning applications. These
include a barrier, broadcast, and allreduce.

Transport of data between participating machines is abstracted so that
IP can be used at all times, or InifiniBand (or RoCE) when available.

Where applicable, algorithms have an implementation that works with
system memory buffers, and one that works with NVIDIA GPU memory
buffers. In the latter case, if the InfiniBand transport is used,
GPUDirect can be used to accelerate cross machine GPU-to-GPU memory
transfers.

## Requirements
Gloo is built to run on Linux and has no hard dependencies other than libc.

Optional dependencies are:
* cuda -- for CUDA algorithms, tests, and benchmark
* googletest -- to build and run tests
* eigen -- for fast floating point routines
* hiredis -- for coordinating machine rendezvous through Redis

## Usage
You can build Gloo using CMake.

Since it is a library, it is most convenient to vendor it in your own
project and include the project root in your own CMake configuration.

For standalone builds (e.g. to run tests or benchmarks), first
populate the `third-party` directory with a few dependencies to
compile both the tests and the benchmark tool:

``` shell
cd third-party
./fetch.sh
```

Then, to build:

``` shell
mkdir build
cd build
cmake ../ -DBUILD_TEST=1 -DBUILD_BENCHMARK=1
ls -l gloo/gloo_{test,benchmark}
```

## Documentation
Please refer to [docs/](docs/) for detailed documentation.

## License
Gloo is BSD-licensed. We also provide an additional patent grant.
