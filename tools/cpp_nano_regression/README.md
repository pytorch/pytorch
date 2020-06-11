# Simple tracker of atomic instructions

Atomic instructions are expensive and it's useful to track their number to avoid regression.

This simple library provides such functionality with minimal intrusiveness to the main PyTorch build.

## Instrumentation

Basic approach is to compile target binary (e.g. libtorch.so) with hooks intercepting atomic accesses. This is done by preprocessor overrides of `__atomic_*` intrinsics used by gcc/clang. In particular `std::atomic` implementation also uses those intrinsics. Overrides are performed by passing `--include=prelude.h` in compiler options and thus forcing it to be included before any standard library headers in each compilation unit.

With pytorch build instrumentation can be enabled without any CMake changes:

```
CXXFLAGS=--include=$(realpath tools/cpp_nano_regression/prelude.h) python setup.py develop
```

## Tracker

By default instrumentation just attempts to call a weak symbol for every tracked event. This allows any binary to be runnable even without tracker installed. It comes handy in PyTorch build which produces many intermediate binaries (e.g. protoc and many shared libraries) and it's cumbersome to pass tracker's translation unit in the correct places.

The tracker implementation can be installed at runtime using `LD_PRELOAD`. `tracker.cpp` provides the simplest tracker implementation which is not thread-safe:

```
$ pushd tools/cpp_nano_regression
$ sh build.sh
$ popd
$ LD_PRELOAD=tools/cpp_nano_regression/tracker.so python -c 'import torch'
NANO TRACKER {
  __atomic_sub_fetch: 1232
  __atomic_compare_exchange_n: 13022
  __atomic_fetch_sub: 9343
  __atomic_add_fetch: 1232
  __atomic_fetch_add: 1239581
  __atomic_load_n: 58610
  __atomic_store_n: 6633
  __atomic_load: 183
  __atomic_store: 102
}
```

It's also possible to access tracker from python using `ctypes` (building a python extension is left as a future exercise):

```
$ LD_PRELOAD=tools/cpp_nano_regression/tracker.so python
>>> import torch
>>> import ctypes
>>> dll=ctypes.CDLL('./nano_tracker.so')
>>> _=dll.nano_tracking_reset()
>>> x=torch.tensor([1,2,3])
>>> _=dll.nano_tracking_dump()
NANO TRACKER {
  __atomic_sub_fetch: 1
  __atomic_fetch_add: 65
  __atomic_fetch_sub: 4
  __atomic_load_n: 8
  __atomic_add_fetch: 7
}
>>> _=dll.nano_tracking_reset()
>>> x=x+1
>>> _=dll.nano_tracking_dump()
NANO TRACKER {
  __atomic_sub_fetch: 15
  __atomic_compare_exchange_n: 1
  __atomic_fetch_sub: 3
  __atomic_load_n: 8
  __atomic_add_fetch: 15
  __atomic_fetch_add: 29
}
```

## Standalone demo

A simple example of how this works is available in this folder (independently of pytorch):

```
$ pushd tools/cpp_nano_regression
$ sh build.sh

$ ./test
before atomics
after atomics

$ LD_PRELOAD=./tracker.so ./test
before atomics
after atomics
NANO TRACKER {
  __atomic_fetch_sub: 1
  __atomic_fetch_add: 1
}

$ LD_PRELOAD=./printing_tracker.so ./test
before atomics
NANO TRACKER __atomic_fetch_add
NANO TRACKER __atomic_fetch_sub
after atomics
```
