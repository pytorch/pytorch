# Lazy Tensors Core

1. Clone a copy of the PyTorch repo, switch to the `lazy_tensor_staging` branch and build it. Use `git submodule update --init --recursive` to fetch all submodules.
1. From the `lazy_tensor_core` subfolder, run `python setup.py develop`.
1. Run `example.py`. It'll report that no lazy tensor backend is registered.

Suggested build environment:

```bash
export CC=clang-8
export CXX=clang++-8
export BUILD_CPP_TESTS=0
```

If you want to debug it as well:

```bash
export DEBUG=1
```
