# Lazy Tensors Core

1. Clone a copy of the PyTorch repo, switch to the `lazy_tensor_staging` branch and use `git submodule update --init --recursive` to fetch all submodules.
1. From the PyTorch project root directory, run `python setup.py develop` to build PyTorch.
1. From the `lazy_tensor_core` subfolder, run `scripts/apply_patches.sh`.
1. From the `lazy_tensor_core` subfolder, run `python setup.py develop`.
1. Run `example.py`. It'll register and use the TorchScript backend.

To run on a CUDA GPU, run with `LTC_TS_CUDA` set, for example:

```bash
LTC_TS_CUDA=1 python example.py
```

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
