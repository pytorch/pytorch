# PyTorch AMD Build

Scripts for building with AMD GPU (ROCm) support.

## Scripts

### build_amd.py

[build_amd.py](build_amd.py) is the top-level entry point for HIPifying our
codebase. This script runs in-place on the repository, switching source files
from CUDA APIs to HIP APIs.

Right now, PyTorch and Caffe2 share logic for how to do this transpilation, but
have separate entry-points for transpiling either PyTorch or Caffe2 code.

Usage:

```bash
python ./tools/amd_build/build_amd.py
```

### build_windows.sh

[build_windows.sh](build_windows.sh) is a utility script that helps set
environment variables for ROCm on Windows builds and then calls
[.ci/pytorch/win-build.sh](/.ci/pytorch/win-build.sh) to produce pytorch wheels.

### write_rocm_init.py

[write_rocm_init.py](write_rocm_init.py) is a utility for writing the
`torch/_rocm_init.py` bootstrap module used by [`torch/__init__.py`](/torch/__init__.py).

## Building PyTorch with ROCm support

Source builds in a few configurations are possible:

1. On Linux, PyTorch can be built against a system install of a stable ROCm
   release, typically installed to `/opt/rocm/`.

   _This build configuration has been supported for the longest time and is
   generally the most stable._
2. On Windows, PyTorch can be built against the
   [AMD HIP SDK for Windows](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html).

   _This build configuration has been tested by a few developers but has not
   been documented. YMMV_.
3. On Linux and Windows, PyTorch can be built against the preview ROCm Python
   packages produced by https://github.com/ROCm/TheRock, typically installed
   into a Python virtual environment.

   _This build configuration is under active development and is where AMD
   would like to focus efforts._

See also [Installation From Source - AMD ROCm Support](/README.md#amd-rocm-support).

### Common setup

No matter the build configuration, you will first need to install requirements
and HIPIFY the codebase:

```bash
pip install -r requirements.txt
python tools/amd_build/build_amd.py

# Optionally commit the HIPIFY changes to keep your history clean.
git checkout -b amd-build
git add -A
git commit -m "DO NOT SUBMIT - HIPIFY"
```

### (Linux) Building with system ROCm

See [Installation From Source - AMD ROCm Support](/README.md#amd-rocm-support).

### (Windows) Building with the HIP SDK

TODO: document (if this continues to be supported, we may focus on the
ROCm Python packages approach instead)

### (Linux and Windows) Building with ROCm Python packages

> [!WARNING]
> This method of building and installing is new and may still have feature gaps.

1. Install ROCm Python packages for your GPU (ideally into a venv) by following
   https://github.com/ROCm/TheRock/blob/main/RELEASES.md#installing-rocm-python-packages:

    ```bash
    # Make sure to use the index for your GPU family here
    # TODO: swap to production index URL once available
    pip install --index-url https://d2awnip2yjpvqn.cloudfront.net/v2/gfx110X-dgpu/ rocm[libraries,devel]
    ```

2. Write the `_rocm_init.py` file (see https://github.com/ROCm/TheRock/blob/main/docs/packaging/python_packaging.md for details):

    ```bash
    python tools/amd_build/write_rocm_init.py
    ```

3. Build (Windows)

    ```bash
    bash ./tools/amd_build/build_windows.sh
    ```
