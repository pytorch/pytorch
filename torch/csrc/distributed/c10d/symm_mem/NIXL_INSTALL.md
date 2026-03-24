# NIXL Symmetric Memory Backend – Developer Installation Guide

This document describes how to install NIXL and build PyTorch with NIXL
symmetric memory support (`USE_NIXL=1`).

> **Future note:** NIXL will be pre-installed in the DLFW base container,
> removing the need for manual installation.  Until then, follow the steps
> below.

## Prerequisites

- CUDA toolkit (11.x or later)
- A working PyTorch development build environment
- For GPU-to-GPU transfers: UCX with CUDA support (see [UCX CUDA
  Support](#ucx-cuda-support) below)

## Quick Start

```bash
# 1. Install NIXL from pip (shared libraries + Python bindings)
pip install nixl          # meta-package; pulls nixl-cu12 by default

# 2. Clone the NIXL repo for C++ headers (pip wheel does not ship them)
git clone --depth 1 https://github.com/ai-dynamo/nixl.git /tmp/nixl-src

# 3. Create a unified prefix with headers, libs, and plugins
export NIXL_HOME=/tmp/nixl-prefix
mkdir -p $NIXL_HOME/include $NIXL_HOME/lib/plugins

cp /tmp/nixl-src/src/api/cpp/*.h $NIXL_HOME/include/

# Find where pip installed nixl-cu12
NIXL_SITE=$(pip show nixl-cu12 | grep Location | awk '{print $2}')

# Symlink all shared libraries (including transitive deps like etcd)
for d in "$NIXL_SITE/.nixl_cu12.mesonpy.libs" "$NIXL_SITE/nixl_cu12.libs"; do
  [ -d "$d" ] || continue
  for f in "$d"/*.so*; do
    ln -sf "$f" $NIXL_HOME/lib/$(basename "$f")
  done
done

# Symlink plugins (UCX, GDS, etc.)
for f in "$NIXL_SITE/.nixl_cu12.mesonpy.libs/plugins"/*.so*; do
  ln -sf "$f" $NIXL_HOME/lib/plugins/$(basename "$f")
done

# 4. Set plugin discovery path
export NIXL_PLUGIN_DIR=$NIXL_HOME/lib/plugins

# 5. Build PyTorch with NIXL enabled
cd /path/to/pytorch
export USE_NIXL=1
pip install -e . -v --no-build-isolation
```

During the CMake configure step you should see:

```
-- NIXL_HOME set to: '/tmp/nixl-prefix'
-- NIXL_LIB: '/tmp/nixl-prefix/lib/libnixl.so'
-- NIXL_INCLUDE_DIR: '/tmp/nixl-prefix/include'
-- NIXL found, building with NIXL support
```

## Environment Variables

| Variable | Purpose |
|---|---|
| `NIXL_HOME` | Prefix containing `include/nixl.h` and `lib/libnixl.so`. Used by CMake `find_path`/`find_library`. |
| `NIXL_PLUGIN_DIR` | Directory containing NIXL backend plugins (`libplugin_UCX.so`, etc.). Required at runtime. |
| `USE_NIXL` | Set to `1` or `ON` to enable NIXL in the PyTorch build. Default: OFF. |
| `TORCH_SYMMMEM` | Set to `NIXL` at runtime to select the NIXL symmetric memory backend. Or call `symm_mem.set_backend("NIXL")` from Python. |

## UCX CUDA Support

NIXL uses UCX for GPU-to-GPU RDMA transfers.  The pip-installed NIXL wheel
**bundles its own UCX** (`libucp-*.so`, `libuct-*.so`), but this bundled UCX
typically lacks CUDA memory support (`libuct_cuda.so`).

**Symptoms** when CUDA support is missing:

```
W: 8 NVIDIA GPU(s) were detected, but UCX CUDA support was not found!
E: VRAM memory is detected as host by UCX. UCX is likely not configured
   with CUDA support. VRAM registration cannot proceed.
```

**Solutions** (pick one):

1. **Point bundled UCX at system CUDA modules.**  If your system has HPC-X
   UCX at `/opt/hpcx/ucx` with `lib/ucx/libuct_cuda.so`:

   ```bash
   export UCX_MODULE_DIR=/opt/hpcx/ucx/lib/ucx
   ```

2. **Build NIXL from source** against system UCX built with
   `--with-cuda=/usr/local/cuda`.

3. **Wait for DLFW container** (see note above) which will ship NIXL with
   a properly-configured UCX.

You can verify UCX CUDA support with:

```bash
LD_LIBRARY_PATH=$NIXL_HOME/lib UCX_MODULE_DIR=/opt/hpcx/ucx/lib/ucx \
  ldd $NIXL_HOME/lib/libnixl.so | grep "not found"
# Should print nothing if all dependencies resolve.
```

## Running Tests

```bash
export LD_LIBRARY_PATH=$NIXL_HOME/lib:$LD_LIBRARY_PATH
export NIXL_PLUGIN_DIR=$NIXL_HOME/lib/plugins

# Single-process test (no GPU needed for availability check)
python test/distributed/test_nixl_symmetric_memory.py NixlSymmetricMemorySingleProcTest

# Multi-process tests (requires 2+ GPUs with P2P access)
python test/distributed/test_nixl_symmetric_memory.py NixlSymmetricMemoryTest
```

If NIXL was not compiled into PyTorch, the multi-process tests will skip
with "NIXL backend not available".  If UCX CUDA support is missing, the
tests may hang during rendezvous — ensure `UCX_MODULE_DIR` is set (see
above).

## Architecture Overview

```
                    Python
                      │
    symm_mem.set_backend("NIXL")
    t = symm_mem.empty(1024, device="cuda")
    hdl = symm_mem.rendezvous(t, group=WORLD)
                      │
              ┌───────┴───────┐
              │  torch_nixl.so │  (built when USE_NIXL=1)
              │                │
              │ NIXLSymmetric- │
              │ MemoryAllocator│
              │   alloc()      │─── cudaMalloc
              │   rendezvous() │─── nixlAgent::registerMem
              │                │    + getLocalMD / loadRemoteMD
              └───────┬───────┘    (metadata via PyTorch Store)
                      │
              ┌───────┴───────┐
              │   libnixl.so   │  (from pip / NIXL_HOME)
              │   nixlAgent    │
              │   UCX backend  │──── libplugin_UCX.so
              └───────────────┘         │
                                   libucp / libuct
                                   (+ libuct_cuda for VRAM)
```

## Device-Side API (Future)

NIXL provides device-side GPU functions in `nixl_device.cuh`:

- `nixlPut(src, dst, size)` – one-sided GPU-initiated memory transfer
- `nixlAtomicAdd(value, counter)` – atomic remote counter increment
- `nixlGetPtr(mvh, index)` – get device pointer to mapped remote memory
- `nixlGpuGetXferStatus(status)` – poll transfer completion

These require host-side `prepMemView()` to create memory view handles
before kernel launch.  Integration with PyTorch's `torch.ops.symm_mem`
namespace (analogous to `nccl_put` / `nccl_get`) is planned.
