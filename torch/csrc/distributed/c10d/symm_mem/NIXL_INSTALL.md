# Building PyTorch with NIXL Symmetric Memory Support

CUDA DL release 26.03 includes NIXL in inference-level containers. For PyTorch
source builds, use a Python-capable PyTorch/DLFW image with CUDA-aware UCX and
build NIXL from source against that UCX. The validation recipe below was tested
with `nvcr.io/nvidia/pytorch:26.03-py3` plus `/opt/hpcx/ucx`.

The NIXL device API path is separate: it requires `nixl_device.cuh` plus UCX GPU
device/GDAKI headers. Do not assume that a container with host-side NIXL support
also supports `USE_NIXL_DEVICE_API`.

## Persistent build locations

Large source builds can be cached on shared storage:

```bash
export NIXL_PT_ROOT=/lustre/fsw/network_software_cloudai/snordmann/nixl-pytorch
mkdir -p "$NIXL_PT_ROOT"/{src,build,install}
```

Use the paths below in place of `/tmp` if rebuild time matters.

## DLFW 26.03 runtime check

Inside the selected CUDA DL / PyTorch 26.03 container, verify that host-side
NIXL and CUDA-aware UCX are both visible:

```bash
python - <<'PY'
import importlib.util
print("nixl_cu12:", importlib.util.find_spec("nixl_cu12"))
PY
ldconfig -p | grep -E 'libnixl|libucp' || true
find /opt /usr /usr/local -path '*ucx*' -name libuct_cuda.so -print 2>/dev/null | head
find /opt /usr /usr/local -name libplugin_UCX.so -print 2>/dev/null | head
find /opt /usr /usr/local -name nixl_device.cuh -print 2>/dev/null | head
```

If NIXL is present but `libuct_cuda.so` or `libplugin_UCX.so` is not visible,
fix the container environment before running PyTorch tests.

## Build NIXL from source

The pip wheel bundles UCX without CUDA memory support, so build from source
against the system UCX which has `libuct_cuda.so`.

```bash
pip install meson ninja
git clone --depth 1 https://github.com/ai-dynamo/nixl.git "$NIXL_PT_ROOT/src/nixl"
meson setup "$NIXL_PT_ROOT/build/nixl" --prefix="$NIXL_PT_ROOT/install/nixl" \
    -Ducx_path=/opt/hpcx/ucx \
    -Denable_plugins=UCX \
    -Dbuild_tests=false -Dbuild_examples=false -Drust=false \
    "$NIXL_PT_ROOT/src/nixl"
ninja -C "$NIXL_PT_ROOT/build/nixl" -j$(nproc)
meson install -C "$NIXL_PT_ROOT/build/nixl"
```

Verify system UCX linkage (must show `/opt/hpcx/ucx`, **not** a bundled hash):
```bash
ldd "$NIXL_PT_ROOT/install/nixl/lib/x86_64-linux-gnu/plugins/libplugin_UCX.so" | grep ucp
# libucp.so.0 => /opt/hpcx/ucx/lib/libucp.so.0
```

On aarch64 images, replace `lib/x86_64-linux-gnu` with
`lib/aarch64-linux-gnu`.

## Build PyTorch

```bash
rm -f build/CMakeCache.txt   # if reconfiguring
export NIXL_HOME="$NIXL_PT_ROOT/install/nixl"
export USE_NIXL=1
export LD_LIBRARY_PATH=$NIXL_HOME/lib/x86_64-linux-gnu:/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH
pip install -e . -v --no-build-isolation
```

CMake should print:
```
-- NIXL found, building with NIXL support
```

## Run tests

```bash
export LD_LIBRARY_PATH=$NIXL_HOME/lib/x86_64-linux-gnu:/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH
export NIXL_PLUGIN_DIR=$NIXL_HOME/lib/x86_64-linux-gnu/plugins
python test/distributed/test_nixl_symmetric_memory.py -v
```

## Environment variables

| Variable | Purpose |
|---|---|
| `NIXL_HOME` | Install prefix with `include/` and `lib/`. Used by CMake. |
| `NIXL_PLUGIN_DIR` | Plugin directory containing `libplugin_UCX.so`. Required at runtime. |
| `USE_NIXL` | Set to `1` to enable NIXL in the PyTorch build (default: OFF). |
| `TORCH_SYMMMEM` | Set to `NIXL` at runtime, or call `symm_mem.set_backend("NIXL")`. |
