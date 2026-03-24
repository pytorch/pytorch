# Building PyTorch with NIXL Symmetric Memory Support

> NIXL will be pre-installed in the DLFW base container in the near future.

## Build NIXL from source

The pip wheel bundles UCX without CUDA memory support, so build from source
against the system UCX which has `libuct_cuda.so`.

```bash
pip install meson ninja
git clone --depth 1 https://github.com/ai-dynamo/nixl.git /tmp/nixl-src
meson setup /tmp/nixl-build --prefix=/tmp/nixl-install \
    -Ducx_path=/opt/hpcx/ucx \
    -Denable_plugins=UCX \
    -Dbuild_tests=false -Dbuild_examples=false -Drust=false \
    /tmp/nixl-src
ninja -C /tmp/nixl-build -j$(nproc)
meson install -C /tmp/nixl-build
```

Verify system UCX linkage (must show `/opt/hpcx/ucx`, **not** a bundled hash):
```bash
ldd /tmp/nixl-install/lib/x86_64-linux-gnu/plugins/libplugin_UCX.so | grep ucp
# libucp.so.0 => /opt/hpcx/ucx/lib/libucp.so.0
```

## Build PyTorch

```bash
rm -f build/CMakeCache.txt   # if reconfiguring
export NIXL_HOME=/tmp/nixl-install
export USE_NIXL=1
export LD_LIBRARY_PATH=$NIXL_HOME/lib/x86_64-linux-gnu:/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH
pip install -e . -v --no-build-isolation   # or your build script
```

CMake should print:
```
-- NIXL found, building with NIXL support
```

## Run tests

```bash
export LD_LIBRARY_PATH=/tmp/nixl-install/lib/x86_64-linux-gnu:/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH
export NIXL_PLUGIN_DIR=/tmp/nixl-install/lib/x86_64-linux-gnu/plugins
python test/distributed/test_nixl_symmetric_memory.py -v
```

## Environment variables

| Variable | Purpose |
|---|---|
| `NIXL_HOME` | Install prefix with `include/` and `lib/`. Used by CMake. |
| `NIXL_PLUGIN_DIR` | Plugin directory containing `libplugin_UCX.so`. Required at runtime. |
| `USE_NIXL` | Set to `1` to enable NIXL in the PyTorch build (default: OFF). |
| `TORCH_SYMMMEM` | Set to `NIXL` at runtime, or call `symm_mem.set_backend("NIXL")`. |
