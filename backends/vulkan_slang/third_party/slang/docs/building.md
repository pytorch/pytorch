# Building Slang From Source

### TLDR

`cmake --workflow --preset release` to configure, build, and package a release
version of Slang.

## Prerequisites:

Please install:

- CMake (3.25 preferred, but 3.22 works[^1])
- A C++ compiler with support for C++17. GCC, Clang and MSVC are supported
- A CMake compatible backend, for example Visual Studio or Ninja
- Python3 (a dependency for building spirv-tools)

Optional dependencies for tests include

- CUDA
- OptiX
- NVAPI
- Aftermath
- X11

Other dependencies are sourced from submodules in the [./external](./external)
directory.

## Get the Source Code

Clone [this](https://github.com/shader-slang/slang) repository. Make sure to
fetch the submodules also.

```bash
git clone https://github.com/shader-slang/slang --recursive
```

## Configure and build

> This section assumes cmake 3.25 or greater, if you're on a lower version
> please see [building with an older cmake](#building-with-an-older-cmake)

For a Ninja based build system (all platforms) run:
```bash
cmake --preset default
cmake --build --preset releaseWithDebugInfo # or --preset debug, or --preset release
```

For Visual Studio run:
```bash
cmake --preset vs2022 # or 'vs2019' or `vs2022-dev`
start devenv ./build/slang.sln # to optionally open the project in Visual Studio
cmake --build --preset releaseWithDebugInfo # to build from the CLI, could also use --preset release or --preset debug
```

There also exists a `vs2022-dev` preset which turns on features to aid
debugging.

### WebAssembly build

In order to build WebAssembly build of Slang, Slang needs to be compiled with
[Emscripten SDK](https://github.com/emscripten-core/emsdk). You can find more
information about [Emscripten](https://emscripten.org/).

You need to clone the EMSDK repo. And you need to install and activate the latest.


```bash
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
```

For non-Windows platforms
```bash
./emsdk install latest
./emsdk activate latest
```

For Windows
```cmd
emsdk.bat install latest
emsdk.bat activate latest
```

After EMSDK is activated, Slang needs to be built in a cross compiling setup: 

- build the `generators` target for the build platform
- configure the build with `emcmake` for the host platform
- build for the host platform.

> Note: For more details on cross compiling please refer to the 
> [cross-compiling](docs/building.md#cross-compiling) section.

```bash
# Build generators.
cmake --workflow --preset generators --fresh
mkdir generators
cmake --install build --prefix generators --component generators

# Configure the build with emcmake.
# emcmake is available only when emsdk_env setup the environment correctly.
pushd ../emsdk
source ./emsdk_env # For Windows, emsdk_env.bat
popd
emcmake cmake -DSLANG_GENERATORS_PATH=generators/bin --preset emscripten -G "Ninja"

# Build slang-wasm.js and slang-wasm.wasm in build.em/Release/bin
cmake --build --preset emscripten --target slang-wasm
```

> Note: If the last build step fails, try running the command that `emcmake`
> outputs, directly.

## Installing

Build targets may be installed using cmake:

```bash
cmake --build . --target install
```

This should install `SlangConfig.cmake` that should allow `find_package` to work.
SlangConfig.cmake defines `SLANG_EXECUTABLE` variable that will point to `slangc`
executable and also define `slang::slang` target to be linked to.

For now, `slang::slang` is the only exported target defined in the config which can
be linked to.

Example usage

```cmake
find_package(slang REQUIRED PATHS ${your_cmake_install_prefix_path} NO_DEFAULT_PATH)
# slang_FOUND should be automatically set
target_link_libraries(yourLib PUBLIC
  slang::slang
)
```

## Testing

```bash
build/Debug/bin/slang-test
```

See the [documentation on testing](../tools/slang-test/README.md) for more information.

## More niche topics

### CMake options

| Option                            | Default                    | Description                                                                                  |
|-----------------------------------|----------------------------|----------------------------------------------------------------------------------------------|
| `SLANG_VERSION`                   | Latest `v*` tag            | The project version, detected using git if available                                         |
| `SLANG_EMBED_CORE_MODULE`         | `TRUE`                     | Build slang with an embedded version of the core module                                      |
| `SLANG_EMBED_CORE_MODULE_SOURCE`  | `TRUE`                     | Embed the core module source in the binary                                                   |
| `SLANG_ENABLE_DXIL`               | `TRUE`                     | Enable generating DXIL using DXC                                                             |
| `SLANG_ENABLE_ASAN`               | `FALSE`                    | Enable ASAN (address sanitizer)                                                              |
| `SLANG_ENABLE_FULL_IR_VALIDATION` | `FALSE`                    | Enable full IR validation (SLOW!)                                                            |
| `SLANG_ENABLE_IR_BREAK_ALLOC`     | `FALSE`                    | Enable IR BreakAlloc functionality for debugging.                                            |
| `SLANG_ENABLE_GFX`                | `TRUE`                     | Enable gfx targets                                                                           |
| `SLANG_ENABLE_SLANGD`             | `TRUE`                     | Enable language server target                                                                |
| `SLANG_ENABLE_SLANGC`             | `TRUE`                     | Enable standalone compiler target                                                            |
| `SLANG_ENABLE_SLANGI`             | `TRUE`                     | Enable Slang interpreter target                                                              |
| `SLANG_ENABLE_SLANGRT`            | `TRUE`                     | Enable runtime target                                                                        |
| `SLANG_ENABLE_SLANG_GLSLANG`      | `TRUE`                     | Enable glslang dependency and slang-glslang wrapper target                                   |
| `SLANG_ENABLE_TESTS`              | `TRUE`                     | Enable test targets, requires SLANG_ENABLE_GFX, SLANG_ENABLE_SLANGD and SLANG_ENABLE_SLANGRT |
| `SLANG_ENABLE_EXAMPLES`           | `TRUE`                     | Enable example targets, requires SLANG_ENABLE_GFX                                            |
| `SLANG_LIB_TYPE`                  | `SHARED`                   | How to build the slang library                                                               |
| `SLANG_ENABLE_RELEASE_DEBUG_INFO` | `TRUE`                     | Enable generating debug info for Release configs                                             |
| `SLANG_ENABLE_RELEASE_LTO`        | `TRUE`                     | Enable LTO for Release builds                                                                |
| `SLANG_ENABLE_SPLIT_DEBUG_INFO`   | `TRUE`                     | Enable generating split debug info for Debug and RelWithDebInfo configs                      |
| `SLANG_SLANG_LLVM_FLAVOR`         | `FETCH_BINARY_IF_POSSIBLE` | How to set up llvm support                                                                   |
| `SLANG_SLANG_LLVM_BINARY_URL`     | System dependent           | URL specifying the location of the slang-llvm prebuilt library                               |
| `SLANG_GENERATORS_PATH`           | ``                         | Path to an installed `all-generators` target for cross compilation                           |

The following options relate to optional dependencies for additional backends
and running additional tests. Left unchanged they are auto detected, however
they can be set to `OFF` to prevent their usage, or set to `ON` to make it an
error if they can't be found.

| Option                   | CMake hints                    | Notes                                                                                        |
|--------------------------|--------------------------------|----------------------------------------------------------------------------------------------|
| `SLANG_ENABLE_CUDA`      | `CUDAToolkit_ROOT` `CUDA_PATH` | Enable running tests with the CUDA backend, doesn't affect the targets Slang itself supports |
| `SLANG_ENABLE_OPTIX`     | `Optix_ROOT_DIR`               | Requires CUDA                                                                                |
| `SLANG_ENABLE_NVAPI`     | `NVAPI_ROOT_DIR`               | Only available for builds targeting Windows                                                  |
| `SLANG_ENABLE_AFTERMATH` | `Aftermath_ROOT_DIR`           | Enable Aftermath in GFX, and add aftermath crash example to project                          |
| `SLANG_ENABLE_XLIB`      |                                |                                                                                              |

### Advanced options

| Option                             | Default | Description                                                                                                                    |
|------------------------------------|---------|--------------------------------------------------------------------------------------------------------------------------------|
| `SLANG_ENABLE_DX_ON_VK`            | `FALSE` | Enable running the DX11 and DX12 tests on non-warning Windows platforms via vkd3d-proton, requires system-provided d3d headers |
| `SLANG_ENABLE_SLANG_RHI`           | `TRUE`  | Enable building and using [slang-rhi](https://github.com/shader-slang/slang-rhi) for tests                                     |
| `SLANG_USE_SYSTEM_MINIZ`           | `FALSE` | Build using system Miniz library instead of the bundled version in [./external](./external)                                    |
| `SLANG_USE_SYSTEM_LZ4`             | `FALSE` | Build using system LZ4 library instead of the bundled version in [./external](./external)                                      |
| `SLANG_USE_SYSTEM_VULKAN_HEADERS`  | `FALSE` | Build using system Vulkan headers instead of the bundled version in [./external](./external)                                   |
| `SLANG_USE_SYSTEM_SPIRV_HEADERS`   | `FALSE` | Build using system SPIR-V headers instead of the bundled version in [./external](./external)                                   |
| `SLANG_USE_SYSTEM_UNORDERED_DENSE` | `FALSE` | Build using system unordered dense instead of the bundled version in [./external](./external)                                  |
| `SLANG_SPIRV_HEADERS_INCLUDE_DIR`  | ``      | Use this specific path to SPIR-V headers instead of the bundled version in [./external](./external)                            |

### LLVM Support

There are several options for getting llvm-support:

- Use a prebuilt binary slang-llvm library:
  `-DSLANG_SLANG_LLVM_FLAVOR=FETCH_BINARY` or `-DSLANG_SLANG_LLVM_FLAVOR=FETCH_BINARY_IF_POSSIBLE` (this is the default)
    - You can set `SLANG_SLANG_LLVM_BINARY_URL` to point to a local
      `libslang-llvm.so/slang-llvm.dll` or set it to a URL of an zip/archive
      containing such a file
    - If this isn't set then the build system tries to download it from the
      release on github matching the current tag. If such a tag doesn't exist
      or doesn't have the correct os*arch combination then the latest release
      will be tried.
    - If `SLANG_SLANG_LLVM_BINARY_URL` is `FETCH_BINARY_IF_POSSIBLE` then in
      the case that a prebuilt binary can't be found then the build will proceed
      as though `DISABLE` was chosen
- Use a system supplied LLVM: `-DSLANG_SLANG_LLVM_FLAVOR=USE_SYSTEM_LLVM`, you
  must have llvm-13.0 and a matching libclang installed. It's important that
  either:
    - You don't end up linking to a dynamic libllvm.so, this will almost
      certainly cause multiple versions of LLVM to be loaded at runtime,
      leading to errors like `opt: CommandLine Error: Option
      'asm-macro-max-nesting-depth' registered more than once!`. Avoid this by
      compiling LLVM without the dynamic library.
    - Anything else which may be linked in (for example Mesa, also dynamically
      loads the same llvm object)
- Do not enable LLVM support: `-DSLANG_SLANG_LLVM_FLAVOR=DISABLE`

To build only a standalone slang-llvm, you can run:

```bash
cmake --workflow --preset slang-llvm
```

This will generate `build/dist-release/slang-slang-llvm.zip` containing the
library. This, of course, uses the system LLVM to build slang-llvm, otherwise
it would just be a convoluted way to download a prebuilt binary.

### Cross compiling

Slang generates some code at build time, using generators build from this
codebase. Due to this, for cross compilation one must already have built these
generators for the build platform. Build them with the `generators` preset, and
pass the install path to the cross building CMake invocation using
`SLANG_GENERATORS_PATH`

Non-Windows platforms:

```bash
# build the generators
cmake --workflow --preset generators --fresh
mkdir build-platform-generators
cmake --install build --config Release --prefix build-platform-generators --component generators
# reconfigure, pointing to these generators
# Here is also where you should set up any cross compiling environment
cmake \
  --preset default \
  --fresh \
  -DSLANG_GENERATORS_PATH=build-platform-generators/bin \
  -Dwhatever-other-necessary-options-for-your-cross-build \
  # for example \
  -DCMAKE_C_COMPILER=my-arch-gcc \
  -DCMAKE_CXX_COMPILER=my-arch-g++
# perform the final build
cmake --workflow --preset release
```

Windows

```bash
# build the generators
cmake --workflow --preset generators --fresh
mkdir build-platform-generators
cmake --install build --config Release --prefix build-platform-generators --component generators
# reconfigure, pointing to these generators
# Here is also where you should set up any cross compiling environment
# For example
./vcvarsamd64_arm64.bat
cmake \
  --preset default \
  --fresh \
  -DSLANG_GENERATORS_PATH=build-platform-generators/bin \
  -Dwhatever-other-necessary-options-for-your-cross-build
# perform the final build
cmake --workflow --preset release
```

### Example cross compiling with MSVC to windows-aarch64

One option is to build using the ninja generator, which requires providing the
native and cross environments via `vcvarsall.bat`

```bash
vcvarsall.bat
cmake --workflow --preset generators --fresh
mkdir generators
cmake --install build --prefix generators --component generators
vsvarsall.bat x64_arm64
cmake --preset default --fresh -DSLANG_GENERATORS_PATH=generators/bin
cmake --workflow --preset release
```

Another option is to build using the Visual Studio generator which can find
this automatically

```
cmake --preset vs2022 # or --preset vs2019
cmake --build --preset generators # to build from the CLI
cmake --install build --prefix generators --component generators
rm -rf build # The Visual Studio generator will complain if this is left over from a previous build
cmake --preset vs2022 --fresh -A arm64 -DSLANG_GENERATORS_PATH=generators/bin
cmake --build --preset release
```

### Nix

This repository contains a [Nix](https://nixos.org/)
[flake](https://wiki.nixos.org/wiki/Flakes) (not officially supported or
tested), which provides the necessary prerequisites for local development. Also,
if you use [direnv](https://direnv.net/), you can run the following commands to
have the Nix environment automatically activate when you enter your clone of
this repository:

```bash
echo 'use flake' >> .envrc
direnv allow
```

## Building with an older CMake

Because older CMake versions don't support all the features we want to use in
CMakePresets, you'll have to do without the presets. Something like the following

```bash
cmake -B build -G Ninja
cmake --build build -j
```

## Static linking against libslang

If linking against a static `libslang.a` you will need to link against some
dependencies also if you're not already incorporating them into your project.

You will need to link against:

```
${SLANG_DIR}/build/Release/lib/libslang.a
${SLANG_DIR}/build/Release/lib/libcompiler-core.a
${SLANG_DIR}/build/Release/lib/libcore.a
${SLANG_DIR}/build/external/miniz/libminiz.a
${SLANG_DIR}/build/external/lz4/build/cmake/liblz4.a
```

## Notes

[^1] below 3.25, CMake lacks the ability to mark directories as being
system directories (https://cmake.org/cmake/help/latest/prop_tgt/SYSTEM.html#prop_tgt:SYSTEM),
this leads to an inability to suppress warnings originating in the
dependencies in `./external`, so be prepared for some additional warnings.
