# External Backends Folder

This folder can be used to download external backends and seamlessly integrate them with the jit.

These backends will need to contain a CMakeLists.txt file that defines three variables:
- `TORCH_JIT_BACKEND_SRCS`: A list of C++ sources to be compiled directly with the `torch` binary.
- `TORCH_JIT_BACKEND_INCLUDE`: A list of include directories to be used during the compilation of `torch` sources.
- `TORCH_JIT_BACKEND_LIBS`: A list of libraries to be linked against the `torch` target.
