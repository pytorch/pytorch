# torch.utils.cpp_extension

```{eval-rst}
.. currentmodule:: torch.utils.cpp_extension

.. autofunction:: CppExtension

.. autofunction:: CUDAExtension

.. autofunction:: SyclExtension

.. autofunction:: BuildExtension

.. autofunction:: load

.. autofunction:: load_inline

.. autofunction:: include_paths

.. autofunction:: get_compiler_abi_compatibility_and_version

.. autofunction:: verify_ninja_availability

.. autofunction:: is_ninja_available
```

## CPU-specific header-only code paths

`CppExtension`, `CUDAExtension`, `load`, and `load_inline` set up the minimum
compiler and linker options needed to build against PyTorch. They do not
automatically copy the CPU capability flags that PyTorch uses internally when it
builds specialized ATen kernels, such as `CPU_CAPABILITY_AVX2` or
`CPU_CAPABILITY_AVX512`.

Some PyTorch header-only implementations use those macros to select optional
CPU-specific code paths. For example, `torch/headeronly/util/Half.h` uses
`CPU_CAPABILITY_AVX2` or `CPU_CAPABILITY_AVX512` to enable vectorized half
conversion helpers, although that particular path is disabled on macOS. A
custom C++ or CUDA operator that includes such headers without defining a CPU
capability macro will still build and run, but it will use the generic
implementation for those header-only functions.

If you intentionally compile an extension source file for a specific CPU
capability, pass both the PyTorch capability macro and the matching compiler ISA
flags to the compiler that builds that source file. For example:

```python
CppExtension(
    name="my_extension",
    sources=["extension.cpp"],
    extra_compile_args=[
        "-O3",
        "-mavx2",
        "-mfma",
        "-mf16c",
        "-DCPU_CAPABILITY=AVX2",
        "-DCPU_CAPABILITY_AVX2",
    ],
)
```

PyTorch's own AVX2 build also conditionally adds the GCC-specific tuning flags
`-mno-avx256-split-unaligned-load` and
`-mno-avx256-split-unaligned-store` when the compiler supports them. Extension
builds that want to mirror PyTorch's internal tuning can add those flags after
checking compiler support.

For `CUDAExtension` builds with separate C++ and CUDA sources, pass these
options under the `cxx` entry of `extra_compile_args` for C++ sources. If a
CUDA source also contains host CPU code that includes one of these headers, pass
the equivalent defines and host compiler ISA options through the `nvcc` entry:

```python
CUDAExtension(
    name="my_cuda_extension",
    sources=["extension.cpp", "extension_kernel.cu"],
    extra_compile_args={
        "cxx": [
            "-O3",
            "-mavx2",
            "-mfma",
            "-mf16c",
            "-DCPU_CAPABILITY=AVX2",
            "-DCPU_CAPABILITY_AVX2",
        ],
        "nvcc": [
            "-O3",
            "-DCPU_CAPABILITY=AVX2",
            "-DCPU_CAPABILITY_AVX2",
            "-Xcompiler",
            "-mavx2",
            "-Xcompiler",
            "-mfma",
            "-Xcompiler",
            "-mf16c",
        ],
    },
)
```

For JIT builds, pass the same options through `extra_cflags` for C++ sources:

```python
load(
    name="my_extension",
    sources=["extension.cpp"],
    extra_cflags=[
        "-O3",
        "-mavx2",
        "-mfma",
        "-mf16c",
        "-DCPU_CAPABILITY=AVX2",
        "-DCPU_CAPABILITY_AVX2",
    ],
)
```

If a JIT-compiled CUDA source also needs these options for host CPU code, pass
the equivalent options through `extra_cuda_cflags`.

On MSVC, use the corresponding `/arch:AVX2`, `/DCPU_CAPABILITY=AVX2`, and
`/DCPU_CAPABILITY_AVX2` options. For AVX512 builds, use
`CPU_CAPABILITY=AVX512`, `CPU_CAPABILITY_AVX512`, and the matching compiler
AVX512 flags for your compiler.

Only define these macros for translation units that are compiled with the
matching ISA flags and that will be run only on compatible CPUs, or behind your
own runtime dispatch. Defining an AVX2 or AVX512 capability for code that may
execute on unsupported CPUs can produce illegal instructions.
