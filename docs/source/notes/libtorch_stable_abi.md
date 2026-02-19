# LibTorch Stable ABI

## Overview

The LibTorch Stable ABI (Application Binary Interface) provides a limited interface for extending PyTorch functionality without being tightly coupled to specific PyTorch versions. This enables the development of custom operators and extensions that remain compatible across PyTorch releases. This limited set of APIs is not intended to replace existing LibTorch, but rather to provide a stable foundation for a majority of custom extension use cases. If there is any API you would like to see added to the stable ABI, please file a request through a [new issue on the PyTorch repo](https://github.com/pytorch/pytorch/issues).

The limited stable ABI consists of three main components:

1. **Stable C headers** - Low-level C API implemented by libtorch (primarily `torch/csrc/inductor/aoti_torch/c/shim.h`)
2. **Header-only C++ library** - Standalone utilities implemented in only headers such that there is no dependence on libtorch (`torch/headeronly/*`)
3. **Stable C++ wrappers** - High-level C++ convenience wrappers (`torch/csrc/stable/*`)

We discuss each of these in detail

### `torch/headeronly`

The inlined C++ headers living in [`torch/headeronly`](https://github.com/pytorch/pytorch/tree/main/torch/headeronly) are completely decoupled from LibTorch. The headers consist of certain utilities that might be familiar to custom extension writers. For example, the
`c10::ScalarType` enum lives here as `torch::headeronly::ScalarType`, as well as a libtorch-independent version of `TORCH_CHECK` that is `STD_TORCH_CHECK`. You can trust all APIs in the `torch::headeronly` namespace to not depend on `libtorch.so`. These APIs are also globally listed in [torch/header_only_apis.txt](https://github.com/pytorch/pytorch/blob/main/torch/header_only_apis.txt).

### `torch/csrc/stable`

This is a set of inlined C++ headers that provide wrappers around the C API that handle the rough edges
discussed below.

It consists of

- torch/csrc/stable/library.h: Provides a stable version of TORCH_LIBRARY and similar macros.
- torch/csrc/stable/tensor_struct.h: Provides torch::stable::Tensor, a stable version of at::Tensor.
- torch/csrc/stable/ops.h: Provides a stable interface for calling ATen ops from `native_functions.yaml`.
- torch/csrc/stable/accelerator.h: Provides a stable interface for device-generic objects and APIs
(e.g. `getCurrentStream`, `DeviceGuard`).

We are continuing to improve coverage in our `torch/csrc/stable` APIs. Please file an issue if you'd like to see support for particular APIs in your custom extension.

For complete API documentation of the stable operators, see the [Torch Stable API cpp documentation](https://docs.pytorch.org/cppdocs/stable.html). <!-- @lint-ignore: URL won't exist till stable.rst cpp docs are published in 2.10 -->

### Stable C headers

The stable C headers started by AOTInductor form the foundation of the stable ABI. Presently, the available C headers include:

- [torch/csrc/inductor/aoti_torch/c/shim.h](https://github.com/pytorch/pytorch/blob/main/torch/csrc/inductor/aoti_torch/c/shim.h): Includes C-style shim APIs for commonly used regarding Tensors, dtypes, CUDA, and the like.
- [torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h](https://github.com/pytorch/pytorch/blob/main/torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h): Includes C-style shim APIs for ATen ops from `native_functions.yaml` (e.g. `aoti_torch_aten_new_empty`).
- [torch/csrc/inductor/aoti_torch/generated/c_shim_*.h](https://github.com/pytorch/pytorch/blob/main/torch/csrc/inductor/aoti_torch/generated): Includes C-style shim APIs for specific backend kernels dispatched from `native_functions.yaml` (e.g. `aoti_torch_cuda_pad`). These APIs should only be used for the specific backend they are named after (e.g. `aoti_torch_cuda_pad` should only be used within CUDA kernels), as they opt out of the dispatcher.
- [torch/csrc/stable/c/shim.h](https://github.com/pytorch/pytorch/blob/main/torch/csrc/stable/c/shim.h): We are building out more ABIs to logically live in `torch/csrc/stable/c` instead of continuing the AOTI naming that no longer makes sense for our general use case.

These headers are promised to be ABI stable across releases and adhere to a stronger backwards compatibility policy than LibTorch. Specifically, we promise not to modify them for at least 2 years after they are released. However, this is **use at your own risk**. For example, users must handle the memory lifecycle of objects returned by certain APIs. Further, the stack-based APIs discussed below which allow the user to call into the PyTorch dispatcher do not provide strong guarantees on forward and backward compatibility of the underlying op that is called.

Unless absolutely necessary, we recommend the high-level C++ API in `torch/csrc/stable`
which will handle all the rough edges of the C API for the user.

## Migrating your kernel to the LibTorch stable ABI

If you'd like your kernel to be ABI stable with LibTorch, meaning you'd the ability to build for one version and run on another, your kernel must only use the limited stable ABI. This following section goes through some steps of migrating an existing kernel and APIs we imagine you would need to swap over.

Firstly, instead of registering kernels through `TORCH_LIBRARY`, LibTorch ABI stable kernels must be registered via `STABLE_TORCH_LIBRARY`. Note that implementations registered via `STABLE_TORCH_LIBRARY` must be boxed unlike `TORCH_LIBRARY`. The `TORCH_BOX` macro handles this automatically for most use cases. See the simple example below or our docs on [Stack-based APIs](stack-based-apis) for more details. For kernels that are registered via `pybind`, before using the stable ABI, it would be useful to migrate to register them via `TORCH_LIBRARY`.

While previously your kernels might have included APIs from `<torch/*.h>` (for example, `<torch/all.h>`), they are now limited to including from the 3 categories of headers mentioned above (`torch/csrc/stable/*.h`, `torch/headeronly/*.h` and the stable C headers). This means that your extension should no longer use any utilities from the `at::` or `c10::` namespaces but instead use their replacements in `torch::stable` and `torch::headeronly`. To provide a couple examples of the necessary migrations:
- all uses of `at::Tensor` must be replaced with `torch::stable::Tensor`
- all uses of `TORCH_CHECK` must be replaced with `STD_TORCH_CHECK`
- all uses of `at::kCUDA` must be replaced with `torch::headeronly::kCUDA` etc.
- native functions such as `at::pad` must be replaced with `torch::stable::pad`
- native functions that are called as Tensor methods (e.g., `Tensor.pad`) must be replaced with the ATen variant through `torch::stable::pad`.

As mentioned above, the LibTorch stable ABI is still under development. If there is any API or feature you would like to see added to the stable ABI/`torch::headeronly`/`torch::stable`, please file a request through a [new issue on the PyTorch repo](https://github.com/pytorch/pytorch/issues).

Below is a simple example of migrating an existing kernel that uses `TORCH_LIBRARY` to the stable ABI (`TORCH_STABLE_LIBRARY`). For a larger end to end example you can take a look at the FA3 repository. Specifically the diff between [`flash_api.cpp`](https://github.com/Dao-AILab/flash-attention/blob/ad70a007e6287d4f7e766f94bcf2f9a813f20f6b/hopper/flash_api.cpp#L1) and the stable variant [`flash_api_stable.cpp`](https://github.com/Dao-AILab/flash-attention/blob/ad70a007e6287d4f7e766f94bcf2f9a813f20f6b/hopper/flash_api_stable.cpp#L1).


### Original Version with `TORCH_LIBRARY`

```cpp
// original_kernel.cpp - Using TORCH_LIBRARY (not stable ABI)
#include <torch/torch.h>
#include <ATen/ATen.h>

namespace myops {

// Simple kernel that adds a scalar value to each element of a tensor
at::Tensor add_scalar(const at::Tensor& input, double scalar) {
  TORCH_CHECK(input.scalar_type() == at::kFloat, "Input must be float32");

  return input.add(scalar);
}

// Register the operator
TORCH_LIBRARY(myops, m) {
  m.def("add_scalar(Tensor input, float scalar) -> Tensor");
}

// Register the implementation
TORCH_LIBRARY_IMPL(myops, CompositeExplicitAutograd, m) {
  m.impl("add_scalar", &add_scalar);
}

} // namespace myops
```

### Migrated Version with `STABLE_TORCH_LIBRARY`

```cpp
// stable_kernel.cpp - Using STABLE_TORCH_LIBRARY (stable ABI)

// (1) Don't include <torch/torch.h> <ATen/ATen.h>
//     only include APIs from torch/csrc/stable, torch/headeronly and C-shims
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

namespace myops {

// Simple kernel that adds a scalar value to each element of a tensor
torch::stable::Tensor add_scalar(const torch::stable::Tensor& input, double scalar) {
  // (2) use STD_TORCH_CHECK instead of TORCH_CHECK
  STD_TORCH_CHECK(
      // (3) use torch::headeronly::kFloat instead of at:kFloat
      input.scalar_type() == torch::headeronly::kFloat,
      "Input must be float32");

  // (4) Use stable ops namespace instead of input.add
  return torch::stable::add(input, scalar);
}

// (5) Register the operator using STABLE_TORCH_LIBRARY
STABLE_TORCH_LIBRARY(myops, m) {
  m.def("add_scalar(Tensor input, float scalar) -> Tensor");
}

// (6) Register the implementation using STABLE_TORCH_LIBRARY_IMPL
//     Use TORCH_BOX to automatically handle boxing/unboxing
STABLE_TORCH_LIBRARY_IMPL(myops, CompositeExplicitAutograd, m) {
  m.impl("add_scalar", TORCH_BOX(&add_scalar));
}

} // namespace myops
```


## How are objects passed across the ABI boundary when interacting with the dispatcher?

When interacting with the dispatcher via the stable APIs (``STABLE_TORCH_LIBRARY`` etc.) we use a boxed convention. Arguments and returns are represented as a stack of ``StableIValue`` which correlates with a `torch::jit::stack` of IValues. We discuss the following below
1. StableIValue Conversions
2. StableIValue stack Conventions
3. Stable APIs that interact with the dispatcher

### StableIValue Conversions

We provide utilities for users to convert objects to and from StableIValues with the synonymous
`to` and `from` APIs in `torch/csrc/stable/stableivalue_conversions.h`. We document the stable custom extension representation, libtorch representation and StableIValue
representations below. Our confidently supported types are the ones in the table that have completed
rows. You can rely on this subset for proper ABI stability, meaning that you can call `to<T_custom_ext>(arg/ret)` or `from(T)` on these types.

For a limited set of use cases, we also implicitly support any literal type that is representable within 64 bits as StableIValues, as the default reinterpret_cast will succeed. (For example: c10::Device.) These types are currently ABI-stable on best effort but might break in the future and thus should be used for short term testing only.

You can always work with StableIValue abstractions in your custom kernel for types such as c10::Device even if there is no standard defined representation of device in custom extensions by not introspecting into the StableIValue. For example, a custom operator can take as argument a StableIValue device and directly pass it through to an aten operator with `aoti_torch_call_dispatcher`.


1. type in custom extension: type used within the end user custom library.
2. StableIValue representation: a stable conversion of the type to liaison between the user model vs libtorch.so in an ABI-stable manner.
3. type in libtorch: type used within libtorch.so (or any code binary locked with libtorch).
4. Schema Type: type as described by the schema, which we hail as the source of truth for both ATen ops in native_functions.yaml and for user defined custom operators registered to the dispatcher via TORCH_LIBRARY or torch.library.

|  type in custom extension    |   StableIValue representation   |   type in libtorch  |   Schema Type  |
| -------- | ------- | ------- | ------- |
| std::optional\<S> | if there is a value, raw bitwise copy into leading bytes of uint64_t of pointer to a new StableIValue representing S. if there is no value, nullptr. | std::optional\<T> | Type? |
| torch::stable::Tensor | raw bitwise copy of underlying AtenTensorHandle into leading bytes of uint64_t | at::Tensor |  Tensor |
| torch::headeronly::ScalarType | raw bitwise copy of the translated underlying enum into leading bytes of uint64_t | torch::headeronly::ScalarType | ScalarType |
| torch::headeronly::Layout | raw bitwise copy of the translated underlying enum into leading bytes of uint64_t | at::Layout | Layout |
| torch::headeronly::MemoryFormat | raw bitwise copy of the translated underlying enum into leading bytes of uint64_t | at::MemoryFormat | MemoryFormat |
| bool | raw bitwise copy into leading bytes of uint64_t | bool | bool |
| int64_t | raw bitwise copy into leading bytes of uint64_t | int64_t | int |
| double | raw bitwise copy into leading bytes of uint64_t | double | float |
| torch::stable::Device | raw bitwise copy of index and type into leading bytes of uint64_t | c10::Device | Device |
| ? | ? | c10::Stream | Stream |
| ? | ? | c10::complex<double> | complex |
| ? | ? | at::Scalar | Scalar |
| std::string/std::string_view | raw bitwise copy of underlying StringHandle into leading bytes of uint64_t | std::string/const char*/ivalue::ConstantString | str |
| ? | ? | at::Storage | Storage |
| ? | ? | at::Generator | Generator |
| std::vector<T>/torch::headeronly::HeaderOnlyArrayRef<T> | raw bitwise copy into leading bytes of uint64_t of pointer to a new StableIValue pointing to a list of StableIValues recursively representing the underlying elements. | c10::List\<T> | Type[] |
| ? | ? | ivalue::Tuple\<T> | (Type, ...) |
| ? | ? | c10::SymInt | SymInt |
| ? | ? | c10::SymFloat | SymFloat |
| ? | ? | c10::SymBool | SymBool |
| ? | ? | at::QScheme | QScheme |


### Stack Conventions

There are two invariants for the stack:

1. The stack is populated left to right.
    a. For example, a stack representing arguments `arg0`, `arg1`, and `arg2` will have `arg0` at index 0, `arg1` at index 1, and `arg2` at index 2.
    b. Returns are also populated left to right, e.g., `ret0` will be at index 0 and `ret1` will be at index 1, and so on.

2. The stack always has ownership of the objects it holds.
    a. When calling a stack-based API, you must give owning references to the calling stack and steal references from the returned stack.
    b. When registering your function to be called with a stack, you must steal references from your argument stack and push onto the stack new references.

(stack-based-apis)=
### Stack-based APIs

The above is relevant in two places:

1. `STABLE_TORCH_LIBRARY`
    Unlike `TORCH_LIBRARY`, the dispatcher expects kernels registered via `STABLE_TORCH_LIBRARY` to be boxed. The `TORCH_BOX` macro automatically handles this boxing for you:

    ```cpp
    Tensor my_amax_vec(Tensor t) {
        std::vector<int64_t> v = {0,1};
        return amax(t, v, false);
    }

    // Use TORCH_BOX to automatically generate the boxed wrapper
    STABLE_TORCH_LIBRARY(myops, m) {
        m.def("my_amax_vec(Tensor t) -> Tensor", TORCH_BOX(&my_amax_vec));
    }
    ```

2. `torch_call_dispatcher`
    This API allows you to call the PyTorch dispatcher from C/C++ code. It has the following signature:

    ```cpp
    torch_call_dispatcher(const char* opName, const char* overloadName, StableIValue* stack, uint64_t extension_build_version);
    ```

    `torch_call_dispatcher` will call the op overload defined by a given `opName`, `overloadName`, a stack of
    StableIValues and the `TORCH_ABI_VERSION` of the user extension. This call will populate any return values of the
    op into the stack in their StableIValue form, with `ret0` at index 0, `ret1` at index 1, and so on.

    We caution against using this API to call functions that have been registered to the dispatcher by other extensions
    unless the caller can guarantee that the signature they expect matches that which the custom extension has
    registered.

### Versioning and Forward/Backward compatibility guarantees

We provide a `TORCH_ABI_VERSION` macro in `torch/headeronly/version.h` of the form

```
[ byte ][ byte ][ byte ][ byte ][ byte ][ byte ][ byte ][ byte ]
[MAJ   ][ MIN  ][PATCH ][                 ABI TAG              ]
```

In the present phase of development, APIs in the C-shim will be versioned based on major.minor.patch release that they are first introduced in, with 2.10 being the first release where this will be enforced. The ABI tag is reserved for future use.

Extensions can select the minimum abi version to be compatible with using:

```
#define TORCH_TARGET_VERSION (((0ULL + major) << 56) | ((0ULL + minor) << 48))
```

before including any stable headers or by passing the equivalent `-D` option to the compiler. Otherwise, the default will be the current `TORCH_ABI_VERSION`.

The above ensures that if a user defines `TORCH_TARGET_VERSION` to be 0x0209000000000000 (2.9) and attempts to use a C shim API `foo` that was introduced in version 2.10, a compilation error will be raised. Similarly, the C++ wrapper APIs in `torch/csrc/stable` are compatible with older libtorch binaries up to the TORCH_ABI_VERSION they are exposed in and forward compatible with newer libtorch binaries.

C++ APIs in ``torch/csrc/stable`` or ``torch/headeronly`` are subject to the same FC/BC policy as the rest of PyTorch (see [policy](https://github.com/pytorch/pytorch/wiki/PyTorch's-Python-Frontend-Backward-and-Forward-Compatibility-Policy)). LibTorch ABI stable C shim APIs are guaranteed to have at least a two year compatibility window.
