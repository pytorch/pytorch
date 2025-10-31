# LibTorch Stable ABI

## Overview

The LibTorch Stable ABI (Application Binary Interface) provides an interface for extending PyTorch functionality without being tightly coupled to specific PyTorch versions. This enables the development of custom operators and extensions that remain compatible across PyTorch releases.

The stable ABI consists of three main components:

1. **Stable C headers** - Low-level C API implemented by libtorch (primarily `torch/csrc/inductor/aoti_torch/c/shim.h`)
2. **Header-only C++ library** - Standalone utilities implemented in only headers such that there is no dependence on libtorch (`torch/headeronly/*`)
3. **Stable C++ wrappers** - High-level C++ convenience wrappers (`torch/csrc/stable/*`)

We discuss each of these in detail

### `torch/headeronly`

This is a set of inlined C++ headers are completely decoupled from libtorch. The headers consist of certain utilities that might be familiar to custom extension writers. For example, the
`c10::ScalarType` enum lives here as `torch::headeronly::ScalarType`.

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

### Stable C headers

The stable C headers used by AOTInductor form the foundation of the stable ABI. However, this is **use at your own risk**. For example, users must handle the memory lifecycle of objects returned by certain APIs.
 Further, the stack-based APIs discussed below which allow the user to call the PyTorch dispatcher don't provide strong guarantees on forward and backward compatibility.

Unless absolutely necessary, we recommend the high-level C++ API in `torch/csrc/stable`
which will handle all the rough edges of the C API for the user.


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
| RAIIATH (outdated) | raw bitwise copy of underlying AtenTensorHandle into leading bytes of uint64_t | at::Tensor |  Tensor |
| torch::headeronly::ScalarType | raw bitwise copy of the translated underlying enum into leading bytes of uint64_t | torch::headeronly::ScalarType | ScalarType |
| int32_t | raw bitwise copy into leading bytes of uint64_t | at::Layout | Layout |
| int32_t | raw bitwise copy into leading bytes of uint64_t | at::MemoryFormat | MemoryFormat |
| bool | raw bitwise copy into leading bytes of uint64_t | bool | bool |
| int64_t | raw bitwise copy into leading bytes of uint64_t | int64_t | int |
| double | raw bitwise copy into leading bytes of uint64_t | double | float |
| ? | ? | c10::Device | Device |
| ? | ? | c10::Stream | Stream |
| ? | ? | c10::complex<double> | complex |
| ? | ? | at::Scalar | Scalar |
| ? | ? | std::string/const char*/ivalue::ConstantString | str |
| ? | ? | at::Storage | Storage |
| ? | ? | at::Generator | Generator |
| ? | ? | c10::List\<T> | Type[] |
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

### Stack-based APIs

The above is relevant in two places:

1. `STABLE_TORCH_LIBRARY`
    Unlike `TORCH_LIBRARY`, the dispatcher expects kernels registered via `STABLE_TORCH_LIBRARY` to be boxed. This means they must have the signature `(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) -> void`.We plan to eventually abstract away the need for manual boxing, but, for the time being, please use `from` and `to`.

    ```cpp
    Tensor my_amax_vec(Tensor t) {
        std::vector<int64_t> v = {0,1};
        return amax(t, v, false);
    }

    void boxed_my_amax_vec(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
        auto res = my_amax_vec(to<Tensor>(stack[0]));
        stack[0] = from(res);
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
