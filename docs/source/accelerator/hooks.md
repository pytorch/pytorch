# Accelerator Hooks

## Background

PyTorch exposes a hooks interface that lets accelerator vendors and extensions provide device-specific implementations for core runtime needs (device selection, allocators, streams, RNGs, pointer queries) without changing core code paths. Hooks enable:

- a clean separation between runtime logic and device-specific behavior;
- pluggable implementations (built-in or loaded as an extension);
- reuse of PyTorch abstractions while delegating platform details.

Common scenarios:

- registering a custom device and allocators;
- providing device-aware RNGs and generators;
- mapping pointers to device info (host/device/pinned);
- managing current device and device exchange;
- supplying custom streams and events.

## Design

Hook implementations derive from `at::PrivateUse1HooksInterface` and are registered with the runtime. The runtime calls hook methods to:

- enumerate and set devices;
- obtain allocators (including pinned allocators);
- query pointer attributes (is device memory, pinned, unmanaged);
- get default or new `at::Generator` instances for the device.

This keeps the runtime device-agnostic: hook implementations map these calls to accelerator APIs and types.

## OpenReg example

OpenReg is a compact test extension that implements the hooks interface for a `PrivateUse1` device. It demonstrates the minimal responsibilities of a hooks implementation and how to register it with PyTorch.

### What OpenReg provides

- Registration: a static initializer registers [`OpenRegHooksInterface`](https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegHooks.h) so the runtime can call it when the extension is loaded.
- Device management: `deviceCount()`, `setCurrentDevice`, `getCurrentDevice`, `exchangeDevice`, and `maybeExchangeDevice` map to OpenReg's device APIs.
- Availability: `isBuilt()` and `isAvailable()` report build-time and runtime availability.
- Allocators: `getPinnedMemoryAllocator()` returns `at::getHostAllocator(at::kPrivateUse1)`; OpenReg also implements host and device allocators.
- Pointer inspection: `isPinnedPtr` and `getDeviceFromPtr` call `orPointerGetAttributes` and convert results to `at::Device` when appropriate.
- RNG: `getDefaultGenerator` and `getNewGenerator` return OpenReg-specific `at::Generator` instances backed by `OpenRegGeneratorImpl`.

### Hook Registration

This snippet from `OpenRegHooks.cpp` registers the hooks at module load:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegHooks.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG HOOK REGISTER
    :end-before: LITERALINCLUDE END: OPENREG HOOK REGISTER
    :linenos:
```

### Hooks Interface

Key methods implemented by OpenReg are declared in `OpenRegHooks.h`, for example, generator related hooks like:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegHooks.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG HOOK EXAMPLES
    :end-before: LITERALINCLUDE END: OPENREG HOOK EXAMPLES
    :linenos:
```
