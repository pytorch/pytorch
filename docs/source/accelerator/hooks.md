# Accelerator Hooks

## Background

Accelerator hooks are the mechanism for integrating custom accelerator devices into PyTorch’s runtime.

## Design

The tables below list the hooks accelerator vendors should implement when integrating a new device backend. These hooks are categorized into two priority levels:

- **High‑priority hooks**: Core APIs that the PyTorch runtime directly depends on. Vendors should implement all high‑priority hooks to ensure core compatibility and basic device functionality.

- **Low‑priority hooks**: Device‑management and utility APIs that PyTorch does not directly depend on. These hooks enhance user experience and multi‑device support and are optional. Vendors can implement them based on specific requirements and use cases.

### High‑priority hooks

| Hook method                        | Description                                               | Application scenarios                                                            |
| ---------------------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `init()`                           | Initializes the accelerator runtime and device contexts   | Set up necessary state when PyTorch first accesses the device                    |
| `hasPrimaryContext(DeviceIndex)`   | Checks if a primary context exists for the device         | Determine whether device initialization has occurred                             |
| `getDefaultGenerator(DeviceIndex)` | Returns the default random number generator for a device  | Access the device’s primary RNG for reproducible random operations               |
| `getNewGenerator(DeviceIndex)`     | Creates a new independent random number generator         | Create isolated RNG instances for parallel operations                            |
| `getDeviceFromPtr(void*)`          | Determines which device a memory pointer belongs to       | Identify the accelerator device associated with a memory allocation              |
| `getPinnedMemoryAllocator()`       | Returns an allocator for pinned (page-locked) host memory | Allocate host memory that can be efficiently transferred to/from the accelerator |
| `isPinnedPtr(void*)`               | Checks if a pointer points to pinned memory               | Validate memory types before performing operations                               |

### Low‑priority hooks

| Hook method                        | Description                                                                  | Application scenarios                                                |
| ---------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `isBuilt()`                        | Returns whether the accelerator backend is built/compiled into the extension | Check whether the accelerator library is available at compile time   |
| `isAvailable()`                    | Returns whether the accelerator hardware is available at runtime             | Verify whether accelerator devices can be detected and initialized   |
| `deviceCount()`                    | Returns the number of available accelerator devices                          | Enumerate all available accelerator devices for device selection     |
| `setCurrentDevice(DeviceIndex)`    | Sets the active device for the current thread                                | Switch the current thread's context to a specific accelerator device |
| `getCurrentDevice()`               | Returns the currently active device index                                    | Query which accelerator device is active in the current thread       |
| `exchangeDevice(DeviceIndex)`      | Atomically exchanges the current device and returns the previous one         | Temporarily switch devices and restore the previous device afterward |
| `maybeExchangeDevice(DeviceIndex)` | Conditionally exchanges device only if the index is valid                    | Safely attempt device switching with validation                      |

## Implementation

For illustration, OpenReg (Open Registration) is a PyTorch integration example that fills the gap for out‑of‑tree accelerator backend integration. It demonstrates how vendors can register custom device backends—without modifying PyTorch core—by implementing the hooks interface (see [`at::PrivateUse1HooksInterface`][PrivateUse1HooksInterface.h]).

We use `getDefaultGenerator` as an example:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegHooks.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG HOOK EXAMPLES
    :end-before: LITERALINCLUDE END: OPENREG HOOK EXAMPLES
    :linenos:
```

In this implementation:

1. **Override the base interface**: The `getDefaultGenerator` method overrides the virtual method from [`at::PrivateUse1HooksInterface`][PrivateUse1HooksInterface.h].

2. **Delegate to the device‑specific implementation**: Call `getDefaultOpenRegGenerator(device_index)`, which manages a per‑device generator instance.

3. **Return a device‑specific generator**: The returned `at::Generator` wraps an `OpenRegGeneratorImpl` that implements device‑specific random number generation.

This pattern applies to all hooks: override the interface method, validate inputs, delegate to your device‑specific API, and return results in PyTorch’s expected format.

## Integration Example

The following sections demonstrate how PyTorch integrates with accelerator hooks when accessing the default random number generator. The example traces the complete flow from user-facing Python code down to the device-specific implementation.

### Layer 1: User Code

User code sets a deterministic seed by calling `manual_seed`:

```python
import torch
torch.openreg.manual_seed(42)
```

### Layer 2: Extension Python API

The Python API layer manages device selection and calls into the C++ extension (defined in [`torch_openreg/openreg/random.py`][random.py]):

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/openreg/random.py
    :language: python
    :start-after: LITERALINCLUDE START: OPENREG MANUAL SEED
    :end-before: LITERALINCLUDE END: OPENREG MANUAL SEED
    :linenos:
```

The `manual_seed` function obtains the current device index, calls `torch_openreg._C._get_default_generator(idx)` to get the device‑specific generator, and sets its seed.

### Layer 3: Python/C++ Bridge

The C++ extension exposes `_getDefaultGenerator` to Python, which bridges into PyTorch core:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/Module.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG GET DEFAULT GENERATOR
    :end-before: LITERALINCLUDE END: OPENREG GET DEFAULT GENERATOR
    :linenos:
    :emphasize-lines: 10-11
```

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/Module.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG MODULE METHODS
    :end-before: LITERALINCLUDE END: OPENREG MODULE METHODS
    :linenos:
    :emphasize-lines: 3
```

This function unpacks the device index from Python, creates a `PrivateUse1` device object, and calls `at::globalContext().defaultGenerator()`. PyTorch’s context then dispatches to the registered hooks.

### Layer 4: PyTorch Core Context

PyTorch’s `Context` class dispatches to the appropriate accelerator hooks ([`aten/src/ATen/Context.h`][Context.h]):

```{eval-rst}
.. literalinclude:: ../../../aten/src/ATen/Context.h
    :language: c++
    :lines: 60-103
    :linenos:
    :emphasize-lines: 8-9, 24-25
```

This layered architecture keeps PyTorch device‑agnostic while delegating hardware‑specific operations to accelerator implementations. Hooks are registered once at module load time:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegHooks.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG HOOK REGISTER
    :end-before: LITERALINCLUDE END: OPENREG HOOK REGISTER
    :linenos:
    :emphasize-lines: 4
```

### Layer 5: Accelerator Hooks

The hooks interface provides the abstraction PyTorch uses to delegate to device‑specific implementations:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegHooks.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG HOOK EXAMPLES
    :end-before: LITERALINCLUDE END: OPENREG HOOK EXAMPLES
    :linenos:
```

The `getDefaultGenerator` hook method overrides the base interface and delegates to `getDefaultOpenRegGenerator`, which manages the actual generator instances.

### Layer 6: Device-Specific Implementation

The device‑specific implementation manages per‑device generator instances:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGenerator.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG GET DEFAULT GENERATOR IMPL
    :end-before: LITERALINCLUDE END: OPENREG GET DEFAULT GENERATOR IMPL
    :linenos:
```

This function maintains a static vector of generators (one per device), initializes them on first access, validates the device index, and returns the appropriate generator instance.

[random.py]: https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/openreg/random.py#L48-L53 "random.py"
[Context.h]: https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/Context.h#L61-L102 "Context.h"
[PrivateUse1HooksInterface.h]: https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/detail/PrivateUse1HooksInterface.h#L15-L72 "PrivateUse1HooksInterface.h"
