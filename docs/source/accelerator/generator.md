# Random Number Generator

## Background

OpenReg Generator provides device-specific random number generation capabilities for the OpenReg accelerator backend. As part of PyTorch's extensibility framework, OpenReg implements custom random number generators that integrate seamlessly with PyTorch's `torch.Generator` infrastructure.

The generator module enables:

- **Per-device RNG state management**: Each OpenReg device maintains its own independent random number generator with isolated state.
- **Reproducible computations**: Support for manual seeding ensures deterministic behavior when needed.
- **PyTorch API compatibility**: Generators work with PyTorch's random operations (e.g., `torch.rand`, `torch.randn`) on OpenReg devices.
- **State serialization**: RNG states can be saved and restored for checkpointing and resuming experiments.

OpenReg's generator implementation subclasses `at::CPUGeneratorImpl`, leveraging CPU-based random number generation while maintaining proper device attribution. This design pattern is suitable for accelerators that delegate random number generation to the CPU or use CPU-compatible algorithms.

## Design

The following table lists the basic functionalities that accelerator vendors need to implement for random number generation:

| Functionality       | Description                                                                  | Application Scenario                                                 |
| ------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `get_rng_state`     | Returns the current RNG state as a ByteTensor for a specific device          | Save RNG state for model checkpointing or experiment reproducibility |
| `get_rng_state_all` | Returns RNG states for all devices as a list of ByteTensors                  | Save multi-device RNG states in distributed or multi-GPU scenarios   |
| `set_rng_state`     | Restores a previously saved RNG state for a specific device                  | Resume experiments from checkpoints with exact RNG state             |
| `set_rng_state_all` | Restores RNG states for all devices from a list of ByteTensors               | Restore multi-device RNG states in distributed settings              |
| `manual_seed`       | Sets a specific seed for the current device                                  | Initialize RNG with a known seed for reproducible experiments        |
| `manual_seed_all`   | Sets the same seed for all devices                                           | Ensure synchronized random number generation across all devices      |
| `seed`              | Sets a random seed for the current device                                    | Initialize RNG with non-deterministic seed for production            |
| `seed_all`          | Sets random seeds for all devices                                            | Initialize all device RNGs with non-deterministic seeds              |
| `initial_seed`      | Returns the initial seed used to initialize the generator for current device | Retrieve the seed for debugging or logging purposes                  |

## Implementation

This section demonstrates how to implement RNG state management using `set_rng_state` as an example.

### Python Side

The Python implementation provides a user-friendly interface that handles device specification and delegates to C++ bindings:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/openreg/random.py
    :language: python
    :start-after: LITERALINCLUDE START: OPENREG GENERATOR PY WRAPPER EXAMPLE
    :end-before: LITERALINCLUDE END: OPENREG GENERATOR PY WRAPPER EXAMPLE
    :linenos:
```

**Key implementation details:**

1. **Device normalization**: Accepts flexible device specifications (`int`, `str`, or `torch.device`) and normalizes them to a device index.
2. **Device index resolution**: Defaults to the current device if no specific index is provided.
3. **C++ delegation**: Calls `torch_openreg._C._get_default_generator(idx)` to obtain the device-specific generator from C++.
4. **State manipulation**: Uses the generator's `set_state()` method to restore the RNG state.

### C++ Side

The C++ implementation manages per-device generator instances and provides access to them:

**Generator Implementation Header:**

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGenerator.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG GENERATOR IMPL HEADER
    :end-before: LITERALINCLUDE END: OPENREG GENERATOR IMPL HEADER
    :linenos:
```

**Generator Implementation:**

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGenerator.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG GET DEFAULT GENERATOR IMPL
    :end-before: LITERALINCLUDE END: OPENREG GET DEFAULT GENERATOR IMPL
    :linenos:
```

**Key implementation details:**

1. **Inheritance**: `OpenRegGeneratorImpl` extends `at::CPUGeneratorImpl` for CPU-compatible RNG behavior.
2. **Device attribution**: Sets `device_` to `PrivateUse1` with the correct device index and configures `key_set_` for dispatch.
3. **Singleton management**: `getDefaultOpenRegGenerator` maintains a static vector of per-device default generators.
4. **Lazy initialization**: Generators are created and seeded on first access using a static initialization pattern.
5. **Device validation**: Validates device indices and defaults to the current device when the index is -1.

## Integration

The following sections demonstrate how random number generator functionalities integrate from user-facing Python APIs down to the C++ implementation. We use `manual_seed` as an example to illustrate the complete layer-by-layer flow.

### Layer 1: User Code

User code initiates the operation by calling the high-level API to set a random seed:

```python
import torch_openreg

# Set seed for reproducible random number generation
torch.openreg.manual_seed(42)
```

### Layer 2: Python Wrapper Implementation

The Python wrapper handles device resolution and calls the C++ binding:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/openreg/random.py
    :language: python
    :start-after: LITERALINCLUDE START: OPENREG MANUAL SEED
    :end-before: LITERALINCLUDE END: OPENREG MANUAL SEED
    :linenos:
    :emphasize-lines: 11
```

**Layer 2 responsibilities:**

1. **Input validation**: Converts the seed to an integer.
2. **Device resolution**: Gets the current device index via `current_device()`.
3. **Generator access**: Calls `torch_openreg._C._get_default_generator(idx)` to obtain the device-specific generator.
4. **Seed application**: Invokes the generator's `manual_seed(seed)` method.

### Layer 3: C++ Binding - Generator Access

The C++ extension exposes `_getDefaultGenerator` to Python for accessing device generators:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/Module.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG MODULE METHODS
    :end-before: LITERALINCLUDE END: OPENREG MODULE METHODS
    :linenos:
    :emphasize-lines: 3
```

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/Module.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG GET DEFAULT GENERATOR
    :end-before: LITERALINCLUDE END: OPENREG GET DEFAULT GENERATOR
    :linenos:
    :emphasize-lines: 11-12
```

**Layer 3 responsibilities:**

1. **Argument unpacking**: Extracts the device index from Python arguments using `THPUtils_unpackLong`.
2. **Device construction**: Creates a `c10::Device` object with `PrivateUse1` type and the device index.
3. **Context delegation**: Calls `at::globalContext().defaultGenerator(device)` to retrieve the generator.
4. **Python wrapping**: Wraps the C++ generator in a `THPGenerator` Python object using `THPGenerator_initDefaultGenerator`.

### Layer 4: PyTorch Core Context

PyTorch's Context dispatches to the registered hooks to obtain the device-specific generator:

```{eval-rst}
.. literalinclude:: ../../../aten/src/ATen/Context.h
    :language: c++
    :lines: 61-73
    :linenos:
    :emphasize-lines: 8-9
```

**Layer 4 responsibilities:**

1. **Device type check**: Determines if the device is CPU or an accelerator.
2. **Hook dispatch**: For accelerator devices, calls `getAcceleratorHooksInterface(device_type).getDefaultGenerator(device.index())`.
3. **Generator return**: Returns the device-specific generator reference.

### Layer 5: Accelerator Hooks

The hooks interface delegates to the device-specific generator implementation:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegHooks.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG HOOK EXAMPLES
    :end-before: LITERALINCLUDE END: OPENREG HOOK EXAMPLES
    :linenos:
```

**Layer 5 responsibilities:**

1. **Hook implementation**: Overrides `getDefaultGenerator` from `PrivateUse1HooksInterface`.
2. **Default generator delegation**: `getDefaultGenerator` calls `getDefaultOpenRegGenerator(device_index)` to access the per-device singleton.

### Layer 6: Device-Specific Generator Implementation

The final layer manages the actual per-device generator instances:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGenerator.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG GET DEFAULT GENERATOR IMPL
    :end-before: LITERALINCLUDE END: OPENREG GET DEFAULT GENERATOR IMPL
    :linenos:
```

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGenerator.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG CREATE GENERATOR IMPL
    :end-before: LITERALINCLUDE END: OPENREG CREATE GENERATOR IMPL
    :linenos:
```

**Layer 6 responsibilities:**

1. **Static storage**: Maintains a static vector `default_generators` for per-device generator instances.
2. **Lazy initialization**: Initializes all device generators on first access using a static lambda.
3. **Generator creation**: Creates generator instances using `createOpenRegGenerator(i)`, which internally calls `createOpenRegGenerator(device_index)`.
4. **Initial seeding**: Seeds each generator during initialization.
5. **Device validation**: Validates the device index and defaults to the current device if the index is -1.
6. **Generator return**: Returns a const reference to the appropriate device's generator.
