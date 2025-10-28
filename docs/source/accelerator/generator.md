# Random Number Generator

## Background

The RNG (generator) module exposes backend-specific random number generation via a Python wrapper (`torch.Generator`) and a C++ `at::Generator` implementation. A backend must provide:

- A generator implementation inheriting from the appropriate `GeneratorImpl` (CPU-based backends commonly subclass `at::CPUGeneratorImpl`).
- A way to create per-device default generators and return them to Python via a `_C` binding (e.g. `_get_default_generator`).
- Python helpers that call into the C++ bindings for convenience (seed, state, manual seeding).

This guide shows a minimal, low-risk integration pattern (examples taken from the OpenReg test extension included with this repo).

## Design

Key responsibilities for a backend RNG implementation:

- Default generators: maintain one default generator per device and expose accessors.
- Creation API: provide a `create<Backend>Generator` helper to construct new `at::Generator` instances for a given device.
- Seed semantics: support `manual_seed`, `manual_seed_all`, `seed_all`, `initial_seed`, and ensure deterministic reproducibility when required.

The examples below demonstrate both Python and C++ side integration points.

## Implementation

### Python Integration

Provide a small Python wrapper module that imports the compiled `_C` bindings and exposes a friendly API for users. The wrapper should:

- Lazily initialize the backend device(s) when needed.
- Convert `device` arguments `(int | str | torch.device)` into a device index.
- Call into `_get_default_generator(idx)` to access per-device default generators.
- Provide high-level functions: `get_rng_state`, `get_rng_state_all`, `set_rng_state`, `set_rng_state_all`, `manual_seed`, `manual_seed_all`, `seed`, `seed_all`, `initial_seed`.

Example (OpenReg test extension) Python wrapper:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/openreg/random.py
    :language: python
    :start-after: LITERALINCLUDE START: OPENREG GENERATOR PY WRAPPER EXAMPLE
    :end-before: LITERALINCLUDE END: OPENREG GENERATOR PY WRAPPER EXAMPLE
    :linenos:
```

### C++ Integration

On the C++ side, implement a `GeneratorImpl` subclass and helper functions to create and return the default generators. Typical pieces:

- A `GeneratorImpl` subclass (e.g. `OpenRegGeneratorImpl`) that implements backend-specific behavior.
- A `create<Backend>Generator(device_index)` that constructs and wraps the backend generator implementation into an `at::Generator`.
- A `getDefault<Backend>Generator(device_index)` that returns a const reference to the per-device default generator (initializing the default vector on first use).
- Bind `_get_default_generator` in the extension `Module.cpp` so Python can call into the runtime.

C++ examples from the OpenReg test extension:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGenerator.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG GENERATOR IMPL HEADER
    :end-before: LITERALINCLUDE END: OPENREG GENERATOR IMPL HEADER
    :linenos:
```

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGenerator.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG GENERATOR IMPL
    :end-before: LITERALINCLUDE END: OPENREG GENERATOR IMPL
    :linenos:
```

And the Python binding that exposes `_get_default_generator`:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/Module.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG DEFAULT GENERATOR MODULE
    :end-before: LITERALINCLUDE END: OPENREG DEFAULT GENERATOR MODULE
    :linenos:
```

## Tests to Add

Add focused tests under your backend's `tests/` folder (see the [OpenReg tests](https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/tests) for examples):

- test_generator_creation: verify `torch.Generator(device="<backend>:0")` device type and index
- test_seed_and_manual_seed: check `manual_seed`, `seed_all`, `initial_seed`
- test_state_roundtrip: `state = get_rng_state(i); set_rng_state(state, i);` sequences match

Example tests excerpt (OpenReg):

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_rng.py
    :start-after: LITERALINCLUDE START: OPENREG GENERATOR TEST EXAMPLES
    :end-before: LITERALINCLUDE END: OPENREG GENERATOR TEST EXAMPLES
    :language: python
    :linenos:
```
