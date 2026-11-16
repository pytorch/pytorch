# torch.accelerator

(lazy-initialization-and-fork-safety-note)=
## Lazy Initialization and Fork Safety

Accelerator runtimes (CUDA, XPU, MPS, etc.) are initialized lazily — only
when the first operation that touches the device runs. This ensures that
`import torch` and capability queries do not poison subsequent `fork()`
calls. See {ref}`multiprocessing-poison-fork-note` for background.

Certain APIs need to be callable before forking (e.g., `device_count()`,
`is_available()`), and some backends provide opt-in mechanisms to make
these fork-safe (e.g., CUDA via NVML, XPU via Level Zero Sysman). To keep
behavior and runtime state consistent between `torch.accelerator` and
per-backend modules, the following APIs are delegated to each backend:

- **`is_available()` / `device_count()`** should ideally answer without
  bringing up the runtime, since `DataLoader` and similar tools rely on
  calling them before forking. Whether this is achievable depends on the
  backend, so `torch.accelerator` forwards to the corresponding backend
  implementation.

- **`_lazy_call()`** is used for deferred RNG management. Calling
  `manual_seed()` before forking should not force runtime initialization.
  `torch.accelerator` wraps the seeding callback via `_lazy_call()`, which
  forwards to the backend's own callback queue (CUDA, XPU, MTIA, ...).
  Each backend owns its init flag and callback queue. If a backend does not
  provide `_lazy_call` (e.g., MPS), the callback executes immediately.


```{eval-rst}
.. automodule:: torch.accelerator
   :no-members:
```

```{eval-rst}
.. currentmodule:: torch.accelerator
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    device_count
    is_available
    current_accelerator
    set_device_index
    set_device_idx
    current_device_index
    current_device_idx
    get_device_capability
    set_stream
    current_stream
    synchronize
    device_index
```

## Graphs

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    Graph
```

```{eval-rst}
.. automodule:: torch.accelerator.memory
```
```{eval-rst}
.. currentmodule:: torch.accelerator.memory
```

## Memory management
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

     empty_cache
     empty_host_cache
     get_memory_info
     max_memory_allocated
     max_memory_reserved
     memory_allocated
     memory_reserved
     memory_stats
     reset_accumulated_memory_stats
     reset_peak_memory_stats
```

```{eval-rst}
.. automodule:: torch.accelerator.random
```
```{eval-rst}
.. currentmodule:: torch.accelerator.random
```

## Random Number Generator
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

     get_rng_state
     get_rng_state_all
     initial_seed
```
