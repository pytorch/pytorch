# torch.xpu
```{eval-rst}
.. automodule:: torch.xpu
```
```{eval-rst}
.. currentmodule:: torch.xpu
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    StreamContext
    can_device_access_peer
    current_device
    current_stream
    device
    device_count
    device_of
    get_arch_list
    get_device_capability
    get_device_name
    get_device_properties
    get_gencode_flags
    get_stream_from_external
    init
    is_available
    is_bf16_supported
    is_initialized
    is_tf32_supported
    set_device
    set_stream
    stream
    synchronize
```

## Random Number Generator
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    get_rng_state
    get_rng_state_all
    initial_seed
    manual_seed
    manual_seed_all
    seed
    seed_all
    set_rng_state
    set_rng_state_all
```

## Streams and events
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    Event
    Stream
```

```{eval-rst}
.. automodule:: torch.xpu.memory
```
```{eval-rst}
.. currentmodule:: torch.xpu.memory
```

## Memory management
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

     empty_cache
     get_per_process_memory_fraction
     max_memory_allocated
     max_memory_reserved
     mem_get_info
     memory_allocated
     memory_reserved
     memory_stats
     memory_stats_as_nested_dict
     reset_accumulated_memory_stats
     reset_peak_memory_stats
     set_per_process_memory_fraction
```

```{eval-rst}
.. toctree::
    :hidden:

    xpu.aliases.md
```