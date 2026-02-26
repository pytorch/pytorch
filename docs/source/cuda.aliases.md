# Aliases in torch.cuda

The following are aliases to their counterparts in ``torch.cuda`` in the nested namespaces in which they are defined. For any of these APIs, feel free to use the top-level version in ``torch.cuda`` like ``torch.cuda.seed`` or the nested version ``torch.cuda.random.seed``.

```{eval-rst}
.. automodule:: torch.cuda.random
.. currentmodule:: torch.cuda.random
.. autosummary::
    :toctree: generated
    :nosignatures:

    get_rng_state
    get_rng_state_all
    set_rng_state
    set_rng_state_all
    manual_seed
    manual_seed_all
    seed
    seed_all
    initial_seed
```

```{eval-rst}
.. automodule:: torch.cuda.graphs
.. currentmodule:: torch.cuda.graphs
.. autosummary::
    :toctree: generated
    :nosignatures:

    is_current_stream_capturing
    graph_pool_handle
    CUDAGraph
    graph
    make_graphed_callables
```

```{eval-rst}
.. automodule:: torch.cuda.streams
.. currentmodule:: torch.cuda.streams
.. autosummary::
    :toctree: generated
    :nosignatures:

    Stream
    ExternalStream
    Event
```