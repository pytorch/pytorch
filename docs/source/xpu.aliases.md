# Aliases in torch.xpu

The following are aliases to their counterparts in ``torch.xpu`` in the nested namespaces in which they are defined. For any of these APIs, feel free to use the top-level version in ``torch.xpu`` like ``torch.xpu.seed`` or the nested version ``torch.xpu.random.seed``.

```{eval-rst}
.. automodule:: torch.xpu.random
.. currentmodule:: torch.xpu.random
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

```{eval-rst}
.. automodule:: torch.xpu.streams
.. currentmodule:: torch.xpu.streams
.. autosummary::
    :toctree: generated
    :nosignatures:

    Event
    Stream
```