```{contents}
:depth: 2
:local: true
```

```{eval-rst}
.. testsetup::

    # These are hidden from the docs, but these are necessary for `doctest`
    # since the `inspect` module doesn't play nicely with the execution
    # environment for `doctest`
    import torch

    original_script = torch.jit.script
    def script_wrapper(obj, *args, **kwargs):
        obj.__module__ = 'FakeMod'
        return original_script(obj, *args, **kwargs)

    torch.jit.script = script_wrapper

    original_trace = torch.jit.trace
    def trace_wrapper(obj, *args, **kwargs):
        obj.__module__ = 'FakeMod'
        return original_trace(obj, *args, **kwargs)

    torch.jit.trace = trace_wrapper
```

(language-reference)=

# TorchScript Language Reference

:::{warning}
TorchScript is deprecated, please use
[torch.export](https://docs.pytorch.org/docs/stable/export.html) instead.
:::