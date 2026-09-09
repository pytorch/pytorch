# Aliases in torch.optim

Most optimizer classes are available both from ``torch.optim`` and from the
nested module in which they are defined. For example, ``torch.optim.Adam`` and
``torch.optim.adam.Adam`` refer to the same class. While some functional optimizer APIs are exposed in their respective nested modules, all functional
optimizer APIs are available in the {mod}`torch.optim.functional` namespace; see
{ref}`functional-optimizer-api` for usage guidance.

```{eval-rst}
.. automodule:: torch.optim.adadelta
.. currentmodule:: torch.optim.adadelta
.. autosummary::
   :toctree: generated
   :nosignatures:

    Adadelta
    adadelta
```

```{eval-rst}
.. automodule:: torch.optim.adagrad
.. currentmodule:: torch.optim.adagrad
.. autosummary::
   :toctree: generated
   :nosignatures:

    Adagrad
    adagrad
```

```{eval-rst}
.. automodule:: torch.optim.adam
.. currentmodule:: torch.optim.adam
.. autosummary::
   :toctree: generated
   :nosignatures:

    Adam
    adam
```

```{eval-rst}
.. automodule:: torch.optim.adamax
.. currentmodule:: torch.optim.adamax
.. autosummary::
   :toctree: generated
   :nosignatures:

    Adamax
    adamax
```

```{eval-rst}
.. automodule:: torch.optim.adamw
.. currentmodule:: torch.optim.adamw
.. autosummary::
   :toctree: generated
   :nosignatures:

    AdamW
    adamw
```

```{eval-rst}
.. automodule:: torch.optim.asgd
.. currentmodule:: torch.optim.asgd
.. autosummary::
   :toctree: generated
   :nosignatures:

    ASGD
    asgd
```

```{eval-rst}
.. automodule:: torch.optim.lbfgs
.. currentmodule:: torch.optim.lbfgs
.. autosummary::
   :toctree: generated
   :nosignatures:

    LBFGS
```

```{eval-rst}
.. automodule:: torch.optim.nadam
.. currentmodule:: torch.optim.nadam
.. autosummary::
   :toctree: generated
   :nosignatures:

    NAdam
    nadam
```

```{eval-rst}
.. automodule:: torch.optim.radam
.. currentmodule:: torch.optim.radam
.. autosummary::
   :toctree: generated
   :nosignatures:

    RAdam
    radam
```

```{eval-rst}
.. automodule:: torch.optim.rmsprop
.. currentmodule:: torch.optim.rmsprop
.. autosummary::
   :toctree: generated
   :nosignatures:

    RMSprop
    rmsprop
```

```{eval-rst}
.. automodule:: torch.optim.rprop
.. currentmodule:: torch.optim.rprop
.. autosummary::
   :toctree: generated
   :nosignatures:

    Rprop
    rprop
```

```{eval-rst}
.. automodule:: torch.optim.sgd
.. currentmodule:: torch.optim.sgd
.. autosummary::
   :toctree: generated
   :nosignatures:

    SGD
    sgd
```

```{eval-rst}
.. automodule:: torch.optim.sparse_adam
.. currentmodule:: torch.optim.sparse_adam
.. autosummary::
   :toctree: generated
   :nosignatures:

    SparseAdam
```
