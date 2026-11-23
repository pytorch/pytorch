# torch.optim.functional

Functional optimizer APIs give callers complete control over the parameters,
gradients, and optimizer state. They are not pure functions; they update
parameters and optimizer-state tensors in place. See
{ref}`functional-optimizer-api` for their common contract and usage examples.

```{eval-rst}
.. automodule:: torch.optim.functional
.. currentmodule:: torch.optim.functional
.. autosummary::
    :toctree: generated
    :nosignatures:

    adadelta
    adafactor
    adagrad
    adam
    adamax
    adamw
    asgd
    muon
    nadam
    radam
    rmsprop
    rprop
    sgd
    sparse_adam
```
