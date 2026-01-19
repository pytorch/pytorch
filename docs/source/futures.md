```{eval-rst}
.. currentmodule:: torch.futures
```

(futures-docs)=

# torch.futures

This package provides a {class}`~torch.futures.Future` type that encapsulates
an asynchronous execution and a set of utility functions to simplify operations
on {class}`~torch.futures.Future` objects. Currently, the
{class}`~torch.futures.Future` type is primarily used by the
{ref}`distributed-rpc-framework`.

```{eval-rst}
.. automodule:: torch.futures
```

```{eval-rst}
.. autoclass:: Future
    :inherited-members:
```

```{eval-rst}
.. autofunction:: collect_all
```

```{eval-rst}
.. autofunction:: wait_all
```
