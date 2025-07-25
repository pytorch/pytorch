# Joint with descriptors

Joint with descriptors is an experimental API for exporting a traced joint
graph that supports all of torch.compile's features in full generality and,
after processing, can be converted back into a differentiable callable that
can be executed as normal.  For example, it is used to implement autoparallel,
a system that takes a model and reshards inputs and parameters to make it
a distributed SPMD program.

```{eval-rst}
.. currentmodule:: torch._functorch.aot_autograd
.. autofunction:: aot_export_joint_with_descriptors
.. autofunction:: aot_compile_joint_with_descriptors
```

## Descriptors

```{eval-rst}
.. automodule:: torch._functorch._aot_autograd.descriptors
  :members:
```

## FX utilities

```{eval-rst}
.. automodule:: torch._functorch._aot_autograd.fx_utils
  :members:
```
