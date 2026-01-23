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
.. currentmodule:: torch._functorch._aot_autograd.descriptors

.. autoclass:: AOTInput
  :members:

.. autoclass:: AOTOutput
  :members:

.. autoclass:: BackwardTokenAOTInput
  :members:

.. autoclass:: BackwardTokenAOTOutput
  :members:

.. autoclass:: BufferAOTInput
  :members:

.. autoclass:: DummyAOTInput
  :members:

.. autoclass:: DummyAOTOutput
  :members:

.. autoclass:: GradAOTOutput
  :members:

.. autoclass:: InputMutationAOTOutput
  :members:

.. autoclass:: IntermediateBaseAOTOutput
  :members:

.. autoclass:: ParamAOTInput
  :members:

.. autoclass:: PhiloxBackwardBaseOffsetAOTInput
  :members:

.. autoclass:: PhiloxBackwardSeedAOTInput
  :members:

.. autoclass:: PhiloxForwardBaseOffsetAOTInput
  :members:

.. autoclass:: PhiloxForwardSeedAOTInput
  :members:

.. autoclass:: PhiloxUpdatedBackwardOffsetAOTOutput
  :members:

.. autoclass:: PhiloxUpdatedForwardOffsetAOTOutput
  :members:

.. autoclass:: PlainAOTInput
  :members:

.. autoclass:: PlainAOTOutput
  :members:

.. autoclass:: SavedForBackwardsAOTOutput
  :members:

.. autoclass:: SubclassGetAttrAOTInput
  :members:

.. autoclass:: SubclassGetAttrAOTOutput
  :members:

.. autoclass:: SubclassSizeAOTInput
  :members:

.. autoclass:: SubclassSizeAOTOutput
  :members:

.. autoclass:: SubclassStrideAOTInput
  :members:

.. autoclass:: SubclassStrideAOTOutput
  :members:

.. autoclass:: SyntheticBaseAOTInput
  :members:

.. autoclass:: ViewBaseAOTInput
  :members:
```

## FX utilities

```{eval-rst}
.. automodule:: torch._functorch._aot_autograd.fx_utils
  :members:
```
