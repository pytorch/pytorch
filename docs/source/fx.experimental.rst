.. currentmodule:: torch.fx.experimental

torch.fx.experimental
=====================

.. warning::
   These APIs are experimental and subject to change without notice.

torch.fx.experimental.symbolic_shapes
-------------------------------------
.. currentmodule:: torch.fx.experimental.symbolic_shapes

.. autoclass:: torch.fx.experimental.symbolic_shapes.ShapeEnv
    :members:

.. autoclass:: torch.fx.experimental.symbolic_shapes.DimDynamic
    :members:

.. autoclass:: torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint
    :members:

.. autoclass:: torch.fx.experimental.symbolic_shapes.RelaxedUnspecConstraint
    :members:

.. autoclass:: torch.fx.experimental.symbolic_shapes.EqualityConstraint
    :members:

.. autoclass:: torch.fx.experimental.symbolic_shapes.SymbolicContext
    :members:

.. autoclass:: torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext
    :members:

.. autoclass:: torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext
    :members:

.. autoclass:: torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext
    :members:

.. autoclass:: torch.fx.experimental.symbolic_shapes.DynamicDimConstraintPrinter
    :members:

.. autoclass:: torch.fx.experimental.symbolic_shapes.DimConstraints
    :members:


.. autofunction:: torch.fx.experimental.symbolic_shapes.hint_int
.. autofunction:: torch.fx.experimental.symbolic_shapes.is_concrete_int
.. autofunction:: torch.fx.experimental.symbolic_shapes.is_concrete_bool
.. autofunction:: torch.fx.experimental.symbolic_shapes.has_free_symbols
.. autofunction:: torch.fx.experimental.symbolic_shapes.definitely_true
.. autofunction:: torch.fx.experimental.symbolic_shapes.definitely_false
.. autofunction:: torch.fx.experimental.symbolic_shapes.parallel_or
.. autofunction:: torch.fx.experimental.symbolic_shapes.parallel_and
.. autofunction:: torch.fx.experimental.symbolic_shapes.sym_eq
.. autofunction:: torch.fx.experimental.symbolic_shapes.constrain_range
.. autofunction:: torch.fx.experimental.symbolic_shapes.constrain_unify
.. autofunction:: torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr
.. autofunction:: torch.fx.experimental.symbolic_shapes.statically_known_true
