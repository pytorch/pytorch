.. currentmodule:: torch.fx.experimental

torch.fx.experimental
=====================

.. warning::
   These APIs are experimental and subject to change without notice.

torch.fx.experimental.symbolic_shapes
-------------------------------------
.. currentmodule:: torch.fx.experimental.symbolic_shapes
.. automodule:: torch.fx.experimental.symbolic_shapes

.. autosummary::
    :toctree: generated
    :nosignatures:

    ShapeEnv
    DimDynamic
    StrictMinMaxConstraint
    RelaxedUnspecConstraint
    EqualityConstraint
    SymbolicContext
    StatelessSymbolicContext
    StatefulSymbolicContext
    SubclassSymbolicContext
    DimConstraints
    ShapeEnvSettings

    hint_int
    is_concrete_int
    is_concrete_bool
    has_free_symbols
    definitely_true
    definitely_false
    guard_size_oblivious
    parallel_or
    parallel_and
    sym_eq
    constrain_range
    constrain_unify
    canonicalize_bool_expr
    statically_known_true
    lru_cache
