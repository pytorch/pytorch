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
    ConvertIntKey
    CallMethodKey
    PropagateUnbackedSymInts
    DivideByKey
    InnerTensorKey

    hint_int
    is_concrete_int
    is_concrete_bool
    is_concrete_float
    has_free_symbols
    has_free_unbacked_symbols
    guard_or_true
    guard_or_false
    guard_size_oblivious
    sym_and
    sym_eq
    sym_or
    constrain_range
    constrain_unify
    canonicalize_bool_expr
    statically_known_true
    has_static_value
    lru_cache
    check_consistent
    compute_unbacked_bindings
    rebind_unbacked
    resolve_unbacked_bindings
    is_accessor_node

torch.fx.experimental.proxy_tensor
-------------------------------------

.. currentmodule:: torch.fx.experimental.proxy_tensor
.. automodule:: torch.fx.experimental.proxy_tensor

.. autosummary::
    :toctree: generated
    :nosignatures:

    make_fx
    handle_sym_dispatch
    get_proxy_mode
    maybe_enable_thunkify
    maybe_disable_thunkify
