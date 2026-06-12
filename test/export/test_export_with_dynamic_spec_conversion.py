# Owner(s): ["oncall: export"]
"""Runs the test_export suite through the new unbacked ShapesSpec/ParamsSpec API.

Each test's named-``Dim`` ``dynamic_shapes`` is auto-converted to a ``ShapesSpec``
(named Dim -> ShapeVar) and re-exported with ``strict=True``. Tests using
``Dim.DYNAMIC``/``AUTO``, ``ShapesCollection``/``AdditionalInputs``, or no
``dynamic_shapes`` pass through unchanged. Tests that can't run on the unbacked
path are marked ``@testing.expectedFailureDynamicSpecConversion`` with a reason.
"""

import torch
from torch.export import export
from torch.export.dynamic_shapes import _combine_args, _DerivedDim, _DimHint
from torch.fx.experimental.dynamic_spec import (
    DictSpec,
    IntVar,
    ParamsSpec,
    SeqSpec,
    ShapesSpec,
    ShapeVar,
    STATIC,
    TensorSpec,
)
from torch.utils._sympy.numbers import int_oo


try:
    from . import test_export, testing
except ImportError:
    import test_export  # @manual=fbcode//caffe2/test:test_export-library

    import testing  # @manual=fbcode//caffe2/test:test_export-library


test_classes = {}


# ---- named-Dim dynamic_shapes -> ShapesSpec converter ---------------------


def _default_min():
    # Tensor dims default to min=1 so `Eq(k*u, 0)`-style internal checks resolve
    # via the range instead of DDE-ing. Under backed_size_oblivious the test
    # deliberately exercises 0/1 sizes, so match it with min=0.
    import torch.fx.experimental._config as fx_config

    return 0 if fx_config.backed_size_oblivious else 1


def _bounds(d, force_min):
    kw = {}
    lo, hi = getattr(d, "min", None), getattr(d, "max", None)
    if lo is not None and lo != -int_oo and lo > 0:
        kw["min"] = int(lo)
    elif force_min:
        kw["min"] = force_min
    if hi is not None and hi != int_oo:
        kw["max"] = int(hi)
    return kw


def _conv_dim(d, cache):
    # Named Dims unify by NAME (two `Dim("b")` are the same dim) -> cache by name.
    if d is None:
        return STATIC
    if isinstance(d, _DimHint):
        raise NotImplementedError("Dim.DYNAMIC/AUTO has no ShapesSpec equivalent")
    if isinstance(d, bool):
        raise NotImplementedError("bool dim")
    if isinstance(d, int):
        return int(d)
    if isinstance(d, _DerivedDim):
        return d.fn(_named(d.root, cache))
    return _named(d, cache)


def _named(d, cache):
    key = d.__name__
    if key not in cache:
        cache[key] = ShapeVar(d.__name__, **_bounds(d, _default_min()))
    return cache[key]


def _conv_tensor(entry, t, cache):
    n = t.dim()
    dims = [STATIC] * n
    if entry is None:
        pass
    elif isinstance(entry, dict):
        for i, d in entry.items():
            dims[int(i)] = _conv_dim(d, cache)
    elif isinstance(entry, (tuple, list)):
        for i, d in enumerate(entry):
            dims[i] = _conv_dim(d, cache)
    else:
        raise NotImplementedError(f"tensor entry {type(entry).__name__}")
    return TensorSpec(dims)


def _has_dyn(spec):
    if isinstance(spec, TensorSpec):
        return any(not isinstance(x, int) and x is not STATIC for x in spec)
    if isinstance(spec, (IntVar, torch.SymInt)):
        return True
    if isinstance(spec, DictSpec):
        return any(_has_dyn(v) for v in spec._entries.values())
    if isinstance(spec, SeqSpec):
        return any(_has_dyn(v) for v in spec._entries)
    return False


def _conv_val(entry, val, cache):
    if isinstance(val, torch.Tensor):
        return _conv_tensor(entry, val, cache)
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        if entry is None or isinstance(entry, int):
            return None
        if isinstance(entry, (_DerivedDim, _DimHint)):
            raise NotImplementedError("derived/auto scalar int")
        return IntVar(entry.__name__, **_bounds(entry, None))
    if isinstance(val, dict):
        e = entry or {}
        kids = {k: _conv_val(e.get(k), v, cache) for k, v in val.items()}
        kids = {k: v for k, v in kids.items() if v is not None and _has_dyn(v)}
        return DictSpec(kids) if kids else None
    if isinstance(val, (list, tuple)):
        e = entry if isinstance(entry, (list, tuple)) else []
        kids = [
            _conv_val(e[i] if i < len(e) else None, v, cache) for i, v in enumerate(val)
        ]
        if any(k is not None and _has_dyn(k) for k in kids):
            return SeqSpec([k if k is not None else STATIC for k in kids])
        return None
    return None


def _contains_named_dim(ds):
    found = [False]

    def walk(x):
        if isinstance(x, (_DerivedDim,)) or (
            hasattr(x, "__name__") and not isinstance(x, type)
        ):
            found[0] = True
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    walk(ds)
    return found[0]


def _to_shapes_spec(mod, args, kwargs, ds):
    cache = {}
    combined = _combine_args(mod, args, kwargs or {})
    if isinstance(ds, dict):
        norm = ds
    elif isinstance(ds, (tuple, list)):
        names = list(combined.keys())
        norm = {names[i]: e for i, e in enumerate(ds)}
    else:
        raise NotImplementedError(f"top-level dynamic_shapes {type(ds).__name__}")
    params = {}
    for name, entry in norm.items():
        if name not in combined:
            raise NotImplementedError(f"ds key {name!r} not a param")
        spec = _conv_val(entry, combined[name], cache)
        if spec is not None and _has_dyn(spec):
            params[name] = spec
    return ShapesSpec(params=ParamsSpec(params)) if params else None


def mocked_dynamic_spec_export(*args, **kwargs):
    ds = kwargs.get("dynamic_shapes")
    # Only convert named-Dim specs; pass everything else (Dim.DYNAMIC/AUTO,
    # ShapesCollection/AdditionalInputs, None, explicit strict) straight through.
    if ds is None or "strict" in kwargs or not _contains_named_dim(ds):
        return export(*args, **kwargs)
    mod = args[0]
    pos = args[1] if len(args) > 1 else kwargs.get("args", ())
    kw = args[2] if len(args) > 2 else kwargs.get("kwargs")
    spec = _to_shapes_spec(mod, pos, kw, ds)
    if spec is None:
        return export(*args, **kwargs)
    new_kwargs = {k: v for k, v in kwargs.items() if k != "dynamic_shapes"}
    new_kwargs["dynamic_shapes"] = spec
    new_kwargs["strict"] = True
    return export(*args, **new_kwargs)


def make_dynamic_cls(cls):
    test_class = testing.make_test_cls_with_mocked_export(
        cls,
        "DynamicSpecConversion",
        test_export.DYNAMIC_SPEC_SUFFIX,
        mocked_dynamic_spec_export,
        xfail_prop="_expected_failure_dynamic_spec_conversion",
    )
    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


tests = [
    test_export.TestDynamismExpression,
    test_export.TestExport,
]
for test in tests:
    make_dynamic_cls(test)
del test


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
