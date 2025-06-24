# mypy: allow-untyped-defs
import builtins
import contextlib
import functools
import inspect
import logging
import math
from collections import defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Callable, Optional, TYPE_CHECKING, Union

import torch
import torch.utils._pytree as pytree
from torch._dynamo.source import (
    AttrSource,
    GetItemSource,
    LocalSource,
    TensorProperty,
    TensorPropertySource,
)
from torch._dynamo.variables.builder import TrackedFake
from torch._export.passes.lift_constants_pass import ConstantAttrMap
from torch._export.utils import _fakify_params_buffers
from torch._guards import Source
from torch._library.fake_class_registry import FakeScriptObject
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.export import Constraint
from torch.export.dynamic_shapes import (
    _check_dynamic_shapes,
    _combine_args,
    _DimHint,
    _DimHintType,
    _IntWrapper,
    _process_dynamic_shapes,
    _RelaxedConstraint,
    _tree_map_with_path,
)
from torch.export.graph_signature import CustomObjArgument
from torch.fx.experimental import _config as config
from torch.fx.experimental.symbolic_shapes import (
    _find_user_code_frame,
    _suggest_fixes_for_data_dependent_error_non_strict,
    ConstraintViolationError,
    DimDynamic,
    EqualityConstraint,
    GuardOnDataDependentSymNode,
    RelaxedUnspecConstraint,
    ShapeEnv,
    StatelessSymbolicContext,
    SymIntSymbolicContext,
    ValueRanges,
)
from torch.utils._pytree import (
    GetAttrKey,
    KeyPath,
    MappingKey,
    SequenceKey,
    tree_map_with_path,
)
from torch.utils._sympy.numbers import int_oo


if TYPE_CHECKING:
    from sympy import Symbol


log = logging.getLogger(__name__)


class _KeyPath:
    """
    Wraps `KeyPath` to aid `isinstance` checks.
    """

    def __init__(self, kp: KeyPath):
        self.kp = kp


class _KeyPathTrie:
    """
    Builds a trie of `KeyPath` prefixes mapping to `Source` leaves.
    """

    def __init__(self):
        self.root = {}

    def add(self, kp: KeyPath, src: Source):
        assert len(kp) > 0
        *path, leaf = kp
        node = self.root
        for k in path:
            if k not in node:
                node[k] = {}
            node = node[k]
        node[leaf] = src

    def get(self, kp: KeyPath) -> tuple[Source, KeyPath]:
        node = self.root
        while not isinstance(node, Source):
            assert len(kp) > 0
            k, *kp = kp  # type: ignore[assignment]
            node = node[k]
        return node, kp


def make_sourced_prefixes(nn_module, args, kwargs) -> _KeyPathTrie:
    kp_args, kp_kwargs = tree_map_with_path(
        lambda kp, _: _KeyPath(kp),
        (tuple(None for _ in args), {k: None for k in kwargs}),  # noqa: C420
    )
    kp_combined_args = _combine_args(nn_module, kp_args, kp_kwargs)

    sourced_prefixes = _KeyPathTrie()
    for name, struct in kp_combined_args.items():
        src = LocalSource(name)

        if isinstance(struct, _KeyPath):
            sourced_prefixes.add(struct.kp, src)
        elif isinstance(struct, tuple):
            for i, prefix in enumerate(struct):
                assert isinstance(prefix, _KeyPath)
                sourced_prefixes.add(prefix.kp, GetItemSource(src, i))
        elif isinstance(struct, dict):
            for k, prefix in struct.items():
                assert isinstance(prefix, _KeyPath)
                sourced_prefixes.add(prefix.kp, GetItemSource(src, k))

    return sourced_prefixes


def key_path_to_source(
    kp: KeyPath, sourced_prefixes: Optional[_KeyPathTrie] = None
) -> Source:
    """
    Given a key path, return the source for the key path.
    """
    if sourced_prefixes is None:
        source: Source = LocalSource("args")
    else:
        source, kp = sourced_prefixes.get(kp)
    for k in kp:
        if isinstance(k, SequenceKey):
            source = GetItemSource(source, k.idx)
        elif isinstance(k, MappingKey):
            source = GetItemSource(source, k.key)
        elif isinstance(k, GetAttrKey):
            source = AttrSource(source, k.name)
        else:
            raise ValueError(f"Unknown KeyEntry {k}")

    return source


def _is_constant_argument(t):
    return t is None or isinstance(t, (float, bool, str))


def fakify(
    mode: FakeTensorMode,
    kp: KeyPath,
    t: Any,
    t_constraints: dict[int, dict[int, Constraint]],
    sources: dict[tuple[int, int], list[Source]],
    sourced_prefixes: Optional[_KeyPathTrie] = None,
):
    source = key_path_to_source(kp, sourced_prefixes=sourced_prefixes)
    if _is_constant_argument(t) or isinstance(t, (torch.ScriptObject, torch.nn.Module)):
        return t

    if isinstance(t, _IntWrapper):
        if t.dynamism is not None and t.dynamism.type in (_DimHintType.DYNAMIC, _DimHintType.AUTO):  # type: ignore[union-attr]
            symint = mode.shape_env.create_unspecified_symint_and_symbol(  # type: ignore[union-attr]
                t.val, source, DimDynamic.DYNAMIC
            )
            context = (
                SymIntSymbolicContext(
                    constraint=RelaxedUnspecConstraint(warn_only=False)
                )
                if t.dynamism.type == _DimHintType.DYNAMIC  # type: ignore[union-attr]
                else None
            )
            mode.shape_env.tracked_fakes.append(  # type: ignore[union-attr]
                TrackedFake(symint, source, context)
            )
            return symint
        else:
            return t.val

    if not isinstance(t, torch.Tensor):
        raise ValueError(
            f"Unsupported input type {type(t)}. "
            "Export only supports pytree containers of basic types (Tensor, int, float, ...) as input. "
            "To register a custom dataclass, use torch.export.register_dataclass. "
            "To register a custom container type, use torch.utils._pytree.register_pytree_node. "
            "To register a constant input, use torch.utils._pytree.register_constant"
        )

    n_dims = len(t.shape)
    dynamic_sizes = []
    constraint_sizes = [None] * n_dims
    for i in range(n_dims):
        if i in getattr(t, "_dynamo_weak_dynamic_indices", {}):
            dynamic_sizes.append(DimDynamic.DYNAMIC)
        elif i in getattr(t, "_dynamo_dynamic_indices", {}):
            # bit annoying, but we need to replicate process in _dynamo/variables/builder.py
            # where a RelaxedUnspecConstraint is created for Dim.DYNAMIC, so constraint violations
            # are raised when specializing.
            dynamic_sizes.append(DimDynamic.DYNAMIC)
            constraint_sizes[i] = RelaxedUnspecConstraint(warn_only=False)  # type: ignore[call-overload]
        else:
            dynamic_sizes.append(DimDynamic.STATIC)
    symbolic_context: StatelessSymbolicContext = (  # make mypy happy
        StatelessSymbolicContext(
            dynamic_sizes=dynamic_sizes,
            constraint_sizes=constraint_sizes,  # type: ignore[arg-type]
        )
    )
    t_id = id(t)
    assert mode.shape_env is not None
    if t_id in t_constraints:
        for i, constraint in t_constraints[t_id].items():
            src = TensorPropertySource(base=source, prop=TensorProperty.SIZE, idx=i)
            sources[(t_id, i)].append(src)
            if isinstance(constraint, _RelaxedConstraint):
                continue
            symbolic_context.constraint_sizes[i] = constraint.constraint_range
            mode.shape_env.source_name_to_debug_name[src.name()] = constraint.name  # type: ignore[assignment]
    fake = mode.from_tensor(t, source=source, symbolic_context=symbolic_context)
    mode.shape_env.tracked_fakes.append(TrackedFake(fake, source, symbolic_context))  # type: ignore[union-attr]
    return fake


def _is_unbacked_symint(symbol):
    if not isinstance(symbol, torch.SymInt):
        return False

    return symbol.node.shape_env.is_unbacked_symint(symbol.node.expr)


def _tensor_min_max(*args, real_callable, tensor_callable, **kwargs):
    """
    This logic is replicated from dynamo/variables/builtin.py
    """
    if len(args) == 2 and not kwargs:
        arg1, arg2 = args

        # Case 1: Both are tensors
        if isinstance(arg1, torch.Tensor) and isinstance(arg2, torch.Tensor):
            return tensor_callable(arg1, arg2)

        # Case 2: One tensor, one scalar
        elif isinstance(arg1, torch.Tensor) or isinstance(arg2, torch.Tensor):
            if not isinstance(arg1, torch.Tensor):
                arg1, arg2 = arg2, arg1

            if isinstance(arg2, (int, float)):
                kwarg = {"min" if tensor_callable is torch.maximum else "max": arg2}
                return torch.clamp(arg1, **kwarg)  # type: ignore[call-overload]
            else:
                return real_callable(arg1, arg2)

        # Case 3: SymInts
        elif isinstance(arg1, torch.SymInt) or isinstance(arg2, torch.SymInt):
            return (
                torch.sym_max(arg1, arg2)
                if tensor_callable is torch.maximum
                else torch.sym_min(arg1, arg2)
            )

        # Fallback
        else:
            return real_callable(arg1, arg2)

    # Single iterable argument handling
    if len(args) == 1 and not kwargs:
        iterable = args[0]

        if isinstance(iterable, torch.Tensor):
            return tensor_callable(iterable)
        try:
            iterator = iter(iterable)
        except TypeError:
            pass
        else:
            items = list(iterator)
            if not items:
                raise ValueError(f"{real_callable.__name__}() arg is an empty sequence")

            return functools.reduce(
                lambda a, b: _tensor_min_max(
                    a, b, real_callable=real_callable, tensor_callable=tensor_callable
                ),
                items,
            )

    # Fallback to original callable
    return real_callable(*args, **kwargs)


@contextmanager
def _override_builtin_ops():
    original_max = builtins.max
    original_min = builtins.min
    original_pow = math.pow

    builtins.max = functools.partial(
        _tensor_min_max, real_callable=original_max, tensor_callable=torch.maximum
    )

    builtins.min = functools.partial(
        _tensor_min_max, real_callable=original_min, tensor_callable=torch.minimum
    )

    math.pow = lambda x, y: x**y  # type: ignore[operator]

    try:
        yield
    finally:
        builtins.max = original_max
        builtins.min = original_min
        math.pow = original_pow


def make_fake_inputs(
    nn_module,
    args,
    kwargs,
    dynamic_shapes,
    _is_torch_jit_trace=False,
    allow_complex_guards_as_runtime_asserts=False,
):
    """
    Given an nn module, example inputs, and constraints, return a new fake mode,
    fake inputs created in that mode whose dynamic shape dimensions are constrained
    by the given ranges, and sources for pairs of dynamic shape dimensions that are
    constrained to be equal.
    """
    # TODO(avik): refactor Dynamo to avoid duplication of the following code
    # between non-strict and strict.
    # Specifically, here (non-strict) we do the following pre-tracing steps:
    #   - Fakify inputs.
    #   - Process input shape equalities.
    # In strict, these steps are spread across multiple files:
    #   - output_graph.py fakifies inputs.
    #   - [post-tracing] guards.py processes input shape equalities.
    import torch._functorch.config as _config

    # Map ints to a wrapper structure to help us mark it as dynamic, if it is
    # dynamic. We will unwrap ints in fakify later.
    args, kwargs = pytree.tree_map_only(int, lambda a: _IntWrapper(a), (args, kwargs))

    combined_args = _combine_args(nn_module, args, kwargs)
    _check_dynamic_shapes(combined_args, dynamic_shapes)
    constraints = _process_dynamic_shapes(combined_args, dynamic_shapes)
    t_constraints: dict[int, dict[int, Constraint]] = defaultdict(dict)
    for constraint in constraints:
        t_constraints[constraint.t_id][constraint.dim] = constraint

    context = torch._guards.TracingContext.try_get()
    if context is not None:
        # This occurs when we are exporting within dynamo. There already exists
        # a toplevel TracingContext with a fake mode, so we do not want to
        # create another fake mode.
        fake_mode = context.fake_mode
    elif not _is_torch_jit_trace:
        if isinstance(nn_module.forward, functools.partial):
            # functools handles nesting by itself, no need to recurse
            code = nn_module.forward.func.__code__
        else:
            code = nn_module.forward.__code__
        co_fields = {
            "co_name": code.co_name,
            "co_filename": code.co_filename,
            "co_firstlineno": code.co_firstlineno,
        }
        with _config.patch(fake_tensor_allow_unsafe_data_ptr_access=False):
            fake_mode = FakeTensorMode(
                shape_env=ShapeEnv(
                    tracked_fakes=[],
                    co_fields=co_fields,
                    prefer_deferred_runtime_asserts_over_guards=True,
                    allow_complex_guards_as_runtime_asserts=allow_complex_guards_as_runtime_asserts,
                    trace_asserts=True,
                ),
                allow_non_fake_inputs=True,
                export=True,
            )
    else:
        with _config.patch(fake_tensor_allow_unsafe_data_ptr_access=False):
            fake_mode = FakeTensorMode(
                shape_env=ShapeEnv(
                    tracked_fakes=[],
                    prefer_deferred_runtime_asserts_over_guards=True,
                    allow_complex_guards_as_runtime_asserts=allow_complex_guards_as_runtime_asserts,
                    trace_asserts=True,
                ),
                allow_non_fake_inputs=True,
            )
    if fake_mode.shape_env is None or fake_mode.shape_env.tracked_fakes is None:
        raise ValueError(
            "Detected fake_mode does not have a shape_env with tracked fakes. "
            "If you constructed the module under a FakeTensorMode, "
            "please initialize it like: FakeTensorMode(shape_env=ShapeEnv(tracked_fakes=[]))"
        )

    with fake_mode:
        # FIXME(ycao) ScriptMethod doesn't have signature, I am using an empty one to unblock
        if not _is_torch_jit_trace:
            original_signature = inspect.signature(nn_module.forward)
        else:
            original_signature = None
        sources: dict[tuple[int, int], list[Source]] = defaultdict(list)
        sourced_prefixes = make_sourced_prefixes(nn_module, args, kwargs)
        fake_args, fake_kwargs = tree_map_with_path(
            lambda kp, val: fakify(
                fake_mode,
                kp,
                val,
                t_constraints,
                sources,
                sourced_prefixes=sourced_prefixes,
            ),
            (args, kwargs),
        )

        names: dict[str, tuple[int, int]] = {}
        source_pairs: list[tuple[Source, Source]] = []
        derived_equalities: list[tuple[Source, Union[Source, Symbol], Callable]] = []
        phantom_symbols: dict[str, Symbol] = {}
        relaxed_sources: set[Source] = set()
        for constraint in constraints:
            torch.export.dynamic_shapes._process_equalities(
                constraint,
                lambda t_id, dim: sources[(t_id, dim)],
                fake_mode.shape_env,
                names,
                source_pairs,
                derived_equalities,
                phantom_symbols,
                relaxed_sources,
            )

        equalities_inputs = EqualityConstraint(
            source_pairs=source_pairs,
            derived_equalities=derived_equalities,
            phantom_symbols=list(phantom_symbols.values()),
            relaxed_sources=relaxed_sources,
            warn_only=False,
        )
        return (
            fake_mode,
            fake_args,
            fake_kwargs,
            equalities_inputs,
            original_signature,
            dynamic_shapes,
        )


def _flatten_dynamic_shapes(
    combined_args: dict[str, Any],
    dynamic_shapes: Union[dict[str, Any], tuple[Any], list[Any]],
) -> list[Any]:
    flat_shapes = []

    def _tree_map_helper(path, t, shape):
        nonlocal flat_shapes
        flat_shapes.append(shape)

    _tree_map_with_path(_tree_map_helper, combined_args, dynamic_shapes)
    return flat_shapes


def _clean_dynamic_markers(tensor: torch.Tensor) -> None:
    for attr in [
        "_dynamo_weak_dynamic_indices",
        "_dynamo_dynamic_indices",
        "_dynamo_dynamic_range",
        "_dynamo_static_indices",
        "_dynamo_unbacked_indices",
    ]:
        if hasattr(tensor, attr):
            delattr(tensor, attr)


def produce_guards_and_solve_constraints(
    fake_mode: FakeTensorMode,
    gm: torch.fx.GraphModule,
    dynamic_shapes: Union[dict[str, Any], tuple[Any], list[Any], None],
    equalities_inputs: EqualityConstraint,
    original_signature: inspect.Signature,
    _is_torch_jit_trace=False,
):
    """
    Given a fake mode, sources pairs corresponding to equal dynamic shape dimensions,
    and a graph module, produce guards on the fake mode's shape env (raising constraint
    violations if any), solve (to suggest simplifications or fixes).
    Dynamo already performs this, so this is for non-strict mode.

    Additional inputs:
        equalities_inputs: the equality constraints to use for guards
        original_signature: the signature of the forward method
    """
    shape_env = fake_mode.shape_env
    assert shape_env is not None
    assert shape_env.tracked_fakes is not None

    placeholders = [tf.fake for tf in shape_env.tracked_fakes]
    sources = [tf.source for tf in shape_env.tracked_fakes]
    input_contexts = [tf.symbolic_context for tf in shape_env.tracked_fakes]
    constraint_violation_error = None
    try:
        shape_env.produce_guards(
            placeholders,
            sources,
            input_contexts=input_contexts,
            equalities_inputs=equalities_inputs,
            ignore_static=False,
        )
    except ConstraintViolationError as e:
        constraint_violation_error = e

    shape_env.frozen = True
    dim_constraints = shape_env.dim_constraints
    if dim_constraints is None:
        # Expected when shape_env.produce_guards throws an early constraint violation error.
        # There is nothing to solve for in this case.
        # TODO(avik): Maybe record the constraint violation error instead and replay later?
        assert constraint_violation_error
        raise constraint_violation_error
    dim_constraints.solve()
    forced_specializations = dim_constraints.forced_specializations()
    if not _is_torch_jit_trace:
        msg = dim_constraints.prettify_results(
            original_signature,
            dynamic_shapes,  # type: ignore[arg-type]
            constraint_violation_error,
            forced_specializations,  # type: ignore[arg-type]
        )
    else:
        # FIXME(ycao): This is a hack to get around missing signature from ScriptMethod
        msg = "dummy constraint violation message"
    if constraint_violation_error:
        constraint_violation_error.args = (constraint_violation_error.args[0] + msg,)
    elif forced_specializations:
        constraint_violation_error = ConstraintViolationError(msg)
    if constraint_violation_error:
        raise constraint_violation_error


def is_int(x: object) -> bool:
    return isinstance(x, int) or (isinstance(x, torch.SymInt) and x.node.expr.is_number)


def _constrain_user_specified_dimhint_range(
    symint: torch.SymInt,
    hint: int,
    dim: _DimHint,
    range_constraints,
    shape_env,
    keypath: KeyPath,
    i: Optional[int] = None,
) -> Optional[str]:
    trace_vr = (
        range_constraints[symint.node.expr]
        if not is_int(symint)
        else ValueRanges(int(symint), int(symint))
    )

    # warn on 0/1 specialization for Dim.AUTO; not an actual error
    if dim.type == _DimHintType.AUTO and trace_vr.is_singleton() and hint in (0, 1):
        pathstr = f"inputs{pytree.keystr(keypath)}"
        if i is not None:
            pathstr += f".shape[{i}]"
        msg = (
            f"dimension {pathstr} 0/1 specialized; Dim.AUTO was specified along "
            + f"with a sample input with hint = {hint}."
        )
        log.warning(msg)

    try:
        user_vr = ValueRanges(
            lower=0 if dim.min is None else dim.min,
            upper=int_oo if dim.max is None else dim.max,
        )
        if is_int(symint):
            out_vr = trace_vr & user_vr
        else:
            range_constraints[symint.node.expr] &= user_vr
            shape_env.var_to_range[symint.node._expr] &= user_vr
            out_vr = range_constraints[symint.node.expr]

        # check for Dim.DYNAMIC specializations; special case error message on 0/1
        if dim.type == _DimHintType.DYNAMIC and out_vr.is_singleton():
            path = f"inputs{pytree.keystr(keypath)}"
            if i is not None:
                path += f".shape[{i}]"
            if (
                trace_vr.is_singleton()
                and hint in (0, 1)
                and not torch.fx.experimental._config.backed_size_oblivious
            ):
                msg = (
                    f"- Received user-specified dim hint Dim.DYNAMIC(min={dim.min}, max={dim.max}), "
                    f"but export 0/1 specialized due to hint of {hint} for dimension {path}."
                )
            else:
                msg = (
                    f"- Received user-specified dim hint Dim.DYNAMIC(min={dim.min}, max={dim.max}), "
                    f"but tracing inferred a static shape of {out_vr.lower} for dimension {path}."
                )
            return msg

    except torch.utils._sympy.value_ranges.ValueRangeError:
        path = f"inputs{pytree.keystr(keypath)}"
        if i is not None:
            path += f".shape[{i}]"
        msg = (
            f"- Received user-specified min/max range of [{dim.min}, {dim.max}], "
            f"conflicting with the inferred min/max range of [{trace_vr.lower}, {trace_vr.upper}], "
            f"for {path}."
        )
        return msg

    return None


def make_constraints(
    fake_mode: FakeTensorMode,
    gm: torch.fx.GraphModule,
    combined_args: dict[str, Any],
    dynamic_shapes: Union[dict[str, Any], tuple[Any], list[Any], None],
    num_lifted_inputs: int,
):
    """
    Given a fake mode's shape env and user-specified dynamic shapes,
    return the resulting range constraints and equality constraints.

    Additional args:
        num_lifted_inputs: the number of non-user-input placeholder nodes in the graph
        (used only to enumerate the user-input nodes)
    """

    shape_env = fake_mode.shape_env
    assert shape_env is not None
    inline_constraints = gm.meta.get("inline_constraints", [])
    range_constraints = defaultdict(lambda: ValueRanges(0, int_oo)) | inline_constraints
    if not dynamic_shapes:
        return dict(range_constraints)

    # clean up dynamic markers from tensors
    flat_paths, flat_args = zip(*pytree.tree_flatten_with_path(combined_args)[0])
    for arg in flat_args:
        if isinstance(arg, torch.Tensor):
            _clean_dynamic_markers(arg)

    # get individual dynamic shapes spec for each input
    if not isinstance(dynamic_shapes, dict):
        assert isinstance(dynamic_shapes, (tuple, list))
        combined_args = type(dynamic_shapes)(combined_args.values())  # type: ignore[assignment, misc]
    flat_dynamic_shapes = _flatten_dynamic_shapes(combined_args, dynamic_shapes)

    # check number of shapes vs. number of inputs
    num_placeholders = [node.op == "placeholder" for node in gm.graph.nodes].count(True)
    assert len(flat_dynamic_shapes) == num_placeholders - num_lifted_inputs

    free_symbols = set()
    range_violations = []
    for input_index, node in enumerate(gm.graph.nodes):
        meta_val = node.meta.get("val")

        if (
            input_index < num_lifted_inputs
            or node.op != "placeholder"
            or meta_val is None
        ):
            continue

        elif _is_constant_argument(meta_val) or isinstance(meta_val, CustomObjArgument):
            continue

        shape_spec = flat_dynamic_shapes[input_index - num_lifted_inputs]
        keypath = flat_paths[input_index - num_lifted_inputs]
        flat_arg = flat_args[input_index - num_lifted_inputs]

        if isinstance(meta_val, int) or (
            isinstance(meta_val, torch.SymInt) and meta_val.node.expr.is_number
        ):
            pass

        elif isinstance(meta_val, torch.SymInt):
            if shape_spec is not None and isinstance(shape_spec, _DimHint):
                hint = flat_arg
                range_constraints[meta_val.node.expr] &= shape_env.bound_sympy(
                    meta_val.node._expr
                )
                violation = _constrain_user_specified_dimhint_range(
                    meta_val,
                    hint,
                    shape_spec,
                    range_constraints,
                    shape_env,
                    keypath,
                    None,
                )
                if violation:
                    range_violations.append(violation)
            else:
                raise RuntimeError("nyi")
            free_symbols.update(meta_val.node.expr.free_symbols)

        elif isinstance(meta_val, torch.Tensor):
            for i, d in enumerate(node.meta["val"].shape):
                dim = None
                if isinstance(shape_spec, (list, tuple)):
                    dim = shape_spec[i]
                elif isinstance(shape_spec, dict):
                    dim = shape_spec.get(i)
                if not is_int(d):
                    # Compute the range constraint for the symbolic expression corresponding
                    # to this shape dimension and store it.
                    if dim is None or isinstance(dim, _DimHint):
                        range_constraints[d.node.expr] &= shape_env.bound_sympy(
                            d.node.expr
                        )
                    else:
                        range_constraints[d.node.expr] &= ValueRanges(
                            lower=dim.min, upper=dim.max
                        )

                    free_symbols.update(d.node.expr.free_symbols)

                # check user-specified min/max range for DimHints;
                # we might want to do this even if model tracing inferred a static dimension.
                if isinstance(dim, _DimHint):
                    hint = flat_arg.shape[i]
                    violation = _constrain_user_specified_dimhint_range(
                        d, hint, dim, range_constraints, shape_env, keypath, i
                    )
                    if violation:
                        range_violations.append(violation)
        else:
            raise RuntimeError(f"Unfamiliar meta val: {meta_val}")

    if range_violations:
        prefix = "Found the following conflicts between user-specified ranges and inferred ranges from model tracing:\n"
        raise ValueError(prefix + "\n".join(range_violations))

    for symbol in free_symbols:
        if symbol not in range_constraints:
            # Placeholders can have symbolic shapes that are derived expressions.
            # The above code will record direct range constraints for them
            # so that we can do runtime assertions. In addition, for serde checks
            # we want to record range constraints for their root symbols.
            range_constraints[symbol] = shape_env.var_to_range[symbol]

    return dict(range_constraints)


def _gather_constant_attrs(m: torch.nn.Module) -> ConstantAttrMap:
    """Search the module hierarchy, gathering up all tensor and ScriptObject constants.

    Returns a dictionary mapping hash(value) to the name of the constant. We
    have to abuse `hash` here unfortunately, see: [ScriptObject hash].
    """
    constants = ConstantAttrMap()
    buffers_parameters = set(m.buffers())
    buffers_parameters.update(m.parameters())

    def inner(m: torch.nn.Module, prefix_atoms: list[str], constants):
        for k, v in m.__dict__.items():
            if isinstance(
                v,
                (
                    torch.Tensor,
                    torch.ScriptObject,
                    FakeScriptObject,
                ),
            ):
                if v in buffers_parameters:
                    # filter out buffers and parameters, leaving only constants
                    continue

                fqn = ".".join(prefix_atoms + [k])
                constants.add(v, fqn)
        for k, v in m.named_children():
            inner(v, prefix_atoms + [k], constants)

    inner(m, [], constants)
    return constants


def _get_graph_inputs_of_type_nn_module(
    args: Optional[tuple[tuple[Any], dict[Any, Any]]],
) -> set[type[torch.nn.Module]]:
    if args is None:
        return set()
    module_types = set()
    for arg in pytree.tree_leaves(args):
        if isinstance(arg, torch.nn.Module):
            module_types.add(type(arg))
    return module_types


def _enter_enable_graph_inputs_of_type_nn_module(
    module_types: set[type[torch.nn.Module]],
) -> None:
    for t in module_types:
        torch._export.utils.register_module_as_pytree_input_node(t)


def _exit_enable_graph_inputs_of_type_nn_module(
    module_types: set[type[torch.nn.Module]],
) -> None:
    for t in module_types:
        torch._export.utils.deregister_module_as_pytree_input_node(t)


@contextlib.contextmanager
def _enable_graph_inputs_of_type_nn_module(
    args: Optional[tuple[tuple[Any], dict[Any, Any]]],
):
    if args is None:
        yield
        return

    module_types = _get_graph_inputs_of_type_nn_module(args)
    _enter_enable_graph_inputs_of_type_nn_module(module_types)
    try:
        yield
    finally:
        _exit_enable_graph_inputs_of_type_nn_module(module_types)


@contextlib.contextmanager
def _fakify_module_inputs(
    args: tuple[Any],
    kwargs: dict[Any, Any],
    fake_mode: torch._subclasses.fake_tensor.FakeTensorMode,
):
    # This context manager is used to fakify module inputs.
    # Inputs:
    #   args, kwargs: the args and kwargs containing module inputs that haven't been fakified.
    #   fake_mode: the fake mode to be used for fakifying script objects. It's the same mode that fakify input tensors.

    ctxs = [_enable_graph_inputs_of_type_nn_module((args, kwargs))]
    for arg in pytree.tree_leaves((args, kwargs)):
        if isinstance(arg, torch.nn.Module):
            fake_params_buffers = _fakify_params_buffers(fake_mode, arg)
            ctxs.append(
                torch.nn.utils.stateless._reparametrize_module(
                    arg,
                    fake_params_buffers,
                    tie_weights=True,
                    strict=True,
                    stack_weights=True,
                )
            )
    with contextlib.ExitStack() as stack:
        for ctx in ctxs:
            stack.enter_context(ctx)
        yield


@contextlib.contextmanager
def _fakify_script_objects(
    mod: torch.nn.Module,
    args: Sequence[Any],
    kwargs: dict[Any, Any],
    fake_mode: torch._subclasses.fake_tensor.FakeTensorMode,
):
    # This context manager is used to fakify script objects into FakeScriptObject.
    # Inputs:
    #   mod: the module to be exported, it (and its recursive submodules)'s script object attrs haven't been fakified.
    #   args, kwargs: the args and kwargs inputs for mod, script object inputs haven't been fakified.
    #   fake_mode: the fake mode to be used for fakifying script objects. It's the same mode that fakify input tensors.
    #
    # Returns:
    #   mod: the patched module, its (and its recursive submodules) script object attrs have been fakified.
    #   fake_args, fake_kwargs: new fakified args and kwargs.
    #        Script object inputs have been fakified. Don't touch the tensors.
    #   fake_constant_attrs: a new map from FakeScriptObject to the fqn of the original script object.
    #   fake_to_real: a mapping between FakeScriptObject and the original script object in order to un-do the patching.

    constant_attrs: ConstantAttrMap = _gather_constant_attrs(mod)
    assert not any(
        isinstance(obj, FakeScriptObject) for obj in constant_attrs.values()
    ), "Mod shouldn't contain any FakeScriptObject."
    assert not pytree.tree_any(
        lambda obj: isinstance(obj, FakeScriptObject), (args, kwargs)
    ), "args and kwargs shouldn't contain any FakeScriptObject."

    patched_attr = {}
    fake_constant_attrs = ConstantAttrMap()
    fake_to_real = {}

    def _maybe_fakify_obj(obj):
        fake_obj = torch._library.fake_class_registry.maybe_to_fake_obj(fake_mode, obj)
        fake_to_real[fake_obj] = obj
        return fake_obj

    def _leaf_mod_and_attr(
        mod: torch.nn.Module, attr_fqn: str
    ) -> tuple[torch.nn.Module, str]:
        *prefix_attr, last_attr = attr_fqn.split(".")
        cur_mod = mod
        for attr in prefix_attr:
            cur_mod = getattr(cur_mod, attr)
        return cur_mod, last_attr

    try:
        for obj, fqns in constant_attrs.items():
            if torch._library.fake_class_registry._is_script_object(obj):
                fake_script_obj = _maybe_fakify_obj(obj)
                for fqn in fqns:
                    cur_mod, attr = _leaf_mod_and_attr(mod, fqn)
                    assert obj is getattr(cur_mod, attr)
                    setattr(cur_mod, attr, fake_script_obj)
                    fake_constant_attrs.add(fake_script_obj, fqn)
                    patched_attr[fqn] = obj
            else:
                for fqn in fqns:
                    fake_constant_attrs.add(obj, fqn)

        fake_args, fake_kwargs = pytree.tree_map_only(
            torch.ScriptObject, _maybe_fakify_obj, (args, kwargs)
        )
        yield (mod, fake_args, fake_kwargs, fake_constant_attrs, fake_to_real)
    finally:
        for fqn, orig_obj in patched_attr.items():
            cur_mod, attr = _leaf_mod_and_attr(mod, fqn)
            setattr(cur_mod, attr, orig_obj)


class _NonStrictTorchFunctionHandler(torch.overrides.TorchFunctionMode):
    """
    1. Handles data-dependent errors raised by torch function calls in non-strict.

    Any data-dependent error is due to some condition on unbacked symints
    that cannot be resolved. A mechanical way of fixing the error is to use
    a torch._check() call to assert either that condition or its negation.
    The handler suggests these options as code and points to the location
    of the torch function call that raised the error as part of the error
    message shown to the user, who can then simply select and copy-paste
    a suggested fix at that location.

    NOTE: Not all data-dependent errors are raised by torch function calls.
    In particular, conditions on unbacked symints can appear outside such
    calls, and as such are not handled here.

    2. Overrides torch functions that are known to cause problems in non-strict.

    Certain Python features, such as indexing/slicing, cannot be intercepted
    in non-strict. Likewise, certain legacy ops, such as distributed collectives,
    may need to be mapped to other ops. When there is special handling in Dynamo
    for such things, tracing can fail in non-strict (while succeeding in strict).
    Fortunately, redirecting to other torch functions can often fix such issues.

    3. Handles line-of-code logging for each torch function call in non-strict.

    Usage: TORCHEXPORT_EXTENDED_DEBUG_CURRENT_LOC=1 TORCH_LOGS="+export" ...
    """

    def _override(self, func, args, kwargs):
        if torch.distributed.is_available():
            from torch.distributed._functional_collectives import (
                REDUCE_OP_TO_STR,
                traceable_collective_remaps,
            )

            if func in traceable_collective_remaps:
                # Redirect to a corresponding functional collective, following Dynamo.
                # See torch/distributed/_functional_collectives.py for details.
                # The following is an adaptation of CollectiveFunctionRewriteVariable.
                mapped_func = traceable_collective_remaps[func]
                signature = inspect.signature(func)
                kwargs = dict(signature.bind(*args, **kwargs).arguments)
                args = ()
                if func in (
                    torch.distributed.all_reduce,
                    torch.distributed.reduce_scatter_tensor,
                    torch.distributed._reduce_scatter_base,
                ):
                    if "op" in kwargs:
                        kwargs["op"] = REDUCE_OP_TO_STR[kwargs["op"]]
                return mapped_func, args, kwargs
        if func is torch.tensor:
            # Redirect to Python implementation of torch.tensor for data with symints.
            # NOTE(avik): We don't unconditionally redirect to this implementation
            # because it has some known incompletenesses, e.g., it doesn't support
            # empty data. See https://github.com/pytorch/pytorch/issues/143216
            if any(
                isinstance(a, (torch.SymInt, torch.SymFloat, torch.SymBool))
                for a in pytree.tree_flatten(args[0])[0]
            ):
                return torch._refs.tensor, args, kwargs
        if func.__name__ == "__getitem__" and isinstance(args[0], torch.Tensor):

            def rewrite(dim, item):
                # Redirect to torch.select for indexing.
                if isinstance(item, (int, torch.SymInt)):
                    return dim, (torch.select, [dim, item])
                # Redirect to torch.ops.aten.slice for slicing.
                if isinstance(item, slice):
                    return dim + 1, (
                        torch.ops.aten.slice,
                        [dim, item.start, item.stop, item.step or 1],
                    )
                # Otherwise do nothing.

            items = args[1] if isinstance(args[1], tuple) else (args[1],)
            dim = 0
            # Sequence rewrites.
            sequence = []
            for item in items:
                if (r := rewrite(dim, item)) is None:
                    return func, args, kwargs
                dim, call_spec = r
                sequence.append(call_spec)

            def run():
                # Run sequence.
                t = args[0]
                for _method, _args in sequence:
                    t = _method(t, *_args)
                return t

            return run, [], {}

        return func, args, kwargs

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if torch.compiler.is_dynamo_compiling():
            return func(*args, **kwargs)

        if log.isEnabledFor(logging.DEBUG) and config.extended_debug_current_loc:
            frame = _find_user_code_frame()
            if frame is not None:
                log.debug(
                    "%s called at %s:%s in %s",
                    func.__qualname__,
                    frame.f_code.co_filename,
                    frame.f_lineno,
                    frame.f_code.co_name,
                )

        func, args, kwargs = self._override(func, args, kwargs)
        try:
            return func(*args, **kwargs)
        except GuardOnDataDependentSymNode as e:
            _suggest_fixes_for_data_dependent_error_non_strict(e)
            raise
