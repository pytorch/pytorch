# mypy: allow-untyped-defs
import contextlib
import inspect
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple, TYPE_CHECKING, Union

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
from torch._export.passes.add_runtime_assertions_for_constraints_pass import InputDim
from torch._export.passes.lift_constants_pass import ConstantAttrMap
from torch._guards import Source
from torch._library.fake_class_registry import FakeScriptObject
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.export import Constraint
from torch.export.dynamic_shapes import (
    _check_dynamic_shapes,
    _combine_args,
    _DimHint,
    _process_dynamic_shapes,
    _transform_shapes_for_default_dynamic,
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
    ShapeEnv,
    StatelessSymbolicContext,
    ValueRanges,
)
from torch.utils._pytree import (
    GetAttrKey,
    KeyPath,
    MappingKey,
    SequenceKey,
    tree_map_with_path,
)


if TYPE_CHECKING:
    from sympy import Symbol


log = logging.getLogger(__name__)


def key_path_to_source(kp: KeyPath) -> Source:
    """
    Given a key path, return the source for the key path.
    """
    source: Source = LocalSource("args")
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
    return t is None or isinstance(t, (int, float, bool, str))


def fakify(
    mode: FakeTensorMode,
    kp: KeyPath,
    t: Any,
    t_constraints: Dict[int, Dict[int, Constraint]],
    sources: Dict[Tuple[int, int], List[Source]],
):
    source = key_path_to_source(kp)
    if _is_constant_argument(t) or isinstance(t, torch.ScriptObject):
        return t

    if not isinstance(t, torch.Tensor):
        raise ValueError(f"Unsupported input type {type(t)}")
    n_dims = len(t.shape)
    symbolic_context = StatelessSymbolicContext(
        dynamic_sizes=[DimDynamic.DYNAMIC] * n_dims,
        constraint_sizes=[None] * n_dims,
    )
    t_id = id(t)
    assert mode.shape_env is not None
    if t_id in t_constraints:
        for i, constraint in t_constraints[t_id].items():
            symbolic_context.constraint_sizes[i] = constraint.constraint_range
            src = TensorPropertySource(base=source, prop=TensorProperty.SIZE, idx=i)
            sources[(t_id, i)].append(src)
            mode.shape_env.source_name_to_debug_name[src.name()] = constraint.name  # type: ignore[assignment]
    fake = mode.from_tensor(t, source=source, symbolic_context=symbolic_context)
    mode.shape_env.tracked_fakes.append(TrackedFake(fake, source, symbolic_context))  # type: ignore[union-attr]
    return fake


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

    combined_args = _combine_args(nn_module, args, kwargs)
    _check_dynamic_shapes(combined_args, dynamic_shapes)
    transformed_dynamic_shapes = _transform_shapes_for_default_dynamic(
        combined_args, dynamic_shapes
    )
    constraints = _process_dynamic_shapes(combined_args, transformed_dynamic_shapes)
    t_constraints: Dict[int, Dict[int, Constraint]] = defaultdict(dict)
    for constraint in constraints:
        t_constraints[constraint.t_id][constraint.dim] = constraint

    context = torch._guards.TracingContext.try_get()
    if context is not None:
        # This occurs when we are exporting within dynamo. There already exists
        # a toplevel TracingContext with a fake mode, so we do not want to
        # create another fake mode.
        fake_mode = context.fake_mode
    elif not _is_torch_jit_trace:
        code = nn_module.forward.__code__
        co_fields = {
            "co_name": code.co_name,
            "co_filename": code.co_filename,
            "co_firstlineno": code.co_firstlineno,
        }
        fake_mode = FakeTensorMode(
            shape_env=ShapeEnv(
                tracked_fakes=[],
                co_fields=co_fields,
                prefer_deferred_runtime_asserts_over_guards=True,
                allow_complex_guards_as_runtime_asserts=allow_complex_guards_as_runtime_asserts,
            ),
            allow_non_fake_inputs=True,
            export=True,
        )
    else:
        fake_mode = FakeTensorMode(
            shape_env=ShapeEnv(
                tracked_fakes=[],
                prefer_deferred_runtime_asserts_over_guards=True,
                allow_complex_guards_as_runtime_asserts=allow_complex_guards_as_runtime_asserts,
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
        sources: Dict[Tuple[int, int], List[Source]] = defaultdict(list)
        fake_args, fake_kwargs = tree_map_with_path(
            lambda kp, val: fakify(fake_mode, kp, val, t_constraints, sources),
            (args, kwargs),
        )

        names: Dict[str, Tuple[int, int]] = {}
        source_pairs: List[Tuple[Source, Source]] = []
        derived_equalities: List[Tuple[Source, Union[Source, Symbol], Callable]] = []
        phantom_symbols: Dict[str, Symbol] = {}
        for constraint in constraints:
            torch.export.dynamic_shapes._process_equalities(
                constraint,
                lambda t_id, dim: sources[(t_id, dim)],
                fake_mode.shape_env,
                names,
                source_pairs,
                derived_equalities,
                phantom_symbols,
            )

        equalities_inputs = EqualityConstraint(
            source_pairs=source_pairs,
            derived_equalities=derived_equalities,
            phantom_symbols=list(phantom_symbols.values()),
            warn_only=False,
        )
        return (
            fake_mode,
            fake_args,
            fake_kwargs,
            equalities_inputs,
            original_signature,
            transformed_dynamic_shapes,
        )


def _flatten_dynamic_shapes(
    combined_args: Dict[str, Any],
    dynamic_shapes: Union[Dict[str, Any], Tuple[Any], List[Any]],
) -> List[Any]:
    flat_shapes = []

    def _tree_map_helper(path, t, shape):
        nonlocal flat_shapes
        flat_shapes.append(shape)

    _tree_map_with_path(_tree_map_helper, combined_args, dynamic_shapes)
    return flat_shapes


def produce_guards_and_solve_constraints(
    fake_mode: FakeTensorMode,
    gm: torch.fx.GraphModule,
    dynamic_shapes: Union[Dict[str, Any], Tuple[Any], List[Any], None],
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
            dynamic_shapes,
            constraint_violation_error,
            forced_specializations,
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


def make_constraints(
    fake_mode: FakeTensorMode,
    gm: torch.fx.GraphModule,
    combined_args: Dict[str, Any],
    dynamic_shapes: Union[Dict[str, Any], Tuple[Any], List[Any], None],
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
    range_constraints = {
        symbol: inline_constraints[symbol] for symbol in inline_constraints
    }
    if not dynamic_shapes:
        return range_constraints

    # get individual dynamic shapes spec for each input
    if not isinstance(dynamic_shapes, dict):
        assert isinstance(dynamic_shapes, (tuple, list))
        combined_args = type(dynamic_shapes)(combined_args.values())  # type: ignore[assignment, misc]
    flat_dynamic_shapes = _flatten_dynamic_shapes(combined_args, dynamic_shapes)

    # check number of shapes vs. number of inputs
    num_placeholders = [node.op == "placeholder" for node in gm.graph.nodes].count(True)
    assert len(flat_dynamic_shapes) == num_placeholders - num_lifted_inputs

    input_dims = defaultdict(list)
    free_symbols = set()
    for input_index, node in enumerate(gm.graph.nodes):
        if input_index < num_lifted_inputs or node.op != "placeholder":
            continue
        if _is_constant_argument(node.meta["val"]) or isinstance(
            node.meta["val"], CustomObjArgument
        ):
            continue
        shape_spec = flat_dynamic_shapes[input_index - num_lifted_inputs]
        for i, d in enumerate(node.meta["val"].shape):
            if isinstance(d, torch.SymInt) and not d.node.expr.is_number:
                # Look up the range constraint for the symbol corresponding to this shape dimension
                # and store it indexed by the symbolic expression corresponding to it.
                # NOTE(avik): Use node._expr instead of node.expr for the lookup here because
                # we want the symbol, not its replacement, which could be an expression. Maybe
                # there's a better way to do this, e.g., by (re)computing value ranges for expressions?
                dim = shape_spec[i] if shape_spec else None
                if dim is None or isinstance(dim, _DimHint):
                    range_constraints[d.node.expr] = shape_env.var_to_range[
                        d.node._expr
                    ]
                else:
                    range_constraints[d.node.expr] = ValueRanges(
                        lower=dim.min, upper=dim.max
                    )
                input_dims[d.node.expr].append(InputDim(input_name=node.name, dim=i))
                free_symbols.update(d.node.expr.free_symbols)

    for symbol in free_symbols:
        if symbol not in range_constraints:
            # Placeholders can have symbolic shapes that are derived expressions.
            # The above code will record direct range constraints for them
            # so that we can do runtime assertions. In addition, for serde checks
            # we want to record range constraints for their root symbols.
            range_constraints[symbol] = shape_env.var_to_range[symbol]

    return range_constraints


def _gather_constant_attrs(m: torch.nn.Module) -> ConstantAttrMap:
    """Search the module hierarchy, gathering up all tensor and ScriptObject constants.

    Returns a dictionary mapping hash(value) to the name of the constant. We
    have to abuse `hash` here unfortunately, see: [ScriptObject hash].
    """
    constants = ConstantAttrMap()
    buffers_parameters = set(m.buffers())
    buffers_parameters.update(m.parameters())

    def inner(m: torch.nn.Module, prefix_atoms: List[str], constants):
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


@contextlib.contextmanager
def _fakify_script_objects(
    mod: torch.nn.Module,
    args: Tuple[Any],
    kwargs: Dict[Any, Any],
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
    ) -> Tuple[torch.nn.Module, str]:
        *prefix_attr, last_attr = attr_fqn.split(".")
        cur_mod = mod
        for attr in prefix_attr:
            cur_mod = getattr(cur_mod, attr)
        return cur_mod, last_attr

    try:
        for obj, fqns in constant_attrs.items():
            if isinstance(obj, torch.ScriptObject):
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

    2. Handles line-of-code logging for each torch function call in non-strict.

    Usage: TORCHEXPORT_EXTENDED_DEBUG_CURRENT_LOC=1 TORCH_LOGS="+export" ...
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
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
        try:
            return func(*args, **kwargs)
        except GuardOnDataDependentSymNode as e:
            _suggest_fixes_for_data_dependent_error_non_strict(e)
            raise
