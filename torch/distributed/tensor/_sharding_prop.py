# mypy: allow-untyped-defs
import threading
from collections.abc import Callable, Sequence
from functools import lru_cache
from itertools import chain, product
from typing import cast, Optional, Union

import torch
from torch._ops import OpOverload
from torch._subclasses import FakeTensorMode
from torch.distributed._functional_collectives import _are_we_tracing
from torch.distributed.tensor._collective_utils import (
    redistribute_cost as _redistribute_cost,
)
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    OpInfo,
    OpSchema,
    OpSpec,
    OpStrategy,
    OutputSharding,
    OutputSpecType,
    RuntimeSchemaInfo,
    StrategyType,
    TupleStrategy,
)
from torch.distributed.tensor._utils import (
    compute_local_shape_and_global_offset,
    compute_local_stride,
)
from torch.distributed.tensor.placement_types import Replicate
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_unflatten


aten = torch.ops.aten


def _length(obj) -> int:
    if obj is None:
        return 0
    if not isinstance(obj, Sequence):
        return 1
    return len(obj)


class LocalLRUCache(threading.local):
    def __init__(self, user_function: Callable) -> None:
        self.cache = lru_cache(None)(user_function)

    def __call__(self, *args, **kwargs) -> object:
        return self.cache(*args, **kwargs)

    def cache_info(self):
        return self.cache.cache_info()


def flatten_inputs(args_schema, kwargs_schema):
    flat_args, args_spec = tree_flatten(args_schema)
    flat_kwargs, kwargs_spec = tree_flatten(kwargs_schema)
    flat_all = flat_args + flat_kwargs
    n_args = len(flat_args)
    dt_indices = [i for i, v in enumerate(flat_all) if isinstance(v, DTensorSpec)]
    return flat_all, args_spec, kwargs_spec, n_args, dt_indices


def unflatten_inputs(flat_all, args_spec, kwargs_spec, n_args):
    new_args = tree_unflatten(flat_all[:n_args], args_spec)
    new_kwargs = tree_unflatten(flat_all[n_args:], kwargs_spec)
    return new_args, new_kwargs


def candidate_lists_per_dt_leaf(
    flat_all, suggested_flat_all, dt_indices, make_replicate_like
):
    """
    for each DTensorSpec leaf position, build a candidate list:
      [original, suggested(if different), replicate-all].
    non-DTensor leaves are implicitly fixed to their original.
    returns a list of lists aligned to dt_indices.
    """
    cands = []
    for idx in dt_indices:
        orig = flat_all[idx]
        assert isinstance(orig, DTensorSpec)
        leaves = [orig.shallow_copy_with_tensor_meta(orig.tensor_meta)]

        # suggested (if provided and different)
        if suggested_flat_all is not None:
            sugg = suggested_flat_all[idx]
            if isinstance(sugg, DTensorSpec) and sugg.placements != orig.placements:
                leaves.append(sugg.shallow_copy_with_tensor_meta(orig.tensor_meta))

        # replicate-all
        repl = make_replicate_like(orig)
        if repl.placements != orig.placements:
            leaves.append(repl.shallow_copy_with_tensor_meta(orig.tensor_meta))

        # dedup by placement
        seen, uniq = set(), []
        for s in leaves:
            key = s.placements
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        cands.append(uniq)
    return cands


def apply_dt_choices(flat_all, dt_indices, choice_tuple):
    """
    build a new flat list from flat_all, replacing DTensorSpec leaves
    at dt_indices with specs from choice_tuple (same length).
    """
    out = list(flat_all)
    for pos, spec in zip(dt_indices, choice_tuple):
        out[pos] = spec
    return out


def total_cost_flat(flat_old, flat_new, dt_indices) -> float:
    total = 0.0
    for i in dt_indices:
        o, n = flat_old[i], flat_new[i]
        if isinstance(o, DTensorSpec) and isinstance(n, DTensorSpec):
            total += float(_redistribute_cost(o, n))
    return total


def shard_score_flat(flat_all, dt_indices) -> int:
    # higher = more shards (we prefer retaining sharding on ties)
    score = 0
    for i in dt_indices:
        s = flat_all[i]
        if isinstance(s, DTensorSpec):
            try:
                if s.is_sharded():
                    score += 1
            except AttributeError:
                if any(
                    getattr(p, "is_shard", lambda **_: False)() for p in s.placements
                ):
                    score += 1
    return score


def sig_flat(flat_all, dt_indices):
    # signature of placements for dedup
    sig = []
    for i in dt_indices:
        s = flat_all[i]
        if isinstance(s, DTensorSpec):
            sig.append(tuple(type(p).__name__ for p in s.placements))
        else:
            sig.append(None)
    return tuple(sig)


def unique_shard_dim_across_inputs(args_schema, kwargs_schema):
    """
    return the unique shard dim if exactly one shard dim is present across
    all DTensorSpec inputs
    """
    dims = set()

    def visit(x):
        if isinstance(x, DTensorSpec):
            for p in x.placements:
                if getattr(p, "is_shard", lambda **_: False)():
                    d = getattr(p, "dim", None)
                    if d is not None:
                        dims.add(d)
        elif isinstance(x, (list, tuple)):
            for y in x:
                visit(y)
        elif isinstance(x, dict):
            for y in x.values():
                visit(y)

    visit(args_schema)
    visit(kwargs_schema)

    if len(dims) == 1:
        return next(iter(dims))
    return None


def align_all_inputs_to_dim(args_schema, kwargs_schema, dim):
    """
    return (args_aligned, kwargs_aligned) where each DTensorSpec leaf is sharded on `dim`.
    only placements are changed.
    """
    from torch.distributed.tensor.placement_types import Shard as ShardPlacement

    def align(x):
        if isinstance(x, DTensorSpec):
            return DTensorSpec(
                mesh=x.mesh,
                placements=tuple(ShardPlacement(dim) for _ in x.placements),
                tensor_meta=x.tensor_meta,
            )
        if isinstance(x, list):
            return [align(y) for y in x]
        if isinstance(x, tuple):
            return tuple(align(y) for y in x)
        if isinstance(x, dict):
            return {k: align(v) for k, v in x.items()}
        return x

    return align(args_schema), align(kwargs_schema)


def _is_prim_op(op: OpOverload) -> bool:
    try:
        # e.g. "prim::device", "prim::dtype", ...
        return op._schema.name.startswith("prim::")
    except Exception:
        return False


def _iter_tensors(x):
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from _iter_tensors(y)
    elif isinstance(x, dict):
        for y in x.values():
            yield from _iter_tensors(y)


class _RedistributeNeeded(Exception):
    """raised by _SpecMode when fail_on_redistribute is enabled and an
    inner op returns needs_redistribute=True."""


class _SpecMode(TorchDispatchMode):
    """
    intercepts aten ops during decomposition executed on FakeTensors,
    maintaining a mapping FakeTensor(id) -> DTensorSpec (with TensorMeta).
    each op call is converted to an OpSchema of DTensorSpecs, routed through
    propagate_op_sharding, and returns fresh FakeTensors registered
    with the output DTensorSpecs so downstream ops see the right specs.

    - fail_on_redistribute=True: if any inner op requests a redistribute,
      immediately stop propagation by raising _RedistributeNeeded.
    - otherwise, drives redistribution suggestions to a fixed-point inside the
      decomposition by rebinding FakeTensor -> DTensorSpec placements whenever
      an inner op requests a redistribute, then re-running propagation.
    """

    def __init__(
        self,
        propagator: "ShardingPropagator",
        *,
        fail_on_redistribute: bool = False,
    ):
        super().__init__()
        self.propagator = propagator
        self._tensor_id_to_spec: dict[int, DTensorSpec] = {}
        self._fail_on_redistribute = fail_on_redistribute

    def register_tensor(self, t: torch.Tensor, spec: DTensorSpec) -> None:
        self._tensor_id_to_spec[id(t)] = spec

    def has_spec(self, t: torch.Tensor) -> bool:
        return id(t) in self._tensor_id_to_spec

    def _ensure_all_tensor_args_bound(self, op: OpOverload, args, kwargs) -> None:
        """
        for this op call, bind specs ONLY for unbound tensor args
        that have an already-bound tensor of the EXACT SAME SHAPE in this call.
        fail if any unbound tensor lacks a same-shape peer
        """

        # build shape -> reference spec map from already-bound peers
        shape_to_spec: dict[tuple[int, ...], DTensorSpec] = {}
        for t in _iter_tensors((args, kwargs)):
            if isinstance(t, torch.Tensor):
                spec = self._tensor_id_to_spec.get(id(t))
                if isinstance(spec, DTensorSpec):
                    shape_to_spec.setdefault(tuple(t.shape), spec)

        if not shape_to_spec:
            return

        # bind unbound tensors iff exact shape matches
        misses: list[tuple[int, ...]] = []
        for t in _iter_tensors((args, kwargs)):
            if isinstance(t, torch.Tensor) and not self.has_spec(t):
                shape = tuple(t.shape)
                spec = shape_to_spec.get(shape)

                if spec is None:
                    misses.append(shape)
                    continue

                self.register_tensor(
                    t,
                    DTensorSpec(
                        mesh=spec.mesh,
                        placements=spec.placements,
                        tensor_meta=TensorMeta(
                            shape=tuple(t.shape),
                            stride=tuple(t.stride()),
                            dtype=t.dtype,
                        ),
                    ),
                )

        if len(misses) > 0:
            uniq_misses = sorted(set(misses))
            uniq_peer_shapes = sorted(set(shape_to_spec.keys()))
            raise RuntimeError(
                f"unbound tensor(s) with shapes {uniq_misses} "
                f"have no same-shape peer among {uniq_peer_shapes} in {op}"
            )

    def lookup_spec(self, t: torch.Tensor) -> DTensorSpec:
        try:
            return self._tensor_id_to_spec[id(t)]
        except KeyError:
            raise RuntimeError(
                "Missing DTensorSpec for FakeTensor encountered in _SpecMode"
            )

    def extract_specs(self, obj):
        if isinstance(obj, torch.Tensor):
            return self.lookup_spec(obj)
        elif isinstance(obj, list):
            return [self.extract_specs(x) for x in obj]
        elif isinstance(obj, tuple):
            return tuple(self.extract_specs(x) for x in obj)
        else:
            return obj

    def _bind_expected_specs(self, fake_arg, expected_spec) -> None:
        if isinstance(fake_arg, torch.Tensor) and isinstance(
            expected_spec, DTensorSpec
        ):
            cur = self._tensor_id_to_spec.get(id(fake_arg))
            if cur is None or cur.placements != expected_spec.placements:
                self.register_tensor(fake_arg, expected_spec)
        elif isinstance(fake_arg, (list, tuple)) and isinstance(
            expected_spec, (list, tuple)
        ):
            for fa, es in zip(fake_arg, expected_spec):
                self._bind_expected_specs(fa, es)

    def __torch_dispatch__(self, op, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        # call directly if prim
        if _is_prim_op(op):
            with no_dispatch(), disable_proxy_modes_tracing():
                return op(*args, **kwargs)

        # check for unbound args, bind to same-shape peers
        # if possible
        self._ensure_all_tensor_args_bound(op, args, kwargs)

        def to_schema(x):
            if isinstance(x, torch.Tensor):
                return self.lookup_spec(x)
            elif isinstance(x, list):
                return [to_schema(i) for i in x]
            elif isinstance(x, tuple):
                return tuple(to_schema(i) for i in x)
            else:
                return x

        args_schema = tuple(to_schema(a) for a in args)
        kwargs_schema = {k: to_schema(v) for k, v in kwargs.items()}
        schema_info = self.propagator.op_to_schema_info.get(op, None)
        op_schema = OpSchema(op, args_schema, kwargs_schema, schema_info)

        # try prop
        out_sharding = self.propagator.propagate_op_sharding(op_schema)

        # maybe bail if downstream op requires redistribute
        if self._fail_on_redistribute and out_sharding.needs_redistribute:
            raise _RedistributeNeeded

        # keep trying prop until we don't need a redistribute
        while (
            out_sharding.needs_redistribute
            and out_sharding.redistribute_schema is not None
        ):
            redist_schema = out_sharding.redistribute_schema

            # bind expected specs for args + kwargs
            for fa, es in zip(args, redist_schema.args_schema):
                self._bind_expected_specs(fa, es)
            for k, es in redist_schema.kwargs_schema.items():
                if k in kwargs:
                    self._bind_expected_specs(kwargs[k], es)

            out_sharding = self.propagator.propagate_op_sharding(redist_schema)

        out_specs = out_sharding.output_spec

        def make_fake_from_spec(spec: DTensorSpec) -> torch.Tensor:
            tm = spec.tensor_meta
            if not isinstance(tm, TensorMeta):
                raise RuntimeError(
                    "Decomposition output DTensorSpec must have TensorMeta"
                )

            with no_dispatch():
                t = (
                    torch.empty_strided(
                        tm.shape, tm.stride, dtype=tm.dtype, device="meta"
                    )
                    if tm.stride is not None
                    else torch.empty(tm.shape, dtype=tm.dtype, device="meta")
                )

            self.register_tensor(t, spec)
            return t

        def wrap_outputs(specs):
            if isinstance(specs, DTensorSpec):
                return make_fake_from_spec(specs)
            elif isinstance(specs, list):
                return [
                    make_fake_from_spec(s) if isinstance(s, DTensorSpec) else s
                    for s in specs
                ]
            elif isinstance(specs, tuple):
                return tuple(
                    make_fake_from_spec(s) if isinstance(s, DTensorSpec) else s
                    for s in specs
                )
            else:
                return specs

        return wrap_outputs(out_specs)


def _fake_from_input_spec(spec: DTensorSpec) -> torch.Tensor:
    tm = spec.tensor_meta
    if not isinstance(tm, TensorMeta):
        raise RuntimeError(
            "Input DTensorSpec used for eager decomposition must have TensorMeta"
        )

    # create a meta tensor that FakeTensorMode will wrap into a FakeTensor
    if tm.stride is not None:
        return torch.empty_strided(tm.shape, tm.stride, dtype=tm.dtype, device="meta")
    else:
        return torch.empty(tm.shape, dtype=tm.dtype, device="meta")


def _materialize_fake_args_from_schema(schema: OpSchema) -> tuple[tuple, dict]:
    def conv_arg(x):
        if isinstance(x, DTensorSpec):
            return _fake_from_input_spec(x)
        elif isinstance(x, list):
            return [conv_arg(i) for i in x]
        elif isinstance(x, tuple):
            return tuple(conv_arg(i) for i in x)
        else:
            return x

    args = tuple(conv_arg(a) for a in schema.args_schema)
    kwargs = {k: conv_arg(v) for k, v in schema.kwargs_schema.items()}
    return args, kwargs


def _attach_specs_to_fakes(mode: _SpecMode, obj, spec_obj):
    if isinstance(obj, torch.Tensor) and isinstance(spec_obj, DTensorSpec):
        mode.register_tensor(obj, spec_obj)
    elif (
        isinstance(obj, (list, tuple))
        and isinstance(spec_obj, (list, tuple))
        and len(obj) == len(spec_obj)
    ):
        for o, s in zip(obj, spec_obj):
            _attach_specs_to_fakes(mode, o, s)
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            _attach_specs_to_fakes(mode, o, spec_obj)


def _make_replicate_like(spec: DTensorSpec) -> DTensorSpec:
    return DTensorSpec(
        mesh=spec.mesh,
        placements=tuple(Replicate() for _ in spec.placements),
        tensor_meta=spec.tensor_meta,
    )


def _argspecs_to_tuple(argspecs):
    if isinstance(argspecs, list):
        return tuple(argspecs)
    return argspecs


class ShardingPropagator:
    def __init__(self) -> None:
        self.op_to_rules: dict[OpOverload, Callable[[OpSchema], OutputSharding]] = {}
        self.op_strategy_funcs: dict[
            OpOverload,
            Callable[[OpSchema], StrategyType],
        ] = {}
        # op map to save static argnum to decide to reuse sharding prop cache or
        # re-run sharding prop
        self.op_to_schema_info: dict[OpOverload, RuntimeSchemaInfo] = {}
        self.propagate_op_sharding = LocalLRUCache(
            self.propagate_op_sharding_non_cached
        )
        # op map to save indices of shape (and stride) args which may need to be
        # modified in sharding prop
        self.op_to_shape_and_stride_idx: dict[
            OpOverload, Union[int, tuple[int, int]]
        ] = {
            # new factory ops
            aten.new_empty.default: 1,
            aten.new_full.default: 1,
            aten.new_ones.default: 1,
            aten.new_zeros.default: 1,
            aten.new_empty_strided.default: (1, 2),
            # view ops
            aten.expand.default: 1,
            aten.reshape.default: 1,
            aten.view.default: 1,
            aten._unsafe_view.default: 1,
            aten.select_backward.default: 1,
            aten.slice_backward.default: 1,
        }
        self.decomposition_fns: dict[OpOverload, Callable] = {}

    def register_op_decomposition(
        self,
        op_overload: OpOverload,
        decomposition_fn: Callable,
        schema_info: Optional[RuntimeSchemaInfo] = None,
    ):
        """
        register a decomposition function for an unsupported op so we can
        shard-propagate by executing its decomposition body eagerly.
        """
        self.decomposition_fns[op_overload] = decomposition_fn
        if schema_info is not None:
            self.op_to_schema_info[op_overload] = schema_info

    def register_sharding_prop_rule(
        self,
        op_overload: OpOverload,
        rule_func: Callable[[OpSchema], OutputSharding],
        schema_info: Optional[RuntimeSchemaInfo] = None,
    ):
        """
        Register a sharding propagation rule for an operator.
        """
        self.op_to_rules[op_overload] = rule_func
        if schema_info is not None:
            self.op_to_schema_info[op_overload] = schema_info

    def register_op_strategy(
        self,
        op_overload: OpOverload,
        strategy_func: Callable[[OpSchema], StrategyType],
        schema_info: Optional[RuntimeSchemaInfo] = None,
    ):
        """
        Register a :class:`OpStrategy` generator for an operator.

        During the sharding propagation, DTensor wants to enumerate all
        acceptable sharding specs (:class:`OpSpec`) for an operator,
        and by "acceptable" we mean that the operator can be executed on
        the ``_local_tensor`` of DTensor args/kwargs (with ``OpSpec.input_specs``)
        and the output(s) constitute valid DTensor(s) (with ``OpSpec.output_specs``).

        ``strategy_func`` is the function that enumerates such acceptable specs
        for the operator ``op_overload``. One general approach to write ``strategy_func``
        is, if the operator has simple arguments structure (e.g. mm, bmm), first enumerating
        all sharding specs for the operands, and then filtering out the ones that
        are not valid. For example, for ``mm``, the operands are two 2D tensors, and
        if both ``input`` and ``mat2`` have sharding placements ``[Shard(0)]``, then this
        is not an acceptable ``input_specs``.

        Once we have a way to enumerate all acceptable sharding specs, we can use each
        of them to construct a :class:`OpSpec`. The ``OpSpec.input_specs`` directly comes
        from the sharding spec, and the ``OpSpec.output_specs`` is therefore determined
        (e.g. ``[Shard(1)]`` @ ``[Shard(0)]`` yields ``[Partial()]``). In addition,
        :class:`OpSpec` also contains ``redistribute_cost`` which records the redistribution
        cost from each :class:`OpSpec` in the source :class:`OpStrategy.strategies` to
        the target sharding spec, for each operand.

        The ``strategy_func`` should return a :class:`OpStrategy` which contains a list of
        all the :class:`OpSpec`s generated in the above.

        The optional ``schema_info`` tells which non-DTensor args/kwargs could affect the
        cache and whether ``pytree`` is needed to flatten the nested args. ``static_argnum``
        marks the starting index of the non-DTensor args that should be hashed into the
        sharding propagation hash key, and ``static_kwargkey`` marks the keys of the
        non-DTensor kwargs that should be hashed. ``needs_pytree`` should be used when
        the input arg has :class:`list` or :class:`dict` structure.

        For example, ``aten.cat.default`` op has a ``List[Tensor]`` argument ``tensors``
        and an ``int`` argument ``dim``. Because ``dim`` affects the sharding propagation
        result, we want to pass ``RuntimeSchemaInfo(static_argnum=1)`` because the argument
        index of ``dim`` is 1. Besides, we also want to set ``needs_pytree=True`` because
        ``tensors`` needs be flattened in sharding propagation. Another example is
        ``aten.histc.default``. ``histc`` has 4 arguments (self, bins, min, max) and the
        last two would affect sharding propagation along with the :class:`DTensor` argument
        ``self``. Since the argument index of ``min`` is 2, the `schema_info` should be
        `RuntimeSchemaInfo(static_argnum=2)`.
        """
        self.op_strategy_funcs[op_overload] = strategy_func
        if schema_info is not None:
            self.op_to_schema_info[op_overload] = schema_info

    def _propagate_tensor_meta_non_cached(
        self, op_schema: OpSchema
    ) -> Union[None, TensorMeta, Sequence[Optional[TensorMeta]]]:
        """
        Propagate the tensor metadata, it could either return a TensorMeta
        or a list/tuple of TensorMetas
        """
        if op_schema.op == aten.equal.default:
            # data dependent ops can't be used for fake propagation
            return None

        # NOTE: We must call the tracing in fake tensor mode so that it avoids
        # materializing memory. Also disable the proxy mode tracing to prevent
        # these operators to be inserted in the fx graph.

        with FakeTensorMode(), disable_proxy_modes_tracing():
            fake_args = op_schema.gen_fake_args()
            fake_kwargs = op_schema.gen_fake_kwargs()
            fake_out = op_schema.op(*fake_args, **fake_kwargs)

        if isinstance(fake_out, torch.Tensor):
            return TensorMeta(
                shape=fake_out.shape, stride=fake_out.stride(), dtype=fake_out.dtype
            )

        elif isinstance(fake_out, (tuple, list)):
            tensor_meta_list: list[Optional[TensorMeta]] = []
            for fake_out_item in fake_out:
                if isinstance(fake_out_item, torch.Tensor):
                    tensor_meta_list.append(
                        TensorMeta(
                            shape=fake_out_item.shape,
                            stride=fake_out_item.stride(),
                            dtype=fake_out_item.dtype,
                        )
                    )
                else:
                    tensor_meta_list.append(None)
            return (
                tuple(tensor_meta_list)
                if isinstance(fake_out, tuple)
                else tensor_meta_list
            )
        else:
            # if fake is not a tensor or tuple of tensor, return as none
            return None

    @lru_cache  # noqa: B019
    def _propagate_tensor_meta(
        self, op_schema: OpSchema
    ) -> Union[None, TensorMeta, Sequence[Optional[TensorMeta]]]:
        """
        Cached version of _propagate_tensor_meta_non_cached
        This is a private API. Use propagate_tensor_meta instead.
        """
        return self._propagate_tensor_meta_non_cached(op_schema)

    def propagate_tensor_meta(
        self, op_schema: OpSchema
    ) -> Union[None, TensorMeta, Sequence[Optional[TensorMeta]]]:
        """
        Propagate the tensor metadata, it could either return a TensorMeta
        or a list/tuple of TensorMetas. This is a public API that should be
        used if cache should be used.
        """
        if _are_we_tracing():
            return self._propagate_tensor_meta_non_cached(op_schema)
        else:
            return self._propagate_tensor_meta(op_schema)

    def _create_output_spec_with_new_tensor_meta(
        self,
        op: OpOverload,
        output_specs: OutputSpecType,
        output_tensor_meta: Union[None, TensorMeta, Sequence[Optional[TensorMeta]]],
    ) -> OutputSpecType:
        """
        Wrap the output_specs with the tensor metadata from the output.
        """

        if isinstance(output_specs, DTensorSpec):
            if not isinstance(output_tensor_meta, TensorMeta):
                # Either error due to ShardingPropagator or due to incorrect OutputSpec
                if not isinstance(output_tensor_meta, (tuple, list)):
                    raise ValueError(
                        "ShardingPropagator error: output does not have an associated "
                        "TensorMeta"
                    )
                raise ValueError(
                    f"For the op {op.name()}, `output_specs` has 1 output which does "
                    "not equal the "
                    f"number of op outputs: {len(output_tensor_meta)}."
                )
            return output_specs.shallow_copy_with_tensor_meta(output_tensor_meta)
        elif isinstance(output_specs, (tuple, list)):
            new_specs: list[Optional[DTensorSpec]] = []
            if not isinstance(output_tensor_meta, (tuple, list)) or len(
                output_specs
            ) != len(output_tensor_meta):
                raise ValueError(
                    f"For the op {op.name()}, `output_specs` has {len(output_specs)} "
                    "outputs which does not equal the "
                    f"number of op outputs {_length(output_tensor_meta)}."
                )

            for i, spec in enumerate(output_specs):
                if isinstance(spec, DTensorSpec):
                    output_tensor_meta_i = output_tensor_meta[i]
                    if not isinstance(output_tensor_meta_i, TensorMeta):
                        # NOTE: aten.convolution_backward.default is an exception and it
                        # needs extra handling because the first Tensor in the output
                        # tuple can be `None` if the input Tensor to convolution op has
                        # `requires_grad=False` (e.g. convolution layer is the first
                        # layer in the model). We explicitly allow its corresponding
                        # TensorMeta to be `None`.
                        if (
                            op == aten.convolution_backward.default
                            and i == 0
                            and output_tensor_meta_i is None
                        ):
                            assert isinstance(output_specs, list)
                            new_specs.append(None)
                            continue
                        else:
                            raise ValueError(
                                f"ShardingPropagator error: output {i} of {op.name()} "
                                "does not have an associated TensorMeta"
                            )

                    new_specs.append(
                        spec.shallow_copy_with_tensor_meta(output_tensor_meta_i)
                    )
                else:
                    new_specs.append(spec)

            return tuple(new_specs)
        else:
            assert output_specs is None
            return output_specs

    def _wrap_with_op_strategy(self, op_schema: OpSchema) -> OpSchema:
        """
        wrap a op_schema that contains DTensorSpec to another op_schema that contains
        OpStrategy/TupleStrategy, the returned op_schema is then used for sharding
        strategy propagation on pytorch operators.
        """

        def spec_to_strategy(spec: object) -> object:
            if isinstance(spec, DTensorSpec):
                return OpStrategy([OpSpec(spec)])
            elif (
                isinstance(spec, (list, tuple))
                and len(spec) > 0
                and isinstance(spec[0], DTensorSpec)
            ):
                # tensor list create tuple strategy
                tuple_strategy = [spec_to_strategy(s) for s in spec]
                tuple_strategy = cast(Sequence[StrategyType], tuple_strategy)
                return TupleStrategy(
                    tuple(tuple_strategy) if isinstance(spec, tuple) else tuple_strategy
                )
            else:
                return spec

        args_op_strategy = [spec_to_strategy(i) for i in op_schema.args_schema]

        kwargs_op_strategy = {
            k: spec_to_strategy(v) for k, v in op_schema.kwargs_schema.items()
        }

        return OpSchema(
            op=op_schema.op,
            args_schema=tuple(args_op_strategy),
            kwargs_schema=kwargs_op_strategy,
            schema_info=op_schema.schema_info,
        )

    def propagate(self, op_info: OpInfo) -> None:
        # We cannot use an lru cache if we know that inputs will have dynamic shapes,
        # because SymInts are not hashable.
        # This is generally ok because this only happens during tracing in torch.compile,
        # and tracing does not need to be as fast as eagermode DTensor usages.
        if _are_we_tracing():
            output_sharding = self.propagate_op_sharding_non_cached(op_info.schema)
        else:
            output_sharding = cast(
                OutputSharding, self.propagate_op_sharding(op_info.schema)
            )
        op_info.output_sharding = output_sharding

    def _try_decomp_once(self, op_schema: OpSchema, *, fail_on_redistribute: bool):
        """run the registered decomp exactly once with SpecMode, returning:
        (out_specs, (observed_args, observed_kwargs)).
        - if fail_on_redistribute=True and any inner op needs redistribution, raises _RedistributeNeeded.
        - otherwise returns (out_specs, observed inputs after any internal rebinding).
        """
        with FakeTensorMode(), disable_proxy_modes_tracing():
            fake_args, fake_kwargs = _materialize_fake_args_from_schema(op_schema)

            with _SpecMode(self, fail_on_redistribute=fail_on_redistribute) as mode:
                _attach_specs_to_fakes(mode, fake_args, op_schema.args_schema)
                _attach_specs_to_fakes(mode, fake_kwargs, op_schema.kwargs_schema)

                outs = self.decomposition_fns[op_schema.op](*fake_args, **fake_kwargs)
                out_specs = mode.extract_specs(outs)

                observed_inputs = (
                    mode.extract_specs(fake_args),
                    mode.extract_specs(fake_kwargs),
                )
                return out_specs, observed_inputs

    def _outputs_with_tensor_meta(
        self, op_schema: OpSchema, out_specs
    ) -> OutputSpecType:
        out_tensor_meta = self._propagate_tensor_meta(op_schema)
        return self._create_output_spec_with_new_tensor_meta(
            op_schema.op, out_specs, out_tensor_meta
        )

    def _try_eager_decomposition(self, op_schema: OpSchema) -> Optional[OutputSharding]:
        decomp_fn = self.decomposition_fns.get(op_schema.op, None)
        if decomp_fn is None:
            return None

        # try no-redistribute case, fail if any inner op requires redistribute
        try:
            out_specs_0, _ = self._try_decomp_once(op_schema, fail_on_redistribute=True)
            out_specs_with_meta = self._outputs_with_tensor_meta(op_schema, out_specs_0)
            return OutputSharding(
                output_spec=out_specs_with_meta,
                redistribute_schema=None,
                needs_redistribute=False,
                use_val_from_redistribute_schema=False,
            )
        except _RedistributeNeeded:
            pass

        # get minimally changed placements from decomp
        out_specs_sugg, (suggested_args, suggested_kwargs) = self._try_decomp_once(
            op_schema, fail_on_redistribute=False
        )

        udim = unique_shard_dim_across_inputs(
            op_schema.args_schema, op_schema.kwargs_schema
        )

        # if there is a unique shard dim in the inputs
        # try to apply local re-chunking (shard inputs
        # consistently across the unique dim, no collective)
        # allowing us to keep the output sharded
        if udim is not None:
            aligned_args, aligned_kwargs = align_all_inputs_to_dim(
                op_schema.args_schema, op_schema.kwargs_schema, udim
            )
            align_schema = OpSchema(
                op_schema.op, _argspecs_to_tuple(aligned_args), aligned_kwargs
            )
            align_schema._inplace_rewrap_schema_suggestion(op_schema)
            try:
                out_specs_align, _ = self._try_decomp_once(
                    align_schema, fail_on_redistribute=True
                )
                out_specs_with_meta = self._outputs_with_tensor_meta(
                    op_schema, out_specs_align
                )
                return OutputSharding(
                    output_spec=out_specs_with_meta,
                    redistribute_schema=align_schema,
                    needs_redistribute=True,
                    use_val_from_redistribute_schema=False,
                )
            except _RedistributeNeeded:
                pass

        flat_orig, args_spec, kwargs_spec, n_args, dt_idx = flatten_inputs(
            op_schema.args_schema, op_schema.kwargs_schema
        )

        if suggested_args is not None or suggested_kwargs is not None:
            flat_sugg, _, _, _, _ = flatten_inputs(
                suggested_args if suggested_args is not None else op_schema.args_schema,
                (
                    suggested_kwargs
                    if suggested_kwargs is not None
                    else op_schema.kwargs_schema
                ),
            )
        else:
            flat_sugg = None

        # try to see if minimal plan propagates all the way
        tried = set()
        sig_orig = sig_flat(flat_orig, dt_idx)
        if flat_sugg is not None:
            sig_min = sig_flat(flat_sugg, dt_idx)
            if sig_min != sig_orig:
                min_args, min_kwargs = unflatten_inputs(
                    flat_sugg, args_spec, kwargs_spec, n_args
                )
                min_schema = OpSchema(
                    op_schema.op, _argspecs_to_tuple(min_args), min_kwargs
                )
                min_schema._inplace_rewrap_schema_suggestion(op_schema)
                try:
                    out_specs_min, _ = self._try_decomp_once(
                        min_schema, fail_on_redistribute=True
                    )
                    out_specs_with_meta = self._outputs_with_tensor_meta(
                        op_schema, out_specs_min
                    )
                    return OutputSharding(
                        output_spec=out_specs_with_meta,
                        redistribute_schema=min_schema,
                        needs_redistribute=True,
                        use_val_from_redistribute_schema=False,
                    )
                except _RedistributeNeeded:
                    tried.add(sig_min)

        # get all possible input candidates and sort by (cost asc, keep-shard desc)
        per_leaf = candidate_lists_per_dt_leaf(
            flat_orig, flat_sugg, dt_idx, _make_replicate_like
        )

        scored = []
        for combo in product(*per_leaf) if per_leaf else [()]:
            flat_choice = apply_dt_choices(flat_orig, dt_idx, combo)
            sig = sig_flat(flat_choice, dt_idx)
            if sig == sig_orig or sig in tried:
                continue
            cost = total_cost_flat(flat_orig, flat_choice, dt_idx)
            neg_shard_pref = -shard_score_flat(flat_choice, dt_idx)
            scored.append((cost, neg_shard_pref, flat_choice, sig))

        scored.sort(key=lambda x: (x[0], x[1]))

        # try each sorted candidate until we find one that props
        for _cost, _neg_score, flat_choice, sig in scored:
            args_choice, kwargs_choice = unflatten_inputs(
                flat_choice, args_spec, kwargs_spec, n_args
            )
            trial_schema = OpSchema(
                op_schema.op, _argspecs_to_tuple(args_choice), kwargs_choice
            )
            trial_schema._inplace_rewrap_schema_suggestion(op_schema)
            try:
                out_specs_trial, _ = self._try_decomp_once(
                    trial_schema, fail_on_redistribute=True
                )
                out_specs_with_meta = self._outputs_with_tensor_meta(
                    op_schema, out_specs_trial
                )
                return OutputSharding(
                    output_spec=out_specs_with_meta,
                    redistribute_schema=trial_schema,
                    needs_redistribute=True,
                    use_val_from_redistribute_schema=False,
                )
            except _RedistributeNeeded:
                tried.add(sig)
                continue

        raise RuntimeError(
            f"eager decomposition failed to find a consistent redistribution for {op_schema.op.name()}."
        )

    def propagate_op_sharding_non_cached(self, op_schema: OpSchema) -> OutputSharding:
        """
        Propagate the sharding for an operator given the op_schema.
        """
        # special case op, we don't need to propagate for local
        # scalar. TODO: figure out a better way to handle this
        if op_schema.op is aten._local_scalar_dense.default:
            return OutputSharding(None, op_schema)

        out_tensor_meta = self._propagate_tensor_meta_non_cached(op_schema)
        if op_schema.op in self.op_strategy_funcs:
            # wrap the op_schema with op strategy for sharding strategy propagation
            strategy_schema = self._wrap_with_op_strategy(op_schema)

            # run sharding strategy propagation/generation
            op_strategy = self.op_strategy_funcs[op_schema.op](strategy_schema)

            if isinstance(op_strategy, OpStrategy):
                # single Op strategy
                output_strategy = self._select_strategy(op_strategy, op_schema)

                # check if we need to redistribute the input
                needs_redistribute = False
                # check if we want to use args value from redistribute_schema
                use_val_from_redistribute_schema = False
                expected_input_specs: list[DTensorSpec] = []

                # in case where the op does not specify input_specs and output_specs
                # is a DTensorSpec, we use output_specs as the spec for each DTensor
                # input arg.
                if output_strategy.input_specs is None:
                    assert isinstance(output_strategy.output_specs, DTensorSpec)

                for idx, input_spec in enumerate(op_schema.args_spec):
                    desired_spec = (
                        output_strategy.output_spec
                        if output_strategy.input_specs is None
                        else output_strategy.input_specs[idx]
                    )
                    expected_input_specs.append(
                        desired_spec.shallow_copy_with_tensor_meta(
                            input_spec.tensor_meta
                        )
                    )
                    if input_spec.placements != desired_spec.placements:
                        needs_redistribute = True

                suggestion_schema = None
                if needs_redistribute:
                    suggestion_schema = OpSchema(
                        op_schema.op, tuple(expected_input_specs), {}
                    )
                    suggestion_schema._inplace_rewrap_schema_suggestion(op_schema)

                # shape and stride args need to be modified for
                # view ops and new factory ops, potentially
                if op_schema.op in self.op_to_shape_and_stride_idx:
                    assert isinstance(output_strategy.output_spec, DTensorSpec)
                    # It happens when the output has the same shape as the input
                    # and the input placements are not all Replicate().
                    if output_strategy.output_spec.is_sharded():
                        schema = suggestion_schema or op_schema
                        assert isinstance(out_tensor_meta, TensorMeta)
                        suggestion_schema = self._adjust_shape_and_stride_args(
                            out_tensor_meta, schema, output_strategy.output_spec
                        )
                        needs_redistribute = True
                        use_val_from_redistribute_schema = True

                # construct output spec for the op
                if op_schema.return_type_tuple_tensor_like():
                    # for ops that return multiple tensors and the output_specs is not
                    # a tuple, we use a tuple of that single output spec as the new
                    # output_specs
                    output_specs: OutputSpecType = output_strategy.output_specs
                    if isinstance(output_specs, DTensorSpec):
                        output_specs = tuple(
                            # create a new DTensorSpec with the same placement as the
                            # output_specs in output_strategy
                            DTensorSpec(
                                mesh=output_specs.mesh,
                                placements=output_specs.placements,
                                tensor_meta=output_specs.tensor_meta,
                            )
                            for _ in range(len(op_schema.op._schema.returns))
                        )
                elif (
                    op_schema.return_type_tensor()
                    or op_schema.return_type_list_tensor_like()
                ):
                    output_specs = output_strategy.output_specs
                else:
                    output_specs = None

                output_sharding = OutputSharding(
                    output_specs,
                    suggestion_schema,
                    needs_redistribute=needs_redistribute,
                    use_val_from_redistribute_schema=use_val_from_redistribute_schema,
                )
            elif isinstance(op_strategy, TupleStrategy):
                # tuple strategy output sharding processing
                # runtime select OpSpec for each TupleStrategy input arg
                selected_strategies: list[OpSpec] = []
                out_spec_list: list[DTensorSpec] = []
                for strategy in op_strategy.children:
                    assert isinstance(strategy, OpStrategy)
                    selected_strategy = self._select_strategy(strategy)
                    selected_strategies.append(selected_strategy)
                    out_spec_list.append(selected_strategy.output_spec)

                needs_redistribute = False
                suggestion_args: list[object] = []
                tensor_or_list_tensor_arg_idx = 0

                for arg in op_schema.args_schema:
                    if (
                        arg
                        and isinstance(arg, (list, tuple))
                        and isinstance(arg[0], DTensorSpec)
                    ):
                        expected_input_spec_list: list[DTensorSpec] = []
                        for idx, arg_spec in enumerate(arg):
                            expected_input_spec = selected_strategies[idx].input_spec(
                                tensor_or_list_tensor_arg_idx
                            )
                            expected_input_spec = (
                                expected_input_spec.shallow_copy_with_tensor_meta(
                                    arg_spec.tensor_meta
                                )
                            )
                            if arg_spec.placements != expected_input_spec.placements:
                                needs_redistribute = True
                            expected_input_spec_list.append(expected_input_spec)
                        suggestion_args.append(
                            tuple(expected_input_spec_list)
                            if isinstance(arg, tuple)
                            else expected_input_spec_list
                        )
                        tensor_or_list_tensor_arg_idx += 1

                    elif isinstance(arg, DTensorSpec):
                        expected_input_spec = selected_strategies[0].input_spec(
                            tensor_or_list_tensor_arg_idx
                        )
                        expected_input_spec = (
                            expected_input_spec.shallow_copy_with_tensor_meta(
                                arg.tensor_meta
                            )
                        )
                        if arg.placements != expected_input_spec.placements:
                            needs_redistribute = True
                        suggestion_args.append(expected_input_spec)
                        tensor_or_list_tensor_arg_idx += 1
                    else:
                        suggestion_args.append(arg)

                suggestion_schema = None
                if needs_redistribute:
                    suggestion_schema = OpSchema(
                        op_schema.op, tuple(suggestion_args), op_schema.kwargs_schema
                    )

                output_sharding = OutputSharding(
                    tuple(out_spec_list) if out_tensor_meta is not None else None,
                    suggestion_schema,
                    needs_redistribute=needs_redistribute,
                    use_val_from_redistribute_schema=False,
                )
            else:
                raise ValueError("Unsupported op strategy type")

            # associate the output sharding with the output tensor metadata
            new_output_spec = self._create_output_spec_with_new_tensor_meta(
                op_schema.op, output_sharding.output_spec, out_tensor_meta
            )
            output_sharding.output_spec = new_output_spec
            return output_sharding
        elif op_schema.op in self.decomposition_fns:
            res = self._try_eager_decomposition(op_schema)
            if res is not None:
                return res
            # fallthrough to error
        elif op_schema.op in self.op_to_rules:
            # propagate the sharding with rule
            sharding_prop_func = self.op_to_rules[op_schema.op]

            # step 1. there's sharding propagation rule, run
            # sharding propagation to get the output sharding
            try:
                output_sharding = sharding_prop_func(op_schema)
            except NotImplementedError as e:
                raise e
            except Exception as e:
                raise RuntimeError(
                    f"Sharding propagation failed on op {op_schema}.\nError: {e}"
                ) from e

            # step 2. if can't get output_spec from sharding
            # propagation (i.e. no rules apply for input
            # placements), we return the output sharding
            # with schema suggestions, which can be used to
            # decide how to do redistribute on inputs
            if output_sharding.output_spec is None:
                if output_sharding.redistribute_schema is None:
                    raise RuntimeError(
                        f"Sharding propagation failed on op {op_schema}!"
                    )
                else:
                    # we do auto redistribute on inputs if necessary
                    # run sharding propagation again with suggested schema
                    propagation_res = sharding_prop_func(
                        output_sharding.redistribute_schema
                    )
                    # we set the output sharding with the new propagation result
                    # so that dispatching know both output_spec and redistribute_schema
                    # exist, which indicates a reshard is needed
                    output_sharding.output_spec = propagation_res.output_spec
                    output_sharding.needs_redistribute = True

            # associate the output sharding with the output tensor metadata
            new_output_spec = self._create_output_spec_with_new_tensor_meta(
                op_schema.op, output_sharding.output_spec, out_tensor_meta
            )
            output_sharding.output_spec = new_output_spec

            return output_sharding
        else:
            raise NotImplementedError(
                f"Operator {op_schema.op} does not have a sharding strategy registered."
            )

    def _select_strategy(
        self, strategy: OpStrategy, op_schema: Optional[OpSchema] = None
    ) -> OpSpec:
        if len(strategy.strategies) == 1:
            # short cut with only one possible OpSpec
            return strategy.strategies[0]

        op_spec_costs: list[float] = []
        no_redistribute_strategy_index: int = -1
        for strategy_idx, op_spec in enumerate(strategy.strategies):
            assert op_spec.redistribute_cost is not None, (
                "must set redistribute cost each OpSpec!"
            )
            redistribute_cost = sum(chain.from_iterable(op_spec.redistribute_cost))
            op_spec_costs.append(redistribute_cost)

            # If there's no redistribute cost, we record the index of the strategy
            # which doesn't need redistribute.
            # TODO: Currently this only applies to OpStrategy selection. Requires extra
            # logic to make it work for TupleStrategy, if needed.
            if op_schema is not None and redistribute_cost == 0:
                needs_redistribute = False
                for spec_idx, input_spec in enumerate(op_schema.args_spec):
                    desired_spec = (
                        op_spec.output_spec
                        if op_spec.input_specs is None
                        else op_spec.input_specs[spec_idx]
                    )
                    if input_spec.placements != desired_spec.placements:
                        needs_redistribute = True
                        break

                if not needs_redistribute:
                    no_redistribute_strategy_index = strategy_idx

        # for eager execution, we just select the one with the minimal redistribute cost
        min_cost = min(op_spec_costs)
        if min_cost < 0:
            # If there's negative cost, we select the one with the minimal cost,
            # even if this means we need to redistribute, e.g. via local chunking.
            # E.g. this can happen for ops in self.op_to_shape_and_stride_idx
            # when the inputs / outputs are sharded.
            selected_strategy_index = op_spec_costs.index(min_cost)
        elif min_cost == 0 and no_redistribute_strategy_index != -1:
            # If there's no redistribute cost, we select the one with no redistribute.
            selected_strategy_index = no_redistribute_strategy_index
        else:
            selected_strategy_index = op_spec_costs.index(min_cost)

        return strategy.strategies[selected_strategy_index]

    def _adjust_shape_and_stride_args(
        self,
        out_tensor_meta: TensorMeta,
        schema: OpSchema,
        spec: DTensorSpec,
    ) -> OpSchema:
        shape_stride_idx = self.op_to_shape_and_stride_idx[schema.op]
        if isinstance(shape_stride_idx, tuple):
            shape_idx, stride_idx = shape_stride_idx
        else:
            shape_idx = shape_stride_idx
            stride_idx = None

        expected_input_schema = list(schema.args_schema)
        # adjust shape to be the same as that of the _local_tensor
        # of the DTensor input arg at index 0, which is inferred
        expected_input_schema[shape_idx], _ = compute_local_shape_and_global_offset(
            out_tensor_meta.shape, spec.mesh, spec.placements
        )

        # adjust the stride arg for aten.new_empty_strided.default
        if stride_idx:
            expected_input_schema[stride_idx] = compute_local_stride(
                out_tensor_meta.stride, spec.mesh, spec.placements
            )

        return OpSchema(schema.op, tuple(expected_input_schema), schema.kwargs_schema)
