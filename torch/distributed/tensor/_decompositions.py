# mypy: allow-untyped-defs
"""
Decomposition-based sharding propagation for DTensor.

When an operator doesn't have a registered sharding strategy, we derive one by
tracing through its decomposition. The decomposed ops (which do have strategies)
determine how placements propagate through the original op.
"""

from __future__ import annotations

import itertools
from typing import Any, TYPE_CHECKING

import torch
from torch._decomp import decomposition_table
from torch.distributed._functional_collectives import _are_we_tracing
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import OpSchema, OpStrategy, RuntimeSchemaInfo
from torch.distributed.tensor._utils import try_find_mesh_from_args
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Placement,
    Replicate,
    Shard,
)
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode
from torch.utils._python_dispatch import TorchDispatchMode


def _infer_schema_info_from_op(op: OpOverload) -> RuntimeSchemaInfo:
    """Infer RuntimeSchemaInfo from an operator's schema for decomposition ops"""
    schema = op._schema

    # Find first non-tensor positional arg index
    static_argnum = None
    for i, arg in enumerate(schema.arguments):
        if arg.kwarg_only:
            break
        if arg.type.kind() != "TensorType" and static_argnum is None:
            static_argnum = i
            break

    # Find keyword-only args that aren't tensors
    kwarg_only_names = []
    for arg in schema.arguments:
        if arg.kwarg_only and arg.type.kind() != "TensorType":
            kwarg_only_names.append(arg.name)

    kwargs = {}
    if static_argnum is not None:
        kwargs["static_argnum"] = static_argnum
    if kwarg_only_names:
        # pyrefly: ignore [unsupported-operation]
        kwargs["static_kwargkey"] = kwarg_only_names

    # pyrefly: ignore [bad-argument-type]
    return RuntimeSchemaInfo(**kwargs)


from torch.utils._pytree import tree_any, tree_flatten, tree_map, tree_map_only


if TYPE_CHECKING:
    from torch._ops import OpOverload
    from torch.distributed.tensor._sharding_prop import ShardingPropagator


def _extract_input_specs(op_schema: OpSchema) -> tuple[DTensorSpec | object, ...]:
    return op_schema.args_schema + tuple(op_schema.kwargs_schema.values())


class PlacementTrackingMode(TorchDispatchMode):
    """
    TorchDispatchMode that tracks DTensor placements through op execution.

    Used during decomposition tracing: intercepts each op, propagates sharding
    via the ShardingPropagator, and records output placements on the result tensors.
    """

    def __init__(self, sharding_prop: ShardingPropagator, mesh: DeviceMesh):
        super().__init__()
        self.sharding_prop = sharding_prop
        self.mesh = mesh

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        args_schema, kwargs_schema = tree_map(
            lambda x: getattr(x, "_spec", x) if isinstance(x, torch.Tensor) else x,
            (args, kwargs or {}),
        )

        if not tree_any(
            lambda x: isinstance(x, DTensorSpec), (args_schema, kwargs_schema)
        ):
            raise NotImplementedError(f"No DTensorSpec found in args/kwargs for {func}")

        # Set schema_info so the LRU cache key includes static args
        op_schema = OpSchema(func, args_schema, kwargs_schema)
        schema_info = self.sharding_prop.op_to_schema_info.get(func)
        if schema_info is None:
            schema_info = (
                self.sharding_prop.op_to_schema_info_for_single_dim_strategy.get(func)
            )
        if schema_info is not None:
            op_schema.schema_info = schema_info
            op_schema._recompute_comparison_key()

        if _are_we_tracing():
            output_sharding = self.sharding_prop.propagate_op_sharding_non_cached(
                op_schema
            )
        else:
            output_sharding = self.sharding_prop.propagate_op_sharding(op_schema)

        if (
            output_sharding.needs_redistribute  # pyrefly: ignore [missing-attribute]
            and (
                redistribute_schema
                := output_sharding.redistribute_schema  # pyrefly: ignore [missing-attribute]
            )
            is not None
        ):
            # a pure .needs_redistribute check is too broad; we want to ban redistribution,
            # but this flag is set for view ops that convert global shape -> local shape args.
            # During decomposition tracing on meta tensors at global shape, the shape adjustment
            # is irrelevant — only reject true redistribution.
            for orig, desired in zip(
                op_schema.args_spec,
                redistribute_schema.args_spec,  # pyrefly: ignore [missing-attribute]
            ):
                if orig.placements != desired.placements:
                    raise RuntimeError(
                        f"Decomposition requires redistribution for {func}"
                    )

        out = func(*args, **kwargs)
        # pyrefly: ignore [missing-attribute]
        self._record_output_specs(out, output_sharding.output_spec)
        return out

    def _record_output_specs(self, output: Any, output_spec: DTensorSpec | Any) -> None:
        if isinstance(output, torch.Tensor) and output_spec is not None:
            output._spec = output_spec  # pyrefly: ignore [missing-attribute]
        elif isinstance(output, (tuple, list)) and isinstance(
            output_spec, (tuple, list)
        ):
            for t, s in zip(output, output_spec):
                self._record_output_specs(t, s)


class DecompShardingStrategy:
    """
    Generates sharding strategies for ops by tracing through their decompositions.

    For each candidate input placement combination, runs the decomposition on meta
    tensors under PlacementTrackingMode to determine the output placement. These
    single-dimension strategies are then expanded to the full mesh.
    """

    def __init__(self, sharding_prop: ShardingPropagator):
        self.sharding_prop = sharding_prop
        # Cache fake meshes per device type to avoid repeated allocation.
        # A fake size-1 mesh ensures identical strategy computation across all ranks
        # during decomposition tracing, avoiding potential SPMD divergence.
        # False negatives are avoided (all sizes % 1 == 0), while false positives
        # are caught on expansion to the real, multi-dim device mesh.
        self._fake_meshes: dict[str, DeviceMesh] = {}

    def _get_fake_mesh(self, device_type: str) -> DeviceMesh:
        fake_mesh = self._fake_meshes.get(device_type)
        if fake_mesh is None:
            fake_mesh = DeviceMesh(device_type, [0], _init_backend=False, _rank=0)
            self._fake_meshes[device_type] = fake_mesh
        return fake_mesh

    @staticmethod
    def has_decomp(op: OpOverload) -> bool:
        # Check if op has a decomposition (explicit or CIA)
        return op in decomposition_table or op._can_decompose()

    def ensure_schema_info(self, op: OpOverload) -> None:
        """
        Register schema_info for decomposition op on first invocation.
        Needed for correct shard prop cache key.
        """
        if op not in self.sharding_prop.op_to_schema_info:
            schema_info = _infer_schema_info_from_op(op)
            self.sharding_prop.op_to_schema_info[op] = schema_info

    def propagate_strategy(
        self,
        op_schema: OpSchema,
    ) -> OpStrategy | None:
        if not tree_any(
            lambda x: isinstance(x, DTensorSpec),
            (op_schema.args_schema, op_schema.kwargs_schema),
        ):
            return None

        candidate_placements = self._get_candidate_placements(op_schema)
        mesh = try_find_mesh_from_args(
            op_schema.op,
            op_schema.args_schema + tuple(op_schema.kwargs_schema.values()),
        )

        fake_mesh = self._get_fake_mesh(mesh.device_type)
        single_dim_strategies = []
        output_placements: list[Placement | tuple[Placement, ...]] = []
        for input_placements in candidate_placements:
            try:
                output = self._propagate_through_decomp(
                    op_schema,
                    input_placements,
                    fake_mesh,
                )
            except NotImplementedError:
                return None
            except GuardOnDataDependentSymNode:
                return None
            except (RuntimeError, KeyError, IndexError):
                # TODO(pianpwk): RuntimeError is raised when redistribution is detected; switch to a custom error type
                # Runtime/KeyError/IndexError can also occur in view ops
                continue

            output_placements = (
                [output] if not isinstance(output, tuple) else list(output)
            )
            single_dim_strategies.append(output_placements + list(input_placements))

        if not single_dim_strategies:
            raise AssertionError(
                "Sharding propagation should have produced at least Replicate() strategy"
            )

        n_outputs = len(output_placements)
        strategy_schema = self.sharding_prop._wrap_with_op_strategy(op_schema)
        # Import here to avoid circular import at module load time
        from torch.distributed.tensor._ops.utils import (  # noqa: F811
            expand_to_full_mesh_op_strategy,
        )

        return expand_to_full_mesh_op_strategy(
            mesh, strategy_schema, single_dim_strategies, input_index=n_outputs
        )

    def _propagate_through_decomp(
        self,
        op_schema: OpSchema,
        placement: tuple[Placement | None],
        mesh: DeviceMesh,
    ) -> Placement | tuple[Placement, ...]:
        op = op_schema.op
        if op in decomposition_table:
            decomp_fn = decomposition_table[op]
        elif op._can_decompose():
            decomp_fn = op.decompose
        else:
            raise NotImplementedError(f"No decomposition found for {op}")

        placement_iter = iter(placement)

        def to_meta(x):
            p = next(placement_iter)
            if isinstance(x, DTensorSpec):
                # pyrefly: ignore [missing-attribute]
                meta = torch.empty(x.shape, dtype=x.tensor_meta.dtype, device="meta")
                # pyrefly: ignore [missing-attribute]
                meta._spec = DTensorSpec(mesh, (p,), tensor_meta=x.tensor_meta)
                return meta
            return x

        # Disable LocalTensorMode during decomposition tracing to prevent
        # interference with meta tensor operations
        from torch.distributed._local_tensor import maybe_disable_local_tensor_mode

        with maybe_disable_local_tensor_mode():
            # Create meta tensors and run decomposition outside LocalTensorMode
            args_meta = tree_map(to_meta, op_schema.args_schema)
            kwargs_meta = tree_map(to_meta, op_schema.kwargs_schema)

            with PlacementTrackingMode(self.sharding_prop, mesh):
                output = decomp_fn(*args_meta, **kwargs_meta)

        def get_placement(t):
            if isinstance(t, torch.Tensor):
                spec = getattr(t, "_spec", None)
                return spec.placements[0] if spec else None
            return None

        result = tree_map(get_placement, output)
        if isinstance(result, (tuple, list)):
            flat = [p for p in result if p is not None]
            return flat[0] if len(flat) == 1 else tuple(flat)
        return result

    @staticmethod
    def _get_candidate_placements(
        op_schema: OpSchema,
    ) -> list[tuple[Placement | None]]:
        tensor_specs = _extract_input_specs(op_schema)
        flat_specs, _ = tree_flatten(list(tensor_specs))

        # Step 1: Collect unique placements across all DTensorSpec inputs
        all_placements: set[Placement] = {Replicate()}
        tree_map_only(
            DTensorSpec,
            lambda spec: all_placements.update(spec.placements),
            flat_specs,
        )

        # Step 2: For each input, use the placement set, but expand Shard/StridedShard to all tensor dims
        candidates: list[list[Placement | None]] = []
        for spec in flat_specs:
            if not isinstance(spec, DTensorSpec):
                candidates.append([None])
            else:
                options = set(all_placements)
                for p in all_placements:
                    if isinstance(p, _StridedShard):
                        options |= {
                            _StridedShard(i, split_factor=p.split_factor)
                            for i in range(spec.ndim)
                        }
                    elif isinstance(p, Shard):
                        options |= {Shard(i) for i in range(spec.ndim)}
                candidates.append(list(options))

        # pyrefly: ignore [no-matching-overload]
        return list(itertools.product(*candidates))
