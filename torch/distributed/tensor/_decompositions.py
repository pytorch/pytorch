# mypy: allow-untyped-defs

import itertools

import torch
from torch._decomp import decomposition_table
from torch._ops import OpOverload
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import OpSchema, OpStrategy
from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy
from torch.distributed.tensor._sharding_prop import ShardingPropagator
from torch.distributed.tensor._utils import (
    ImplicitRedistributionError,
    try_find_mesh_from_args,
)
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.fx.experimental.proxy_tensor import get_isolated_graphmodule
from torch.utils._pytree import tree_any, tree_flatten, tree_map_only, tree_unflatten


# figure out how to properly cache
def _trace_decomposition(op_schema: OpSchema) -> torch.fx.GraphModule:
    from torch._guards import detect_fake_mode
    from torch._subclasses.fake_tensor import FakeTensorMode

    decomp_fn = decomposition_table[op_schema.op]
    fake_mode = detect_fake_mode() or FakeTensorMode()
    with fake_mode:
        args = op_schema.gen_fake_args()
        kwargs = op_schema.gen_fake_kwargs()
        return get_isolated_graphmodule(decomp_fn, args, kwargs)


def _extract_input_specs(op_schema: OpSchema) -> tuple[DTensorSpec | object, ...]:
    return op_schema.args_schema + tuple(op_schema.kwargs_schema.values())


class DecompShardingInterpreter(torch.fx.Interpreter):
    def __init__(
        self,
        module: torch.fx.GraphModule,
        mesh_device: str,
        sharding_prop: ShardingPropagator,
    ):
        super().__init__(module)
        self.sharding_prop = sharding_prop

        # Build single mesh dim strategies using fake 1d mesh
        self.mesh = DeviceMesh(mesh_device, torch.arange(2))

    def call_function(self, target, args, kwargs):
        if not isinstance(target, OpOverload):
            return super().call_function(target, args, kwargs)

        node_schema = OpSchema(target, args, kwargs)
        sharding = self.sharding_prop.propagate_op_sharding(node_schema)
        if sharding.needs_redistribute:  # type: ignore[possibly-undefined]
            raise ImplicitRedistributionError("decomposition required redistribute")
        return sharding.output_spec  # type: ignore[possibly-undefined]

    def output(self, target, args, kwargs):
        result = args[0]
        return tree_map_only(DTensorSpec, lambda s: s.placements[0], result)  # type: ignore[possibly-undefined]

    def run(self, *args, **kwargs):
        return super().run(*args, **kwargs)


class DecompShardingStrategy:
    @classmethod
    def has_decomp(cls, op: OpOverload) -> bool:
        return op in decomposition_table

    @classmethod
    def propagate_strategy(
        cls, op_schema: OpSchema, sharding_prop: ShardingPropagator
    ) -> OpStrategy | None:
        if not tree_any(
            lambda x: isinstance(x, DTensorSpec),
            (op_schema.args_schema, op_schema.kwargs_schema),
        ):
            # this case means we likely recursively traced a decomposition,
            # into a factory method that takes no DTensor args (e.g. torch.ones).
            # Just error out saying no sharding strategy is registered.
            return None

        graph = _trace_decomposition(op_schema)
        placements = cls._get_candidate_placements(graph, op_schema)
        mesh = try_find_mesh_from_args(
            op_schema.op,
            op_schema.args_schema + tuple(op_schema.kwargs_schema.values()),
        )

        single_dim_strategies = []
        interp = DecompShardingInterpreter(graph, mesh.device_type, sharding_prop)
        for placement in placements:
            try:
                output = cls._propagate_through_decomp(interp, placement, op_schema)
            except NotImplementedError:
                # immediately return; some op doesn't have a sharding strategy.
                return None
            except (ImplicitRedistributionError, RuntimeError):
                # I feel like we shouldn't have to catch RuntimeErrors?
                # But seeing this with view strategies.
                # If redistribution found, just skip this placement.
                continue

            output_placements = (
                [output] if not isinstance(output, tuple) else list(output)
            )
            placement_list = output_placements + list(placement)
            single_dim_strategies.append(placement_list)

        if not single_dim_strategies:
            raise AssertionError(
                "Sharding propagation should have at least produced the full Replicate() strategy"
            )

        n_outputs = len(output_placements)  # type: ignore[possibly-undefined]
        strategy_schema = sharding_prop._wrap_with_op_strategy(op_schema)
        return expand_to_full_mesh_op_strategy(
            mesh,
            strategy_schema,
            single_dim_strategies,
            input_index=n_outputs,
        )

    @classmethod
    def _propagate_through_decomp(
        cls,
        interpreter: DecompShardingInterpreter,
        placement: tuple[Placement | None],
        op_schema: OpSchema,
    ) -> list[Placement]:
        tensor_specs = _extract_input_specs(op_schema)
        flat_specs, spec = tree_flatten(list(tensor_specs))
        input_specs_flat = [
            DTensorSpec(interpreter.mesh, (p,), tensor_meta=s.tensor_meta)  # type: ignore[arg-type]
            if isinstance(s, DTensorSpec)
            else s
            for s, p in zip(flat_specs, placement)
        ]
        input_specs = tree_unflatten(input_specs_flat, spec)

        return interpreter.run(*input_specs)

    @classmethod
    def _get_candidate_placements(
        cls, graph: torch.fx.GraphModule, op_schema: OpSchema
    ) -> list[tuple[Placement | None]]:
        tensor_specs = _extract_input_specs(op_schema)
        placeholders = [n for n in graph.graph.nodes if n.op == "placeholder"]

        # Flatten specs to match placeholders
        flat_specs, _ = tree_flatten(list(tensor_specs))
        if len(placeholders) != len(flat_specs):
            raise AssertionError(
                f"Expected {len(placeholders)} placeholders, but got {len(flat_specs)} specs"
            )

        candidates: list[list[Placement] | list[None]] = []
        for spec, node in zip(flat_specs, placeholders):
            if not isinstance(spec, DTensorSpec):
                candidates.append([None])
                continue

            options: set[Placement] = {Replicate()}
            for placement in spec.placements:
                if isinstance(placement, (Shard, _StridedShard)):
                    for i in range(spec.ndim):
                        options.add(
                            Shard(i)
                            if isinstance(placement, Shard)
                            else _StridedShard(i, split_factor=placement.split_factor)
                        )
                elif isinstance(placement, Partial):
                    options.add(placement)
            candidates.append(list(options))

        return list(itertools.product(*candidates))  # type: ignore[arg-type]
