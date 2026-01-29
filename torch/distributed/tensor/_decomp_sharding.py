# mypy: allow-untyped-defs
"""Decomposition-based sharding propagation for DTensor."""

import logging

import torch
import torch.fx as fx
from torch._ops import OpOverload
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSchema, OpSpec, OpStrategy
from torch.distributed.tensor._utils import ExplicitRedistributionContext
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.fx.experimental.proxy_tensor import make_fx

logger = logging.getLogger(__name__)


class ConstructorDetectedError(Exception):
    pass


class ExplicitRedistributionError(Exception):
    pass


class DecompShardingInterpreter(fx.Interpreter):
    """Interpreter that propagates DTensorSpecs through a decomposition graph."""

    def __init__(self, module, input_specs, propagator):
        super().__init__(module)
        self.input_specs = input_specs
        self.propagator = propagator
        self.input_idx = 0

    def placeholder(self, target, args, kwargs):
        """Return the corresponding input DTensorSpec."""
        spec = self.input_specs[self.input_idx]
        self.input_idx += 1
        return spec

    def call_function(self, target, args, kwargs):
        """Propagate sharding through the operation."""
        # Skip non-operator targets (like namedtuple constructors)
        if not isinstance(target, OpOverload):
            # Pass through - construct tuple from args
            return tuple(args) if len(args) > 1 else args[0]

        # Build OpSchema from args/kwargs (which are now DTensorSpecs)
        node_schema = OpSchema(target, args_schema=args, kwargs_schema=kwargs)

        # Propagate sharding (hits shard prop cache!)
        output_spec = self.propagator.propagate_op_sharding_non_cached(node_schema)
        return output_spec.output_spec

    def output(self, target, args, kwargs):
        """Extract placements from the final output."""
        result = args[0]
        if isinstance(result, DTensorSpec):
            return result.placements[0]
        elif isinstance(result, tuple):
            # Multi-output case
            return tuple(s.placements[0] if s else None for s in result)
        return None

    def run(self):
        """Run the interpreter and return output placements."""
        return super().run()


def _extract_tensor_specs(op_schema: OpSchema) -> list:
    """Extract all DTensorSpecs from OpSchema args and kwargs."""
    from torch.utils._pytree import tree_map, tree_flatten

    def extract_spec(arg):
        return arg if isinstance(arg, DTensorSpec) else None

    specs_args = tree_map(extract_spec, op_schema.args_schema)
    specs_kwargs = tree_map(extract_spec, op_schema.kwargs_schema)

    all_specs, _ = tree_flatten([specs_args, specs_kwargs])
    return [s for s in all_specs if s is not None]


def _trace_decomposition(op: OpOverload, op_schema: OpSchema) -> fx.GraphModule:
    from torch._decomp import decomposition_table
    from torch._subclasses.fake_tensor import FakeTensorMode

    decomp_fn = decomposition_table[op]
    fake_mode = torch._guards.detect_fake_mode() or FakeTensorMode()

    with fake_mode:
        args = op_schema.gen_fake_args()
        kwargs = op_schema.gen_fake_kwargs()
        wrapped = lambda *a: decomp_fn(*a, **kwargs)
        return make_fx(wrapped, tracing_mode="fake")(*args)


class DecompositionShardingStrategy:

    @classmethod
    def can_handle(cls, op: OpOverload) -> bool:
        from torch._decomp import decomposition_table
        return op in decomposition_table

    @classmethod
    def generate_strategy(
        cls, op: OpOverload, op_schema: OpSchema, propagator
    ) -> OpStrategy:
        graph = _trace_decomposition(op, op_schema)
        candidates = cls._get_candidate_placements(graph, op_schema)

        # Build single mesh dim strategies (for first dimension only)
        single_mesh_dim_strategies = []

        for placement in candidates:
            try:
                output = cls._propagate_through_decomp(
                    graph, placement, op_schema, propagator
                )

                # Build PlacementList: [output_placement(s), input_placements...]
                if isinstance(output, tuple):
                    # Multi-output: flatten tuple
                    output_placements = list(output)
                else:
                    output_placements = [output]

                placement_list = output_placements + list(placement)
                single_mesh_dim_strategies.append(placement_list)
            except (ConstructorDetectedError, ExplicitRedistributionError, Exception):
                continue

        if not single_mesh_dim_strategies:
            raise NotImplementedError(f"Decomposition for {op} produced no valid strategies")

        # Expand to full mesh dimensions using standard utility
        # This handles multi-dim mesh AND computes proper redistribute costs
        from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy

        tensor_specs = _extract_tensor_specs(op_schema)

        # Determine number of outputs from the first strategy
        num_outputs = len(single_mesh_dim_strategies[0]) - len(tensor_specs)

        # Wrap OpSchema to have OpStrategy in args/kwargs for expand_to_full_mesh_op_strategy
        strategy_schema = propagator._wrap_with_op_strategy(op_schema)

        return expand_to_full_mesh_op_strategy(
            tensor_specs[0].mesh,
            strategy_schema,
            single_mesh_dim_strategies,
            input_index=num_outputs,
        )

    @classmethod
    def _get_candidate_placements(cls, graph, op_schema):
        from torch.distributed.tensor._ops.single_dim_strategy import (
            _get_unique_placements,
        )

        specs = [arg for arg in op_schema.args_schema if isinstance(arg, DTensorSpec)]
        if not specs:
            return []

        # Get all unique placement types from inputs
        unique_placements = _get_unique_placements(op_schema)
        placement_types = {type(p) for p in unique_placements}

        # Build candidate placements per input
        placements_per_input = []
        for spec in specs:
            placements = [Replicate()]

            # Add sharding candidates for each tensor dimension
            if Shard in placement_types:
                for dim in range(len(spec.tensor_meta.shape)):
                    placements.append(Shard(dim))

            # Add StridedShard with each unique split_factor
            strided_shards = [p for p in unique_placements if isinstance(p, _StridedShard)]
            unique_split_factors = {s.split_factor for s in strided_shards}
            for split_factor in unique_split_factors:
                for dim in range(len(spec.tensor_meta.shape)):
                    placements.append(_StridedShard(dim, split_factor=split_factor))

            # Add Partial with each unique reduce_op
            partials = [p for p in unique_placements if isinstance(p, Partial)]
            unique_reduce_ops = {p.reduce_op for p in partials}
            for reduce_op in unique_reduce_ops:
                placements.append(Partial(reduce_op))

            placements_per_input.append(placements)

        from itertools import product

        return list(product(*placements_per_input))

    @classmethod
    def _propagate_through_decomp(cls, graph_module, placements, op_schema, propagator):
        # Extract tensor specs from original op_schema
        tensor_specs = _extract_tensor_specs(op_schema)
        if not tensor_specs:
            return Replicate()

        mesh = tensor_specs[0].mesh

        # Build DTensorSpecs for inputs with candidate placements
        input_specs = [
            DTensorSpec(mesh, (p,), s.tensor_meta)
            for s, p in zip(tensor_specs, placements)
        ]

        # Create interpreter and run it
        with ExplicitRedistributionContext(enable=True, mode="raise"):
            try:
                interp = DecompShardingInterpreter(graph_module, input_specs, propagator)
                return interp.run()
            except NotImplementedError as e:
                # If any op in decomposition has no sharding rule, it's likely a constructor
                raise ConstructorDetectedError(
                    f"Constructor or unhandled op in decomposition"
                ) from e
            except RuntimeError as e:
                # ExplicitRedistributionContext raises RuntimeError
                if "ExplicitRedistributionContext" in str(e) or "redistribute" in str(
                    e
                ).lower():
                    raise ExplicitRedistributionError() from e
                raise
