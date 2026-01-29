# ShRT Implementation Plan

## Overview

The Sharding Rule Tester (ShRT) is an automated system for validating DTensor sharding rules. It exhaustively tests all placement combinations against ground truth computed via direct execution.

## Key Design Decisions

1. **OpInfo-only**: Only test operators that have OpInfo entries in torch's op database. Skip operators without OpInfo (they can be added later).

2. **Exhaustive enumeration**: Test ALL input/output placement combinations, including incorrect ones. This generates ground truth by execution, which we compare against the strategy's claims.

3. **Include Partial placements**: Test `Partial(SUM)` and `Partial(AVG)` for inputs to validate reduction-related sharding rules.

4. **Single-dim focus**: Test on 1-D meshes only. Multi-dim correctness follows from single-dim correctness for properly composed rules.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              ShRT System                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────┐                                                      │
│  │  Op Discovery  │  Find ops with both OpInfo AND registered strategy  │
│  └───────┬────────┘                                                      │
│          │                                                               │
│          ▼                                                               │
│  ┌────────────────┐                                                      │
│  │ Sample Inputs  │  Get valid tensor inputs from OpInfo                 │
│  └───────┬────────┘                                                      │
│          │                                                               │
│          ▼                                                               │
│  ┌────────────────┐  Enumerate ALL combinations:                         │
│  │   Placement    │  - Replicate                                         │
│  │  Enumeration   │  - Shard(dim) for each valid dim                     │
│  └───────┬────────┘  - Partial(SUM), Partial(AVG) for inputs             │
│          │                                                               │
│          ▼                                                               │
│  ┌────────────────┐                                                      │
│  │  Ground Truth  │  Execute each combination, determine if correct      │
│  │   Generator    │  by comparing local outputs to sharded reference     │
│  └───────┬────────┘                                                      │
│          │                                                               │
│          ▼                                                               │
│  ┌────────────────┐                                                      │
│  │   Strategy     │  Query strategy function for each combination        │
│  │    Checker     │  Compare strategy answer vs ground truth             │
│  └───────┬────────┘                                                      │
│          │                                                               │
│          ▼                                                               │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐              │
│  │    Database    │  │     XFail      │  │   Test Report  │              │
│  │    (Cache)     │  │   Registry     │  │   Generator    │              │
│  └────────────────┘  └────────────────┘  └────────────────┘              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
torch/distributed/tensor/_testing/
├── __init__.py                    # Public API exports
├── shrt/
│   ├── __init__.py
│   ├── runner.py                  # Main test orchestration
│   ├── discovery.py               # Op discovery (OpInfo + registry matching)
│   ├── enumeration.py             # Placement combination enumeration
│   ├── ground_truth.py            # Execute combinations, compute correctness
│   ├── strategy_checker.py        # Query strategies, compare to ground truth
│   ├── database.py                # On-disk cache management
│   ├── xfail.py                   # XFAIL registry
│   ├── report.py                  # Test result reporting
│   └── types.py                   # Shared data types
├── cli.py                         # Developer CLI tools
└── data/
    ├── shrt_cache.yaml            # Cached validation results
    └── shrt_xfails.yaml           # Known failures

test/distributed/tensor/
└── test_shrt.py                   # CI test entry point
```

---

## Implementation Phases

### Phase 1: Core Infrastructure

**Files to create:**
- `types.py` - Data structures
- `discovery.py` - Op discovery
- `enumeration.py` - Placement enumeration

#### 1.1 Data Types (`types.py`)

```python
from dataclasses import dataclass, field
from typing import Callable, Literal
from torch.distributed.tensor.placement_types import Placement, Shard, Replicate, Partial
from torch._ops import OpOverload
from torch.testing._internal.opinfo.core import OpInfo, SampleInput

@dataclass
class PlacementSpec:
    """Serializable representation of a placement."""
    type: Literal["Replicate", "Shard", "Partial"]
    dim: int | None = None           # For Shard
    reduce_op: str | None = None     # For Partial: "sum", "avg"

    def to_placement(self) -> Placement:
        if self.type == "Replicate":
            return Replicate()
        elif self.type == "Shard":
            return Shard(self.dim)
        elif self.type == "Partial":
            from torch.distributed.tensor._ops._math_ops import (
                _PartialOp,  # or appropriate import
            )
            # Map string to reduce op
            reduce_ops = {"sum": ReduceOp.SUM, "avg": ReduceOp.AVG}
            return Partial(reduce_ops[self.reduce_op])

    @classmethod
    def from_placement(cls, p: Placement) -> "PlacementSpec":
        if isinstance(p, Replicate):
            return cls(type="Replicate")
        elif isinstance(p, Shard):
            return cls(type="Shard", dim=p.dim)
        elif isinstance(p, Partial):
            return cls(type="Partial", reduce_op=str(p.reduce_op).lower())
        raise ValueError(f"Unknown placement type: {type(p)}")

@dataclass
class PlacementCombination:
    """A complete specification of input and output placements."""
    input_placements: list[PlacementSpec]   # One per tensor input
    output_placements: list[PlacementSpec]  # One per tensor output

    def to_tuple(self) -> tuple:
        """For hashing/comparison."""
        return (
            tuple((p.type, p.dim, p.reduce_op) for p in self.input_placements),
            tuple((p.type, p.dim, p.reduce_op) for p in self.output_placements),
        )

    def __hash__(self):
        return hash(self.to_tuple())

    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()

@dataclass
class TestableOp:
    """An operator that can be tested by ShRT."""
    op_info: OpInfo
    aten_overload: OpOverload
    strategy_type: Literal["single_dim", "full", "rule"]
    strategy_func: Callable

@dataclass
class GroundTruthResult:
    """Result of executing a placement combination."""
    combination: PlacementCombination
    is_valid: bool                    # Does this combination produce correct results?
    error: str | None = None          # Error message if execution failed

@dataclass
class StrategyResult:
    """What the strategy says about a placement combination."""
    combination: PlacementCombination
    strategy_accepts: bool            # Does strategy say this is valid?
    redistribution_required: bool     # Does strategy require redistribution?

@dataclass
class ValidationResult:
    """Comparison of ground truth vs strategy."""
    op_name: str
    combination: PlacementCombination
    ground_truth: GroundTruthResult
    strategy: StrategyResult

    @property
    def status(self) -> Literal["pass", "false_positive", "false_negative", "error"]:
        """
        - pass: strategy matches ground truth
        - false_positive: strategy accepts but execution fails
        - false_negative: strategy rejects but execution would succeed
        - error: execution error (neither valid nor cleanly invalid)
        """
        if self.ground_truth.error and "expected" not in self.ground_truth.error.lower():
            return "error"
        if self.strategy.strategy_accepts == self.ground_truth.is_valid:
            return "pass"
        if self.strategy.strategy_accepts and not self.ground_truth.is_valid:
            return "false_positive"
        return "false_negative"

@dataclass
class TestReport:
    """Aggregated test results."""
    passed: list[ValidationResult] = field(default_factory=list)
    false_positives: list[ValidationResult] = field(default_factory=list)
    false_negatives: list[ValidationResult] = field(default_factory=list)
    errors: list[ValidationResult] = field(default_factory=list)
    xfailed: list[tuple[ValidationResult, "XFail"]] = field(default_factory=list)
    skipped: list[tuple[TestableOp, PlacementCombination]] = field(default_factory=list)
```

#### 1.2 Op Discovery (`discovery.py`)

```python
from torch.testing._internal.opinfo.definitions import op_db
from torch.distributed.tensor._api import DTensor

def get_sharding_propagator():
    """Get the ShardingPropagator instance."""
    # Ensure DTensor ops are registered
    import torch.distributed.tensor._ops  # noqa: F401
    return DTensor._op_dispatcher.sharding_propagator

def discover_testable_ops(
    strategy_types: set[str] = {"single_dim", "full", "rule"},
) -> list[TestableOp]:
    """
    Find all operators that have:
    1. An OpInfo entry (for sample inputs)
    2. A registered sharding strategy

    Returns list of TestableOp instances.
    """
    propagator = get_sharding_propagator()

    # Build mapping: op base name -> list of OpInfo
    # e.g., "add" -> [OpInfo for torch.add, OpInfo for torch.Tensor.add, ...]
    opinfo_by_name: dict[str, list[OpInfo]] = {}
    for op_info in op_db:
        name = op_info.aten_name or op_info.name
        opinfo_by_name.setdefault(name, []).append(op_info)

    # Collect registered strategies
    registries = []
    if "single_dim" in strategy_types:
        registries.append(("single_dim", propagator.op_single_dim_strategy_funcs))
    if "full" in strategy_types:
        registries.append(("full", propagator.op_strategy_funcs))
    if "rule" in strategy_types:
        registries.append(("rule", propagator.op_to_rules))

    testable_ops = []
    seen = set()  # Avoid duplicates

    for strategy_type, registry in registries:
        for op_overload, strategy_func in registry.items():
            # Extract base op name from overload
            # e.g., "aten::add.Tensor" -> "add"
            full_name = op_overload.name()  # e.g., "add.Tensor"
            base_name = full_name.split(".")[0]

            if base_name not in opinfo_by_name:
                continue  # No OpInfo available, skip

            for op_info in opinfo_by_name[base_name]:
                key = (op_overload, id(op_info))
                if key in seen:
                    continue
                seen.add(key)

                testable_ops.append(TestableOp(
                    op_info=op_info,
                    aten_overload=op_overload,
                    strategy_type=strategy_type,
                    strategy_func=strategy_func,
                ))

    # Sort for deterministic ordering
    testable_ops.sort(key=lambda x: (x.aten_overload.name(), x.op_info.name))
    return testable_ops

def filter_ops(
    ops: list[TestableOp],
    include_names: list[str] | None = None,
    exclude_names: list[str] | None = None,
) -> list[TestableOp]:
    """Filter testable ops by name patterns."""
    result = ops
    if include_names:
        result = [op for op in result if any(
            n in op.aten_overload.name() for n in include_names
        )]
    if exclude_names:
        result = [op for op in result if not any(
            n in op.aten_overload.name() for n in exclude_names
        )]
    return result
```

#### 1.3 Placement Enumeration (`enumeration.py`)

```python
from itertools import product
from torch.distributed.tensor.placement_types import Shard, Replicate, Partial
import torch.distributed as dist

def get_all_placements_for_tensor(
    shape: torch.Size,
    is_input: bool,
    include_partial: bool = True,
) -> list[PlacementSpec]:
    """
    Generate all valid placements for a tensor with given shape.

    For inputs:
    - Replicate
    - Shard(dim) for each dim in range(ndim)
    - Partial(sum), Partial(avg) if include_partial

    For outputs:
    - Replicate
    - Shard(dim) for each dim in range(ndim)
    - (No Partial for outputs - outputs should be materialized)
    """
    placements = [PlacementSpec(type="Replicate")]

    ndim = len(shape)
    for dim in range(ndim):
        # Skip sharding on size-1 dimensions (no point)
        if shape[dim] > 1:
            placements.append(PlacementSpec(type="Shard", dim=dim))

    if is_input and include_partial:
        placements.append(PlacementSpec(type="Partial", reduce_op="sum"))
        placements.append(PlacementSpec(type="Partial", reduce_op="avg"))

    return placements

def enumerate_all_combinations(
    input_shapes: list[torch.Size],
    output_shapes: list[torch.Size],
    include_partial: bool = True,
) -> list[PlacementCombination]:
    """
    Generate ALL possible placement combinations for given input/output shapes.

    This is exhaustive - includes both valid and invalid combinations.
    We determine validity by execution, not by pre-filtering.
    """
    # Get possible placements for each input tensor
    input_placement_options = [
        get_all_placements_for_tensor(shape, is_input=True, include_partial=include_partial)
        for shape in input_shapes
    ]

    # Get possible placements for each output tensor
    output_placement_options = [
        get_all_placements_for_tensor(shape, is_input=False, include_partial=False)
        for shape in output_shapes
    ]

    combinations = []

    # Cartesian product of all input placements
    for input_combo in product(*input_placement_options):
        # Cartesian product of all output placements
        for output_combo in product(*output_placement_options):
            combinations.append(PlacementCombination(
                input_placements=list(input_combo),
                output_placements=list(output_combo),
            ))

    return combinations

def get_tensor_args_from_sample(
    sample_input: SampleInput,
) -> tuple[list[torch.Tensor], list[int]]:
    """
    Extract tensor arguments and their positions from a SampleInput.

    Returns:
    - List of tensor arguments
    - List of their positions in the full args list
    """
    all_args = [sample_input.input] + list(sample_input.args)

    tensors = []
    positions = []

    for i, arg in enumerate(all_args):
        if isinstance(arg, torch.Tensor):
            tensors.append(arg)
            positions.append(i)
        elif isinstance(arg, (list, tuple)):
            # Handle list of tensors (e.g., for cat, stack)
            for j, item in enumerate(arg):
                if isinstance(item, torch.Tensor):
                    tensors.append(item)
                    positions.append((i, j))  # Nested position

    return tensors, positions

def count_combinations(
    input_shapes: list[torch.Size],
    output_shapes: list[torch.Size],
    include_partial: bool = True,
) -> int:
    """Count total combinations without generating them."""
    input_counts = [
        len(get_all_placements_for_tensor(s, True, include_partial))
        for s in input_shapes
    ]
    output_counts = [
        len(get_all_placements_for_tensor(s, False, False))
        for s in output_shapes
    ]

    total = 1
    for c in input_counts + output_counts:
        total *= c
    return total
```

---

### Phase 2: Ground Truth Generation

**Files to create:**
- `ground_truth.py` - Execute combinations and determine correctness

#### 2.1 Ground Truth Generator (`ground_truth.py`)

```python
import torch
from torch.distributed.tensor._local_tensor import LocalTensor
from torch.distributed._fake_pg import FakeStore
from torch.distributed.device_mesh import DeviceMesh
import torch.distributed as dist

class GroundTruthGenerator:
    """
    Executes placement combinations to determine ground truth correctness.

    A combination is "correct" if:
    1. Each rank's local input (sharded according to input_placements) can be
       fed to the operator
    2. Each rank's local output matches the corresponding shard of the
       reference output (computed on full tensors)
    3. For Partial inputs: the sum/avg across ranks of local outputs equals
       the reference output
    """

    def __init__(self, world_size: int = 4, device: str = "cpu"):
        self.world_size = world_size
        self.device = device
        self._init_fake_pg()

    def _init_fake_pg(self):
        """Initialize fake process group if not already done."""
        if not dist.is_initialized():
            store = FakeStore()
            dist.init_process_group(
                backend="fake",
                rank=0,
                world_size=self.world_size,
                store=store,
            )
        self.mesh = DeviceMesh(self.device, torch.arange(self.world_size))

    def compute_reference_output(
        self,
        op: Callable,
        args: tuple,
        kwargs: dict,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Compute reference output using full (unsharded) tensors."""
        return op(*args, **kwargs)

    def shard_tensor(
        self,
        tensor: torch.Tensor,
        placement: PlacementSpec,
        rank: int,
    ) -> torch.Tensor:
        """
        Shard a tensor according to placement for a given rank.

        - Replicate: return full tensor
        - Shard(dim): return the rank's slice
        - Partial: return tensor / world_size (simulating partial sum state)
        """
        if placement.type == "Replicate":
            return tensor.clone()

        elif placement.type == "Shard":
            dim = placement.dim
            size = tensor.size(dim)
            chunk_size = (size + self.world_size - 1) // self.world_size
            start = rank * chunk_size
            end = min(start + chunk_size, size)

            # Handle case where this rank has no data
            if start >= size:
                # Return empty tensor with correct shape
                new_shape = list(tensor.shape)
                new_shape[dim] = 0
                return tensor.new_empty(new_shape)

            return tensor.narrow(dim, start, end - start).clone()

        elif placement.type == "Partial":
            # For Partial, each rank has the same tensor but represents
            # a partial contribution. For testing, we divide by world_size
            # so that sum across ranks equals the original.
            if placement.reduce_op == "sum":
                return tensor.clone() / self.world_size
            elif placement.reduce_op == "avg":
                # For avg, each rank has full value (avg of same values = value)
                return tensor.clone()

        raise ValueError(f"Unknown placement: {placement}")

    def unshard_output(
        self,
        local_outputs: dict[int, torch.Tensor],
        placement: PlacementSpec,
        reference_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Reconstruct full tensor from local outputs.

        - Replicate: all ranks should have same value, return any
        - Shard(dim): concatenate along dim
        - (Partial not used for outputs)
        """
        if placement.type == "Replicate":
            # All should be equal, return first
            return local_outputs[0]

        elif placement.type == "Shard":
            dim = placement.dim
            tensors = [local_outputs[r] for r in range(self.world_size)]
            return torch.cat(tensors, dim=dim)

        raise ValueError(f"Cannot unshard output with placement: {placement}")

    def validate_combination(
        self,
        op: Callable,
        args: tuple,
        kwargs: dict,
        combination: PlacementCombination,
        tensor_positions: list[int | tuple[int, int]],
    ) -> GroundTruthResult:
        """
        Test if a placement combination produces correct results.

        Steps:
        1. Compute reference output on full tensors
        2. For each rank, shard inputs according to input_placements
        3. Execute op on each rank's local inputs
        4. Verify local outputs match expected shards of reference
        """
        try:
            # Step 1: Reference output
            reference_output = self.compute_reference_output(op, args, kwargs)
            if not isinstance(reference_output, tuple):
                reference_output = (reference_output,)

            # Step 2 & 3: Execute on each rank
            local_outputs_per_rank = {}

            for rank in range(self.world_size):
                # Shard inputs for this rank
                local_args = self._shard_args(
                    args, combination.input_placements, tensor_positions, rank
                )

                # Execute
                try:
                    local_out = op(*local_args, **kwargs)
                    if not isinstance(local_out, tuple):
                        local_out = (local_out,)
                    local_outputs_per_rank[rank] = local_out
                except Exception as e:
                    # Op failed on this rank's inputs
                    return GroundTruthResult(
                        combination=combination,
                        is_valid=False,
                        error=f"Execution failed on rank {rank}: {e}",
                    )

            # Step 4: Verify outputs
            for out_idx, out_placement in enumerate(combination.output_placements):
                ref_out = reference_output[out_idx]

                # Collect local outputs for this output tensor
                local_outs = {
                    r: local_outputs_per_rank[r][out_idx]
                    for r in range(self.world_size)
                }

                # Verify each rank's output matches expected shard
                if not self._verify_outputs(
                    local_outs, out_placement, ref_out
                ):
                    return GroundTruthResult(
                        combination=combination,
                        is_valid=False,
                        error=f"Output {out_idx} mismatch for placement {out_placement}",
                    )

            return GroundTruthResult(
                combination=combination,
                is_valid=True,
            )

        except Exception as e:
            return GroundTruthResult(
                combination=combination,
                is_valid=False,
                error=f"Validation error: {e}",
            )

    def _shard_args(
        self,
        args: tuple,
        input_placements: list[PlacementSpec],
        tensor_positions: list[int | tuple[int, int]],
        rank: int,
    ) -> tuple:
        """Shard tensor arguments according to placements."""
        args = list(args)

        for placement, pos in zip(input_placements, tensor_positions):
            if isinstance(pos, int):
                args[pos] = self.shard_tensor(args[pos], placement, rank)
            else:
                # Nested position (list of tensors)
                outer, inner = pos
                lst = list(args[outer])
                lst[inner] = self.shard_tensor(lst[inner], placement, rank)
                args[outer] = type(args[outer])(lst)

        return tuple(args)

    def _verify_outputs(
        self,
        local_outputs: dict[int, torch.Tensor],
        placement: PlacementSpec,
        reference: torch.Tensor,
    ) -> bool:
        """Verify local outputs match expected shards of reference."""
        if placement.type == "Replicate":
            # All ranks should have output equal to reference
            for rank, local_out in local_outputs.items():
                if not torch.allclose(local_out, reference, rtol=1e-5, atol=1e-8):
                    return False
            return True

        elif placement.type == "Shard":
            # Each rank should have its shard of reference
            dim = placement.dim
            for rank in range(self.world_size):
                expected = self.shard_tensor(reference, placement, rank)
                actual = local_outputs[rank]

                if expected.shape != actual.shape:
                    return False
                if expected.numel() > 0 and not torch.allclose(
                    actual, expected, rtol=1e-5, atol=1e-8
                ):
                    return False
            return True

        return False
```

---

### Phase 3: Strategy Checker

**Files to create:**
- `strategy_checker.py` - Query strategies and compare to ground truth

#### 3.1 Strategy Checker (`strategy_checker.py`)

```python
from torch.distributed.tensor._op_schema import OpSchema, OpSpec, DTensorSpec
from torch.distributed.tensor._sharding_prop import ShardingPropagator
from torch.distributed.tensor.placement_types import Placement

class StrategyChecker:
    """
    Queries registered strategies to determine if they accept a placement combination.
    """

    def __init__(self, propagator: ShardingPropagator, mesh: DeviceMesh):
        self.propagator = propagator
        self.mesh = mesh

    def check_combination(
        self,
        op: TestableOp,
        combination: PlacementCombination,
        tensor_metas: list[TensorMeta],
    ) -> StrategyResult:
        """
        Query the strategy to see if it accepts this combination.

        For single-dim strategies:
        1. Build OpSchema with input DTensorSpecs using the combination's placements
        2. Call strategy function
        3. Check if any returned strategy matches the combination's output placements
        """
        try:
            # Build input specs
            input_specs = []
            for i, (placement, meta) in enumerate(
                zip(combination.input_placements, tensor_metas)
            ):
                dtensor_spec = DTensorSpec(
                    mesh=self.mesh,
                    placements=(placement.to_placement(),),
                    tensor_meta=meta,
                )
                input_specs.append(dtensor_spec)

            # Build OpSchema
            op_schema = OpSchema(
                op=op.aten_overload,
                args_schema=tuple(input_specs),
                kwargs_schema={},
            )

            # Query strategy
            if op.strategy_type == "single_dim":
                strategies = self._query_single_dim_strategy(op, op_schema)
            elif op.strategy_type == "full":
                strategies = self._query_full_strategy(op, op_schema)
            else:
                strategies = self._query_rule(op, op_schema)

            # Check if combination's output placements are in strategies
            expected_outputs = tuple(
                p.to_placement() for p in combination.output_placements
            )

            accepts = False
            requires_redistribution = False

            for strategy in strategies:
                if self._output_matches(strategy, expected_outputs):
                    accepts = True
                    requires_redistribution = strategy.get("redistribute", False)
                    break

            return StrategyResult(
                combination=combination,
                strategy_accepts=accepts,
                redistribution_required=requires_redistribution,
            )

        except Exception as e:
            # Strategy raised an error - treat as rejection
            return StrategyResult(
                combination=combination,
                strategy_accepts=False,
                redistribution_required=False,
            )

    def _query_single_dim_strategy(
        self,
        op: TestableOp,
        op_schema: OpSchema,
    ) -> list[dict]:
        """Query single-dim strategy function."""
        strategy_func = op.strategy_func

        # Single-dim strategies return list of [output_placements, *input_placements]
        # Need to expand for our mesh
        from torch.distributed.tensor._ops.single_dim_strategy import (
            _expand_single_dim_strategy_to_mesh,
        )

        expanded = _expand_single_dim_strategy_to_mesh(
            self.mesh,
            op_schema,
            strategy_func,
            # ... other params
        )

        raw_strategies = expanded(
            op.aten_overload,
            op_schema.args_meta,
            op_schema.kwargs_meta,
        )

        # Convert to standard format
        return self._parse_strategies(raw_strategies)

    def _query_full_strategy(
        self,
        op: TestableOp,
        op_schema: OpSchema,
    ) -> list[dict]:
        """Query full strategy function."""
        strategy_func = op.strategy_func
        op_strategy = strategy_func(op_schema)

        strategies = []
        for op_spec in op_strategy.strategies:
            strategies.append({
                "output_placements": op_spec.output_spec.placements,
                "input_placements": [s.placements for s in op_spec.input_specs],
                "redistribute": op_spec.redistribute_cost > 0,
            })

        return strategies

    def _output_matches(
        self,
        strategy: dict,
        expected_outputs: tuple[Placement, ...],
    ) -> bool:
        """Check if strategy's output placements match expected."""
        strategy_outputs = strategy.get("output_placements", ())
        if len(strategy_outputs) != len(expected_outputs):
            return False

        for s_out, e_out in zip(strategy_outputs, expected_outputs):
            if not self._placement_equal(s_out, e_out):
                return False
        return True

    def _placement_equal(self, p1: Placement, p2: Placement) -> bool:
        """Check if two placements are equivalent."""
        if type(p1) != type(p2):
            return False
        if isinstance(p1, Shard):
            return p1.dim == p2.dim
        if isinstance(p1, Partial):
            return p1.reduce_op == p2.reduce_op
        return True  # Replicate
```

---

### Phase 4: Database and XFail Systems

**Files to create:**
- `database.py` - Cache management
- `xfail.py` - Known failure tracking

#### 4.1 Database (`database.py`)

```python
import yaml
import hashlib
import inspect
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class CachedEntry:
    combination: PlacementCombination
    ground_truth_valid: bool
    strategy_accepts: bool
    status: str  # "pass", "false_positive", "false_negative", "error"
    error: str | None = None

@dataclass
class OpCache:
    op_name: str
    strategy_hash: str
    validated_at: str
    entries: list[CachedEntry]

class ShrtDatabase:
    """On-disk cache of validated combinations."""

    VERSION = 1

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, OpCache] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path) as f:
                raw = yaml.safe_load(f)
            if raw and raw.get("version") == self.VERSION:
                self.data = self._parse(raw.get("operators", {}))

    def _save(self):
        raw = {
            "version": self.VERSION,
            "last_updated": datetime.utcnow().isoformat(),
            "operators": self._serialize(self.data),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            yaml.dump(raw, f, default_flow_style=False, sort_keys=True)

    def is_cached(
        self,
        op_name: str,
        strategy_hash: str,
        combination: PlacementCombination,
    ) -> CachedEntry | None:
        """Check if combination is cached with matching strategy hash."""
        cache = self.data.get(op_name)
        if not cache or cache.strategy_hash != strategy_hash:
            return None

        for entry in cache.entries:
            if entry.combination == combination:
                return entry
        return None

    def cache_result(
        self,
        op_name: str,
        strategy_hash: str,
        result: ValidationResult,
    ):
        """Store a validation result."""
        if op_name not in self.data:
            self.data[op_name] = OpCache(
                op_name=op_name,
                strategy_hash=strategy_hash,
                validated_at=datetime.utcnow().isoformat(),
                entries=[],
            )

        cache = self.data[op_name]

        # Update strategy hash if changed
        if cache.strategy_hash != strategy_hash:
            cache.strategy_hash = strategy_hash
            cache.entries = []  # Invalidate old entries
            cache.validated_at = datetime.utcnow().isoformat()

        # Add entry
        cache.entries.append(CachedEntry(
            combination=result.combination,
            ground_truth_valid=result.ground_truth.is_valid,
            strategy_accepts=result.strategy.strategy_accepts,
            status=result.status,
            error=result.ground_truth.error,
        ))

    def clear_op(self, op_name: str):
        """Clear cache for a specific operator."""
        self.data.pop(op_name, None)
        self._save()

    def save(self):
        """Persist to disk."""
        self._save()

def hash_strategy(strategy_func: Callable) -> str:
    """Generate hash of strategy function source."""
    try:
        source = inspect.getsource(strategy_func)
        return hashlib.sha256(source.encode()).hexdigest()[:16]
    except (OSError, TypeError):
        # Can't get source, use function id
        return f"id_{id(strategy_func)}"
```

#### 4.2 XFail Registry (`xfail.py`)

```python
import yaml
from pathlib import Path
from dataclasses import dataclass
from datetime import date

@dataclass
class XFail:
    op: str
    placements: PlacementCombination | Literal["*"]  # "*" means all combinations
    reason: str
    issue: str | None = None
    added: str = ""
    expected_fix: str | None = None  # Target date for fix

class XFailRegistry:
    """Registry of known failing combinations."""

    VERSION = 1

    def __init__(self, path: Path):
        self.path = path
        self.xfails: list[XFail] = []
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path) as f:
                raw = yaml.safe_load(f)
            if raw and raw.get("version") == self.VERSION:
                self.xfails = self._parse(raw.get("xfails", []))

    def is_xfail(
        self,
        op_name: str,
        combination: PlacementCombination,
    ) -> XFail | None:
        """Check if combination is a known failure."""
        for xfail in self.xfails:
            if xfail.op != op_name:
                continue
            if xfail.placements == "*" or xfail.placements == combination:
                return xfail
        return None

    def add_xfail(
        self,
        op: str,
        placements: PlacementCombination | Literal["*"],
        reason: str,
        issue: str | None = None,
    ):
        """Add a new xfail entry."""
        self.xfails.append(XFail(
            op=op,
            placements=placements,
            reason=reason,
            issue=issue,
            added=date.today().isoformat(),
        ))
        self._save()

    def remove_xfail(self, op: str, placements: PlacementCombination | None = None):
        """Remove xfail entries for an operator."""
        self.xfails = [
            x for x in self.xfails
            if x.op != op or (placements and x.placements != placements)
        ]
        self._save()
```

---

### Phase 5: Test Runner and CLI

**Files to create:**
- `runner.py` - Main orchestration
- `report.py` - Result reporting
- `cli.py` - Developer tools
- `test_shrt.py` - CI entry point

#### 5.1 Main Runner (`runner.py`)

```python
class ShardingRuleTester:
    """Main test orchestration."""

    def __init__(
        self,
        world_size: int = 4,
        device: str = "cpu",
        cache_path: Path | None = None,
        xfail_path: Path | None = None,
        use_cache: bool = True,
        include_partial: bool = True,
    ):
        self.world_size = world_size
        self.device = device
        self.use_cache = use_cache
        self.include_partial = include_partial

        self.ground_truth = GroundTruthGenerator(world_size, device)
        self.database = ShrtDatabase(cache_path) if cache_path else None
        self.xfails = XFailRegistry(xfail_path) if xfail_path else None

        propagator = get_sharding_propagator()
        self.strategy_checker = StrategyChecker(propagator, self.ground_truth.mesh)

    def run_tests(
        self,
        ops: list[TestableOp] | None = None,
        strategy_types: set[str] = {"single_dim"},
        max_samples_per_op: int = 3,
        max_combinations_per_sample: int | None = None,
    ) -> TestReport:
        """
        Run ShRT tests.

        Args:
            ops: Specific ops to test, or None for all discovered ops
            strategy_types: Which strategy types to test
            max_samples_per_op: Limit sample inputs per operator
            max_combinations_per_sample: Limit combinations per sample (None = all)

        Returns:
            TestReport with all results
        """
        if ops is None:
            ops = discover_testable_ops(strategy_types)

        report = TestReport()

        for op in ops:
            if op.strategy_type not in strategy_types:
                continue

            strategy_hash = hash_strategy(op.strategy_func)

            # Get sample inputs
            try:
                samples = list(op.op_info.sample_inputs(
                    self.device, torch.float32, requires_grad=False
                ))[:max_samples_per_op]
            except Exception as e:
                # Can't generate samples, skip
                continue

            for sample in samples:
                self._test_sample(
                    op, sample, strategy_hash, report, max_combinations_per_sample
                )

        if self.database:
            self.database.save()

        return report

    def _test_sample(
        self,
        op: TestableOp,
        sample: SampleInput,
        strategy_hash: str,
        report: TestReport,
        max_combinations: int | None,
    ):
        """Test all placement combinations for a sample input."""
        # Extract tensor info
        tensors, positions = get_tensor_args_from_sample(sample)
        input_shapes = [t.shape for t in tensors]

        # Get output shape by running op
        args = (sample.input,) + sample.args
        kwargs = sample.kwargs
        try:
            output = op.op_info.op(*args, **kwargs)
            if isinstance(output, torch.Tensor):
                output_shapes = [output.shape]
            elif isinstance(output, tuple):
                output_shapes = [o.shape for o in output if isinstance(o, torch.Tensor)]
            else:
                return  # Non-tensor output, skip
        except Exception:
            return  # Op failed, skip

        # Enumerate combinations
        combinations = enumerate_all_combinations(
            input_shapes, output_shapes, self.include_partial
        )

        if max_combinations:
            combinations = combinations[:max_combinations]

        op_name = op.aten_overload.name()

        for combination in combinations:
            # Check cache
            if self.use_cache and self.database:
                cached = self.database.is_cached(op_name, strategy_hash, combination)
                if cached:
                    report.skipped.append((op, combination))
                    continue

            # Check xfail
            xfail = None
            if self.xfails:
                xfail = self.xfails.is_xfail(op_name, combination)

            # Compute ground truth
            ground_truth = self.ground_truth.validate_combination(
                op.op_info.op, args, kwargs, combination, positions
            )

            # Query strategy
            tensor_metas = [
                TensorMeta(shape=t.shape, stride=t.stride(), dtype=t.dtype)
                for t in tensors
            ]
            strategy = self.strategy_checker.check_combination(
                op, combination, tensor_metas
            )

            # Create result
            result = ValidationResult(
                op_name=op_name,
                combination=combination,
                ground_truth=ground_truth,
                strategy=strategy,
            )

            # Categorize
            if result.status == "pass":
                report.passed.append(result)
            elif xfail:
                report.xfailed.append((result, xfail))
            elif result.status == "false_positive":
                report.false_positives.append(result)
            elif result.status == "false_negative":
                report.false_negatives.append(result)
            else:
                report.errors.append(result)

            # Cache result
            if self.database:
                self.database.cache_result(op_name, strategy_hash, result)

        return report
```

#### 5.2 CI Test (`test_shrt.py`)

```python
# test/distributed/tensor/test_shrt.py
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.distributed.tensor._testing.shrt import (
    ShardingRuleTester,
    discover_testable_ops,
)
from pathlib import Path

class TestShardingRules(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache_path = Path(__file__).parent / "data" / "shrt_cache.yaml"
        cls.xfail_path = Path(__file__).parent / "data" / "shrt_xfails.yaml"

    def test_single_dim_strategies_correctness(self):
        """Validate all single-dim sharding strategies produce correct results."""
        tester = ShardingRuleTester(
            cache_path=self.cache_path,
            xfail_path=self.xfail_path,
            world_size=4,
        )

        report = tester.run_tests(strategy_types={"single_dim"})

        # Print summary
        print(f"\nShRT Results:")
        print(f"  Passed: {len(report.passed)}")
        print(f"  XFailed: {len(report.xfailed)}")
        print(f"  Skipped (cached): {len(report.skipped)}")
        print(f"  False Positives: {len(report.false_positives)}")
        print(f"  False Negatives: {len(report.false_negatives)}")
        print(f"  Errors: {len(report.errors)}")

        # Fail on false positives (strategy says valid but execution fails)
        if report.false_positives:
            msg = "\n\nFalse Positives (strategy accepts invalid combinations):\n"
            for r in report.false_positives[:10]:  # Limit output
                msg += f"  {r.op_name}: {r.combination}\n"
                msg += f"    Error: {r.ground_truth.error}\n"
            if len(report.false_positives) > 10:
                msg += f"  ... and {len(report.false_positives) - 10} more\n"
            self.fail(msg)

        # Warn on false negatives (strategy rejects valid combinations)
        # These are less severe - strategy is conservative
        if report.false_negatives:
            print(f"\nWarning: {len(report.false_negatives)} false negatives")
            print("(Strategy rejects combinations that would actually work)")

if __name__ == "__main__":
    run_tests()
```

#### 5.3 Developer CLI (`cli.py`)

```python
#!/usr/bin/env python
"""
ShRT Developer Tools

Usage:
    python -m torch.distributed.tensor._testing.cli status
    python -m torch.distributed.tensor._testing.cli run [--op NAME] [--no-cache]
    python -m torch.distributed.tensor._testing.cli update-cache [--op NAME]
    python -m torch.distributed.tensor._testing.cli add-xfail OP REASON [--issue URL]
    python -m torch.distributed.tensor._testing.cli remove-xfail OP
    python -m torch.distributed.tensor._testing.cli clear-cache [--op NAME]
"""

import argparse
from pathlib import Path

def cmd_status(args):
    """Show current ShRT status."""
    from .database import ShrtDatabase
    from .xfail import XFailRegistry
    from .discovery import discover_testable_ops

    ops = discover_testable_ops()
    print(f"Testable operators: {len(ops)}")

    by_type = {}
    for op in ops:
        by_type.setdefault(op.strategy_type, []).append(op)
    for stype, type_ops in sorted(by_type.items()):
        print(f"  {stype}: {len(type_ops)}")

    if args.cache.exists():
        db = ShrtDatabase(args.cache)
        total_cached = sum(len(c.entries) for c in db.data.values())
        print(f"\nCached validations: {total_cached}")

    if args.xfail.exists():
        xfails = XFailRegistry(args.xfail)
        print(f"XFails: {len(xfails.xfails)}")

def cmd_run(args):
    """Run ShRT tests."""
    from .runner import ShardingRuleTester
    from .discovery import discover_testable_ops, filter_ops

    ops = discover_testable_ops()
    if args.op:
        ops = filter_ops(ops, include_names=[args.op])

    tester = ShardingRuleTester(
        cache_path=args.cache if not args.no_cache else None,
        xfail_path=args.xfail,
        use_cache=not args.no_cache,
    )

    report = tester.run_tests(ops=ops)

    print(f"\nResults:")
    print(f"  Passed: {len(report.passed)}")
    print(f"  False Positives: {len(report.false_positives)}")
    print(f"  False Negatives: {len(report.false_negatives)}")
    print(f"  Errors: {len(report.errors)}")
    print(f"  XFailed: {len(report.xfailed)}")
    print(f"  Skipped: {len(report.skipped)}")

    if report.false_positives:
        print("\nFalse Positives:")
        for r in report.false_positives[:5]:
            print(f"  {r.op_name}: {r.combination}")
            print(f"    {r.ground_truth.error}")

def cmd_add_xfail(args):
    """Add an xfail entry."""
    from .xfail import XFailRegistry

    registry = XFailRegistry(args.xfail)
    registry.add_xfail(
        op=args.op,
        placements="*",
        reason=args.reason,
        issue=args.issue,
    )
    print(f"Added xfail for {args.op}")

def main():
    parser = argparse.ArgumentParser(description="ShRT Developer Tools")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(__file__).parent / "data" / "shrt_cache.yaml",
    )
    parser.add_argument(
        "--xfail",
        type=Path,
        default=Path(__file__).parent / "data" / "shrt_xfails.yaml",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # status
    status_parser = subparsers.add_parser("status")
    status_parser.set_defaults(func=cmd_status)

    # run
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--op", help="Filter to specific operator")
    run_parser.add_argument("--no-cache", action="store_true")
    run_parser.set_defaults(func=cmd_run)

    # add-xfail
    xfail_parser = subparsers.add_parser("add-xfail")
    xfail_parser.add_argument("op")
    xfail_parser.add_argument("reason")
    xfail_parser.add_argument("--issue")
    xfail_parser.set_defaults(func=cmd_add_xfail)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
```

---

## Implementation Order

### Sprint 1: Foundation
1. `types.py` - Data structures
2. `discovery.py` - Op discovery
3. `enumeration.py` - Placement enumeration
4. Basic tests for discovery and enumeration

### Sprint 2: Core Logic
5. `ground_truth.py` - Execution-based validation
6. `strategy_checker.py` - Strategy querying
7. Integration tests with a few simple ops (add, mul)

### Sprint 3: Infrastructure
8. `database.py` - Caching
9. `xfail.py` - Known failures
10. `runner.py` - Orchestration
11. `report.py` - Reporting

### Sprint 4: Integration
12. `cli.py` - Developer tools
13. `test_shrt.py` - CI integration
14. Initial xfail population for known issues
15. Documentation

---

## Configuration

### Default Test Parameters

```python
# In runner.py or a config module
DEFAULT_CONFIG = {
    "world_size": 4,
    "device": "cpu",
    "strategy_types": {"single_dim"},
    "include_partial": True,
    "max_samples_per_op": 3,
    "max_combinations_per_sample": None,  # Test all
    "rtol": 1e-5,
    "atol": 1e-8,
}
```

### Environment Variables

```bash
SHRT_WORLD_SIZE=4           # Override world size
SHRT_DEVICE=cpu             # Device for testing
SHRT_CACHE_PATH=...         # Override cache location
SHRT_MAX_SAMPLES=3          # Limit samples per op
SHRT_MAX_COMBINATIONS=1000  # Limit combinations
```

---

## CI Integration

### Test Tiers

1. **PR Tests** (fast, <5 min):
   - `max_samples_per_op=1`
   - `max_combinations_per_sample=100`
   - Cache enabled (only test changed strategies)

2. **Nightly Tests** (comprehensive):
   - `max_samples_per_op=5`
   - `max_combinations_per_sample=None` (all)
   - Cache disabled (full re-validation)

3. **Weekly Full Scan**:
   - All ops, all samples, all combinations
   - Update cache file for PR tests

### GitHub Actions Integration

```yaml
# .github/workflows/shrt.yml
name: ShRT Tests

on:
  pull_request:
    paths:
      - 'torch/distributed/tensor/_ops/**'
      - 'torch/distributed/tensor/_testing/shrt/**'

jobs:
  shrt:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run ShRT
        run: python -m pytest test/distributed/tensor/test_shrt.py -v
```

---

## Success Criteria

1. **Discovery**: Correctly identify all ops with both OpInfo and sharding strategies
2. **Enumeration**: Generate all valid placement combinations including Partial
3. **Ground Truth**: Accurately determine if a combination produces correct results
4. **Strategy Checking**: Correctly query strategy functions
5. **Comparison**: Identify false positives and false negatives
6. **Caching**: Speed up repeated runs by caching validated combinations
7. **CI Integration**: Block PRs that introduce incorrect sharding rules

---

## Issues, Risks, and Mitigations

### Critical Issues

#### 1. Ground Truth Definition for Partial Inputs

**Issue**: The definition of "correct" for Partial inputs is subtle. For `Partial(SUM)`, each rank holds a partial contribution that must sum to the full value. But how do we simulate this for ground truth computation?

**Current approach**: Divide input by world_size so sum = original. But this changes the mathematical operation:
- Original: `output = f(input)`
- Simulated: `output = f(input / world_size)` then verify sum across ranks

**Risk**: This only works for linear operations. For nonlinear ops (e.g., `relu`, `exp`), `f(a/n) + f(b/n) ≠ f(a+b)`.

**Mitigation options**:
1. **Restrict Partial testing to linear ops**: Only test Partial placements for ops known to be linear
2. **Define Partial semantics carefully**: Partial inputs mean "this is a partial sum waiting to be reduced" - most ops should reject Partial inputs unless they're reductions
3. **Use allreduce simulation**: Actually simulate what happens - each rank computes on its partial, then we allreduce and compare

**Recommendation**: Option 2. Most ops should reject Partial inputs as invalid. Only reduction ops (sum, mean) and specific ops that can handle partials should accept them. The ground truth test should verify that:
- For most ops with Partial inputs → execution should fail or produce wrong results → strategy should reject
- For reduction ops with Partial inputs → allreduce result should match reference

#### 2. OpInfo Sample Inputs May Not Cover All Edge Cases

**Issue**: OpInfo's `sample_inputs_func` generates representative inputs, but may miss edge cases that exercise sharding logic:
- Tensors with size-1 dimensions (sharding on dim with size 1)
- Tensors with sizes not divisible by world_size
- Broadcasting scenarios
- Empty tensors

**Risk**: Strategy might be wrong for edge cases not covered by samples.

**Mitigation**:
1. **Augment samples**: Add ShRT-specific sample generators that create edge-case tensors
2. **Shape mutation**: For each OpInfo sample, also test with modified shapes (size-1 dims, odd sizes)
3. **Accept limitation**: Document that ShRT tests representative cases, not all possible shapes

**Recommendation**: Start with OpInfo samples, add shape mutation in Phase 2. Create a `mutate_sample_for_sharding_tests()` function.

#### 3. Strategy Function Interface Varies

**Issue**: The three strategy types (`single_dim`, `full`, `rule`) have different interfaces:
- `single_dim`: Returns `list[list[Placement | _ShardingPlaceholder]]`
- `full`: Returns `OpStrategy` with `OpSpec` objects
- `rule`: Returns `OutputSharding`

Querying them uniformly is complex.

**Risk**: Strategy checker might misinterpret strategy output, leading to wrong test results.

**Mitigation**:
1. **Separate checkers per type**: Implement `SingleDimStrategyChecker`, `FullStrategyChecker`, `RuleChecker`
2. **Focus on single_dim first**: Per SHRT.md, single-dim is the focus. Get it working first.
3. **Thorough unit tests**: Test each checker independently before integration

**Recommendation**: Option 2 + 3. Implement single-dim checker first with good tests. Add full/rule later if needed.

### Medium-Priority Issues

#### 4. Combinatorial Explosion

**Issue**: For an op with N tensor inputs, D dimensions each, the number of combinations is:
`(1 + D + P)^N * (1 + D)^M` where P = partial variants (2), M = output tensors

For a simple matmul (2 2D inputs, 1 2D output):
- Input options: 1 (Replicate) + 2 (Shard(0), Shard(1)) + 2 (Partial) = 5
- Output options: 1 + 2 = 3
- Total: 5 * 5 * 3 = 75 combinations

For complex ops with many inputs, this explodes.

**Risk**: Tests take too long, CI times out, developers skip running ShRT.

**Mitigation**:
1. **Tiered testing**: Fast subset for PRs, full test nightly
2. **Caching**: Only re-test changed strategies
3. **Smart sampling**: For ops with >1000 combinations, sample representative subset
4. **Parallelization**: Run combinations in parallel

**Recommendation**: Implement caching first (biggest win), add tiered testing in CI config.

#### 5. aten Overload Matching

**Issue**: OpInfo's `aten_name` is a base name (e.g., "add"), but sharding strategies register on specific overloads (e.g., `aten.add.Tensor` vs `aten.add.Scalar`). Need to match correctly.

**Risk**: Might test with wrong sample inputs for an overload, or miss some overloads entirely.

**Mitigation**:
1. **Match by operation signature**: Check if sample input types match overload signature
2. **Filter samples**: Only use samples that can dispatch to the specific overload
3. **Log mismatches**: Warn when OpInfo exists but no samples match overload

**Recommendation**: Start with base name matching, filter out samples that raise TypeErrors. Add signature matching if needed.

#### 6. Numerical Precision in Comparison

**Issue**: When comparing local outputs to reference shards, floating-point precision matters. Some ops have inherent numerical variance (reductions on different orders, etc.).

**Risk**: False test failures due to precision differences, or missed bugs due to too-loose tolerances.

**Mitigation**:
1. **Per-op tolerances**: Allow ops to specify custom rtol/atol
2. **Use OpInfo ref**: Some OpInfo entries specify reference implementations with expected tolerance
3. **Relative comparison**: Compare relative error, not absolute

**Recommendation**: Start with default tolerances (rtol=1e-5, atol=1e-8), add per-op overrides as needed.

### Low-Priority Issues

#### 7. Non-Tensor Outputs

**Issue**: Some ops return non-tensor outputs (e.g., `max` returns (values, indices), `topk` similar). Need to handle tuple outputs correctly.

**Risk**: Incorrect handling of multi-output ops.

**Mitigation**: Already handled in plan - detect tuple outputs and validate each tensor output separately. Non-tensor outputs (like shapes) are ignored.

#### 8. In-Place Operations

**Issue**: In-place ops (`add_`, etc.) modify inputs. Ground truth computation needs to be careful not to corrupt test state.

**Risk**: Test state pollution causing incorrect results.

**Mitigation**: Clone all tensors before ground truth computation. Already planned in `shard_tensor` which returns `.clone()`.

#### 9. View Operations

**Issue**: View ops (reshape, transpose, etc.) don't copy data. Sharding semantics for views are complex - does sharding the view mean sharding the underlying storage?

**Risk**: Ground truth computation for views might not match DTensor's actual behavior.

**Mitigation**:
1. **Skip views initially**: Focus on compute ops first
2. **Add view-specific tests**: Separate test suite for view operations
3. **Consult DTensor team**: Clarify expected semantics

**Recommendation**: Exclude view ops in v1, add them as a separate focused effort.

#### 10. LocalTensor Fidelity

**Issue**: LocalTensor simulates distributed execution on a single machine. It might not perfectly match real distributed behavior (e.g., timing-dependent issues, collective semantics).

**Risk**: Tests pass with LocalTensor but fail in real distributed setting.

**Mitigation**:
1. **Nightly real distributed tests**: Run subset of tests with real NCCL
2. **Trust LocalTensor for sharding logic**: The sharding math should be identical; LocalTensor just avoids communication
3. **Add integration test**: Single test that runs with real distributed for sanity check

**Recommendation**: Trust LocalTensor for sharding correctness. Add optional real distributed mode for nightly.

### Architectural Risks

#### 11. Cache Invalidation Correctness

**Issue**: Cache is invalidated by strategy source hash. But strategy correctness might depend on:
- Helper functions called by the strategy
- Global configuration
- PyTorch version

**Risk**: Cache returns stale results after dependency changes.

**Mitigation**:
1. **Conservative invalidation**: Clear cache on PyTorch version changes
2. **Include dependencies in hash**: Hash strategy + called functions
3. **Periodic full re-validation**: Weekly nightly that ignores cache

**Recommendation**: Use simple source hash initially. Add dependency tracking if cache staleness becomes an issue.

#### 12. XFail Maintenance Burden

**Issue**: XFails can accumulate if not actively maintained. Old xfails might mask new bugs.

**Risk**: XFail list grows indefinitely, masking regressions.

**Mitigation**:
1. **Expiration dates**: XFails have `expected_fix` date, warn if overdue
2. **Periodic review**: Quarterly audit of xfails
3. **Require issue links**: All xfails must reference a tracking issue

**Recommendation**: Require issue links, add warning for >90 day old xfails.

### Summary of Key Recommendations

| Issue | Recommendation | Priority |
|-------|----------------|----------|
| Partial semantics | Most ops reject Partial; test reduction ops separately | High |
| Sample coverage | Start with OpInfo, add shape mutation later | Medium |
| Strategy interface variance | Focus on single-dim first | High |
| Combinatorial explosion | Implement caching first | High |
| Overload matching | Base name + error filtering | Medium |
| Precision | Default tolerances, per-op overrides | Low |
| View ops | Exclude in v1 | Low |
| Cache invalidation | Simple hash, periodic full re-validation | Medium |
| XFail maintenance | Require issue links, add expiration warnings | Low |

### Proposed v1 Scope

Given the risks, recommend limiting v1 to:

1. **Single-dim strategies only** (skip full strategies and rules)
2. **OpInfo samples only** (no augmentation initially)
3. **Exclude view ops** (reshape, transpose, etc.)
4. **Partial inputs for reduction ops only**
5. **Caching enabled** (critical for performance)
6. **XFails with required issue links**

This keeps v1 focused and achievable while laying groundwork for expansion.
