# mypy: ignore-errors

import random
from dataclasses import dataclass
from typing import Optional

from tensor_fuzzer import (
    fuzz_tensor_size,
    fuzz_torch_tensor_type,
    fuzz_valid_stride,
    ScalarSpec,
    Spec,
    specs_compatible,
    TensorSpec,
)

import torch


@dataclass
class OperationNode:
    """
    Represents a node in the operation graph.

    Attributes:
        node_id: Unique identifier for this node
        op_name: Name of the operation (e.g., 'torch.ops.aten.add', 'scalar_add', 'arg')
        input_specs: List of input specifications required by this operation
        output_spec: Output specification produced by this operation
        input_nodes: List of node IDs that provide inputs to this operation
        depth: Depth level of this node in the generation tree
    """

    node_id: str
    op_name: str
    input_specs: list[Spec]
    output_spec: Spec
    input_nodes: list[str]
    depth: int

    def __str__(self) -> str:
        """String representation for debugging."""
        return (
            f"{self.node_id}: {self.op_name} -> {self.output_spec} (depth {self.depth})"
        )

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"OperationNode(node_id='{self.node_id}', op_name='{self.op_name}', "
            f"input_specs={self.input_specs}, output_spec={self.output_spec}, "
            f"input_nodes={self.input_nodes}, depth={self.depth})"
        )


@dataclass
class OperationGraph:
    """
    Represents a graph of operations.

    Attributes:
        nodes: Dictionary mapping node_id to OperationNode
        root_node_id: ID of the root node that produces the final output (the output node)
        target_spec: The specification that the root node should produce
    """

    nodes: dict[str, OperationNode]
    root_node_id: str  # The output node - produces the final result of the graph
    target_spec: Spec

    def __post_init__(self):
        """Validate the graph structure after initialization."""
        if self.root_node_id not in self.nodes:
            raise ValueError(f"Root node {self.root_node_id} not found in nodes")

    def get_topological_order(self) -> list[str]:
        """
        Get nodes in topological order (dependencies before dependents).

        Returns:
            List of node IDs in topological order
        """
        visited = set()
        temp_visited = set()
        result = []

        def visit(node_id: str):
            if node_id in temp_visited:
                raise ValueError(f"Cycle detected involving node {node_id}")
            if node_id in visited:
                return

            temp_visited.add(node_id)
            node = self.nodes[node_id]

            # Visit all input nodes first
            for input_node_id in node.input_nodes:
                if input_node_id in self.nodes:  # Skip external inputs
                    visit(input_node_id)

            temp_visited.remove(node_id)
            visited.add(node_id)
            result.append(node_id)

        # Start from all nodes to handle disconnected components
        for node_id in self.nodes:
            if node_id not in visited:
                visit(node_id)

        return result

    def get_leaf_nodes(self) -> list[str]:
        """Get all leaf nodes (nodes with no inputs)."""
        return [node_id for node_id, node in self.nodes.items() if not node.input_nodes]

    def get_node_dependencies(self, node_id: str) -> list[str]:
        """Get all nodes that this node depends on (transitive closure)."""
        visited = set()
        dependencies = []

        def collect_deps(current_id: str):
            if current_id in visited or current_id not in self.nodes:
                return
            visited.add(current_id)

            node = self.nodes[current_id]
            for input_node_id in node.input_nodes:
                dependencies.append(input_node_id)
                collect_deps(input_node_id)

        collect_deps(node_id)
        return dependencies

    def __str__(self) -> str:
        """String representation for debugging."""
        lines = [
            f"OperationGraph (root: {self.root_node_id}, target: {self.target_spec})"
        ]
        for node_id in self.get_topological_order():
            node = self.nodes[node_id]
            inputs_str = f" <- {node.input_nodes}" if node.input_nodes else ""
            lines.append(f"  {node}{inputs_str}")
        return "\n".join(lines)


def fuzz_spec() -> Spec:
    """
    Generate a random Spec (either TensorSpec or ScalarSpec) using tensor fuzzing functions.

    Utilizes:
    - fuzz_torch_tensor_type() for random dtype
    - fuzz_tensor_size() for random tensor size
    - fuzz_valid_stride() for random valid strides

    Returns:
        Spec: Either a TensorSpec (80% probability) or ScalarSpec (20% probability) with random properties
    """
    # Get random dtype
    dtype = fuzz_torch_tensor_type()

    # 20% probability of returning ScalarSpec
    if random.random() < 0.2:
        return ScalarSpec(dtype=dtype)

    # 80% probability of returning TensorSpec
    # Get random size and corresponding stride
    size = fuzz_tensor_size()
    stride = fuzz_valid_stride(size)
    return TensorSpec(size=size, stride=stride, dtype=dtype)


def fuzz_op(target_spec: Spec, depth, stack_size) -> tuple[str, list[Spec]]:
    """
    Given an output specification, returns an operation that can
    produce a tensor with that layout.

    Supports:
    - For scalars: scalar_add, scalar_multiply, item, constant, arg
    - For tensors: aten.add, aten.mul, constant, arg

    Args:
        target_spec: Desired output specification (TensorSpec or ScalarSpec)
        depth: Maximum depth for operation generation. At depth 0, only leaf operations
               (constant, arg) are allowed. Higher depths allow more complex operations.
        stack_size: Current stack size. When < 10, reduces probability of leaf operations.

    Returns:
        Tuple of (operation_name, list_of_argument_specs) where each argument spec
        describes the layout requirements for the operation's inputs
    """
    if isinstance(target_spec, ScalarSpec):
        if target_spec.constant is not None:
            # At depth 0, only allow constant operation
            return _get_constant_args_specs(target_spec)
        if depth == 0:
            # At depth 0, only allow leaf operations
            ops = ["constant", "arg"]
            chosen_op = random.choice(ops)
        else:
            # At higher depths, allow all scalar operations
            non_leaf_ops = ["scalar_add", "scalar_multiply", "torch.ops.aten.item"]
            leaf_ops = ["constant", "arg"]

            # Reduce probability of leaf operations when stack_size < 10
            if stack_size < 10 or depth > 7:
                # 80% chance of non-leaf, 20% chance of leaf
                if random.random() < 0.8:
                    chosen_op = random.choice(non_leaf_ops)
                else:
                    chosen_op = random.choice(leaf_ops)
            else:
                # Normal probability distribution
                all_ops = non_leaf_ops + leaf_ops
                chosen_op = random.choice(all_ops)

        if chosen_op == "scalar_add":
            return _get_scalar_add_args_specs(target_spec)
        elif chosen_op == "scalar_multiply":
            return _get_scalar_multiply_args_specs(target_spec)
        elif chosen_op == "torch.ops.aten.item":
            return _get_item_args_specs(target_spec)
        elif chosen_op == "constant":
            return _get_constant_args_specs(target_spec)
        else:  # arg
            return _get_arg_args_specs(target_spec)

    elif isinstance(target_spec, TensorSpec):
        if depth == 0:
            # At depth 0, only allow leaf operations
            ops = ["arg"]
            chosen_op = random.choice(ops)
        else:
            # At higher depths, allow all tensor operations
            non_leaf_ops = [
                "torch.ops.aten.add",
                "torch.ops.aten.mul",
            ]

            leaf_ops = ["arg"]

            # Reduce probability of leaf operations when stack_size < 10
            if stack_size < 10:
                # 80% chance of non-leaf, 20% chance of leaf
                if random.random() < 0.8:
                    chosen_op = random.choice(non_leaf_ops)
                else:
                    chosen_op = random.choice(leaf_ops)
            else:
                # Normal probability distribution
                all_ops = non_leaf_ops + leaf_ops
                chosen_op = random.choice(all_ops)

        if chosen_op == "torch.ops.aten.add":
            return _get_aten_add_args_specs(target_spec)
        elif chosen_op == "torch.ops.aten.mul":
            return _get_aten_mul_args_specs(target_spec)
        elif chosen_op == "constant":
            return _get_constant_args_specs(target_spec)
        else:  # arg
            return _get_arg_args_specs(target_spec)

    else:
        raise ValueError(f"Unknown target spec type: {type(target_spec)}")


def _get_scalar_add_args_specs(target_spec: ScalarSpec) -> tuple[str, list[Spec]]:
    """Get argument specifications for scalar_add operation using type promotion rules."""
    # Use PyTorch's implicit type promotion rules to generate diverse input types
    arg_specs = _get_promoted_scalar_args(target_spec.dtype)
    return "scalar_add", arg_specs


def _get_scalar_multiply_args_specs(target_spec: ScalarSpec) -> tuple[str, list[Spec]]:
    """Get argument specifications for scalar_multiply operation using type promotion rules."""
    # Use PyTorch's implicit type promotion rules to generate diverse input types
    arg_specs = _get_promoted_scalar_args(target_spec.dtype)
    return "scalar_multiply", arg_specs


# Define promotion chains - types that can promote to the target
# PyTorch promotion hierarchy (simplified):
# - bool < int8 < int16 < int32 < int64 < float16 < float32 < float64 < complex64 < complex128
# - uint types have limited promotion support
_PROMOTION_CHAINS = {
    torch.bool: [torch.bool],
    torch.int8: [torch.bool, torch.int8],
    torch.int16: [torch.bool, torch.int8, torch.int16],
    torch.int32: [torch.bool, torch.int8, torch.int16, torch.int32],
    torch.int64: [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64],
    torch.float16: [
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
    ],
    torch.float32: [
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
    ],
    torch.float64: [
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
    ],
    torch.complex64: [
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.complex64,
    ],
    torch.complex128: [
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ],
}


def _get_promoted_dtypes(target_dtype: torch.dtype) -> list[torch.dtype]:
    """
    Generate two dtypes that will promote to target_dtype via PyTorch's type promotion rules.
    """

    # Get compatible input types for the target dtype
    compatible_types = _PROMOTION_CHAINS.get(target_dtype, [target_dtype])

    # Strategy: Choose between same type or mixed promotion
    strategies = ["same_type", "mixed_promotion"]
    strategy = random.choice(strategies)

    if strategy == "same_type":
        # Both args same type as target
        return [target_dtype, target_dtype]

    else:  # mixed_promotion
        # Mixed types where the result will promote to target_dtype
        lower_types = compatible_types[:-1]  # All except the last (target_dtype)

        if lower_types:
            # One arg is target_dtype, one is lower (will promote to target)
            lower_dtype = random.choice(lower_types)
            if random.random() < 0.5:
                return [target_dtype, lower_dtype]
            else:
                return [lower_dtype, target_dtype]
        else:
            # Fallback to same type if no lower types available
            return [target_dtype, target_dtype]


def _get_promoted_scalar_args(target_dtype: torch.dtype) -> list[Spec]:
    """
    Generate two argument specs that will promote to target_dtype via PyTorch's type promotion rules.
    """
    arg_dtypes = _get_promoted_dtypes(target_dtype)

    # For ScalarSpec output, both inputs must be ScalarSpec
    # (mixing with 0-D TensorSpec would produce 0-D TensorSpec output)
    return [ScalarSpec(arg_dtypes[0]), ScalarSpec(arg_dtypes[1])]


def _get_item_args_specs(target_spec: ScalarSpec) -> tuple[str, list[Spec]]:
    """Get argument specifications for torch.ops.aten.item operation."""
    # torch.ops.aten.item: tensor -> scalar (extract single element)
    # Create a tensor spec that can produce a scalar via .item()
    tensor_spec = TensorSpec(
        size=(1,), stride=(1,), dtype=target_spec.dtype
    )  # 1-D tensor with 1 element
    arg_specs: list[Spec] = [tensor_spec]
    return "torch.ops.aten.item", arg_specs


def _get_aten_add_args_specs(target_spec: TensorSpec) -> tuple[str, list[Spec]]:
    """Get argument specifications for torch.ops.aten.add operation using type promotion rules."""
    # Use promotion rules to generate diverse tensor input types
    arg_dtypes = _get_promoted_dtypes(target_spec.dtype)

    arg_specs: list[Spec] = [
        TensorSpec(target_spec.size, target_spec.stride, arg_dtypes[0]),
        TensorSpec(target_spec.size, target_spec.stride, arg_dtypes[1]),
    ]
    return "torch.ops.aten.add", arg_specs


def _get_aten_mul_args_specs(target_spec: TensorSpec) -> tuple[str, list[Spec]]:
    """Get argument specifications for torch.ops.aten.mul operation using type promotion rules."""
    # Use promotion rules to generate diverse tensor input types
    arg_dtypes = _get_promoted_dtypes(target_spec.dtype)

    arg_specs: list[Spec] = [
        TensorSpec(target_spec.size, target_spec.stride, arg_dtypes[0]),
        TensorSpec(target_spec.size, target_spec.stride, arg_dtypes[1]),
    ]
    return "torch.ops.aten.mul", arg_specs


def _get_constant_args_specs(target_spec: Spec) -> tuple[str, list[Spec]]:
    """Get argument specifications for constant operation."""
    # Constant operation takes no arguments - generates a fixed constant value/tensor
    return "constant", []


# Global counter for generating unique argument IDs
_next_arg_id = 0


def _get_arg_args_specs(target_spec: Spec) -> tuple[str, list[Spec]]:
    """Get argument specifications for arg operation."""
    global _next_arg_id

    # Generate a unique argument ID
    arg_id = _next_arg_id
    _next_arg_id += 1

    # Return the operation name with the arg_id embedded and no input specs
    return f"arg_{arg_id}", []


def fuzz_operation_graph(
    target_spec: Spec,
    max_depth: int = 7,
    seed: Optional[int] = None,
) -> OperationGraph:
    """
    Generate a graph of operations that produces the target specification.

    The graph-based approach allows for better visualization, debugging, and
    potential optimizations like common subexpression elimination.

    Args:
        target_spec: The desired output specification (TensorSpec or ScalarSpec)
        max_depth: Maximum depth of operations. At depth 0, only leaf operations (constant, arg) are used.
        seed: Random seed for reproducible generation. If None, uses current random state.

    Returns:
        OperationGraph with nodes organized in a DAG structure
    """

    # Set seed for reproducible generation
    if seed is not None:
        import random

        random.seed(seed)
        torch.manual_seed(seed)

    # Global counter for unique node IDs
    node_counter = 0

    # Dictionary to store all nodes: node_id -> OperationNode
    nodes: dict[str, OperationNode] = {}

    def _generate_node(spec: Spec, depth: int, stack_size: int = 0) -> str:
        """
        Generate a node for the given spec and return its node_id.
        """
        nonlocal node_counter

        # Generate new operation
        op_name, input_specs = fuzz_op(spec, depth, stack_size)

        # Create unique node ID
        node_id = f"node_{node_counter}"
        node_counter += 1

        # Generate input nodes
        input_node_ids = []
        if input_specs:  # Non-leaf operations
            for input_spec in input_specs:
                input_node_id = _generate_node(
                    input_spec, max(0, depth - 1), stack_size + len(input_node_ids) + 1
                )
                input_node_ids.append(input_node_id)

        # Create the operation node
        node = OperationNode(
            node_id=node_id,
            op_name=op_name,
            input_specs=input_specs,
            output_spec=spec,
            input_nodes=input_node_ids,
            depth=depth,
        )

        # Store the node
        nodes[node_id] = node

        return node_id

    # Generate the root node
    root_node_id = _generate_node(target_spec, max_depth, 0)

    # Create and return the operation graph
    graph = OperationGraph(
        nodes=nodes, root_node_id=root_node_id, target_spec=target_spec
    )

    # Verify that the root node produces the target spec
    root_node = nodes[root_node_id]
    if not specs_compatible(root_node.output_spec, target_spec):
        raise ValueError(
            f"Generated graph root node produces {root_node.output_spec}, "
            f"but target spec is {target_spec}"
        )

    return graph
