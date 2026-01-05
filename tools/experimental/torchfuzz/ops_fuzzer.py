# mypy: ignore-errors

import random
from dataclasses import dataclass

import torch

from torchfuzz.operators import get_operator, list_operators
from torchfuzz.tensor_fuzzer import (
    fuzz_tensor_size,
    fuzz_torch_tensor_type,
    fuzz_valid_stride,
    ScalarSpec,
    Spec,
    specs_compatible,
    TensorSpec,
)


# Cache operators at module level to avoid repeated calls to list_operators()
_CACHED_OPERATORS = None


def _get_cached_operators():
    """Get cached operators, initializing if necessary."""
    global _CACHED_OPERATORS
    if _CACHED_OPERATORS is None:
        _CACHED_OPERATORS = list_operators()
    return _CACHED_OPERATORS


def _get_template_filtered_operators(
    template: str = "default", supported_ops: list[str] | None = None
):
    """Get operators filtered by template's supported_ops, with user override.

    If supported_ops is provided, it takes precedence and is used to filter the
    registry. Otherwise, the template's supported_ops are used. If neither are
    specified, all operators are returned.
    """
    # Instantiate template
    if template == "dtensor":
        from torchfuzz.codegen import DTensorFuzzTemplate

        fuzz_template = DTensorFuzzTemplate()
    elif template == "dtensor_placements":
        from torchfuzz.codegen import DTensorFuzzPlacementsTemplate

        fuzz_template = DTensorFuzzPlacementsTemplate()
    elif template == "unbacked":
        from torchfuzz.codegen import UnbackedFuzzTemplate

        fuzz_template = UnbackedFuzzTemplate()
    else:
        from torchfuzz.codegen import DefaultFuzzTemplate

        fuzz_template = DefaultFuzzTemplate()

    all_operators = _get_cached_operators()

    # Determine allowed ops list
    allowed_ops = supported_ops if supported_ops else fuzz_template.supported_ops

    # If no supported_ops specified, return all operators
    if not allowed_ops:
        return all_operators

    # Filter operators based on allowed_ops
    filtered_ops = {}

    for op_name, operator in all_operators.items():
        # Always include operations that don't have a specific torch operation
        # (utility operations like arg, constant, item, scalar ops)
        torch_op = operator.torch_op_name
        if torch_op is None:
            # Set template on operators that support it
            if hasattr(operator, "set_template"):
                operator.set_template(template)  # type: ignore[attr-defined]
            filtered_ops[op_name] = operator
            continue

        # Check if the operator supports any of the allowed operations
        should_include = False
        for supported_op in allowed_ops:
            # Direct torch operation matching
            if torch_op == supported_op:
                should_include = True
                break

            # Direct name matching as fallback
            if supported_op in op_name or op_name in supported_op:
                should_include = True
                break

        if should_include:
            # Set template on operators that support it
            if hasattr(operator, "set_template"):
                operator.set_template(template)  # type: ignore[attr-defined]
            filtered_ops[op_name] = operator

    return filtered_ops


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


def fuzz_spec(template: str = "default") -> Spec:
    """
    Generate a random Spec (either TensorSpec or ScalarSpec) using template's distribution preferences.

    Args:
        template: Template name to determine configuration and distribution

    Returns:
        Spec: Either a TensorSpec or ScalarSpec according to template's distribution
    """
    # Try to use template's custom distribution if available
    try:
        # Instantiate template
        if template == "dtensor":
            from torchfuzz.codegen import DTensorFuzzTemplate

            fuzz_template = DTensorFuzzTemplate()
        elif template == "dtensor_placements":
            from torchfuzz.codegen import DTensorFuzzPlacementsTemplate

            fuzz_template = DTensorFuzzPlacementsTemplate()
        elif template == "unbacked":
            from torchfuzz.codegen import UnbackedFuzzTemplate

            fuzz_template = UnbackedFuzzTemplate()
        else:
            from torchfuzz.codegen import DefaultFuzzTemplate

            fuzz_template = DefaultFuzzTemplate()

        # Use template's custom spec generation
        return fuzz_template.fuzz_spec_custom()

    except Exception:
        # Fallback to original hardcoded behavior if template fails
        # Get random dtype based on template
        dtype = fuzz_torch_tensor_type(template)

        # 20% probability of returning ScalarSpec
        if random.random() < 0.2:
            return ScalarSpec(dtype=dtype)

        # 80% probability of returning TensorSpec
        # Get random size and corresponding stride
        size = fuzz_tensor_size()
        stride = fuzz_valid_stride(size)
        return TensorSpec(size=size, stride=stride, dtype=dtype)


def fuzz_op(
    target_spec: Spec,
    depth,
    stack_size,
    template: str = "default",
    supported_ops: list[str] | None = None,
) -> tuple[str, list[Spec]]:
    """
    Given an output specification, returns an operation that can
    produce a tensor with that layout using the operator class system.

    Args:
        target_spec: Desired output specification (TensorSpec or ScalarSpec)
        depth: Maximum depth for operation generation. At depth 0, only leaf operations
               (constant, arg) are allowed. Higher depths allow more complex operations.
        stack_size: Current stack size. When < 10, reduces probability of leaf operations.

    Returns:
        Tuple of (operation_name, list_of_argument_specs) where each argument spec
        describes the layout requirements for the operation's inputs
    """
    # Get template-filtered operators
    available_operators = _get_template_filtered_operators(template, supported_ops)

    # Filter operators that can produce the target spec
    # IMPORTANT: iterate in a deterministic order to avoid dict-order nondeterminism
    compatible_ops = []
    for op_name in sorted(available_operators.keys()):
        operator = available_operators[op_name]
        if operator.can_produce(target_spec):
            compatible_ops.append((op_name, operator))

    # Shuffle with seeded RNG (caller seeds random), but from a deterministic base order
    random.shuffle(compatible_ops)

    if not compatible_ops:
        raise ValueError(f"No operators available that can produce {target_spec}")

    # Categorize operators into leaf and non-leaf
    leaf_ops = []
    non_leaf_ops = []

    for op_name, operator in compatible_ops:
        if op_name in ["constant", "arg"] or op_name.startswith("arg_"):
            leaf_ops.append((op_name, operator))
        else:
            non_leaf_ops.append((op_name, operator))

    # Choose operation based on depth and stack size constraints
    if depth == 0:
        # At depth 0, only allow leaf operations
        if not leaf_ops:
            # If no leaf ops can produce this spec, fallback to arg
            return _get_arg_args_specs(target_spec)
        # Weighted choice among leaf ops
        leaf_weights = [
            op.get_weight(
                target_spec=target_spec,
                depth=depth,
                stack_size=stack_size,
                template=template,
            )
            for _, op in leaf_ops
        ]
        idx = random.choices(range(len(leaf_ops)), weights=leaf_weights, k=1)[0]
        chosen_op_name, chosen_operator = leaf_ops[idx]
    else:
        # At higher depths, choose between leaf and non-leaf operations
        # Reduce probability of leaf operations when stack_size < 10
        if (stack_size < 10 or depth > 7) and non_leaf_ops:
            # 80% chance of non-leaf, 20% chance of leaf
            if random.random() < 0.8:
                # Weighted choice among non-leaf ops
                nonleaf_weights = [
                    op.get_weight(
                        target_spec=target_spec,
                        depth=depth,
                        stack_size=stack_size,
                        template=template,
                    )
                    for _, op in non_leaf_ops
                ]
                idx = random.choices(
                    range(len(non_leaf_ops)), weights=nonleaf_weights, k=1
                )[0]
                chosen_op_name, chosen_operator = non_leaf_ops[idx]
            else:
                if leaf_ops:
                    leaf_weights = [
                        op.get_weight(
                            target_spec=target_spec,
                            depth=depth,
                            stack_size=stack_size,
                            template=template,
                        )
                        for _, op in leaf_ops
                    ]
                    idx = random.choices(
                        range(len(leaf_ops)), weights=leaf_weights, k=1
                    )[0]
                    chosen_op_name, chosen_operator = leaf_ops[idx]
                else:
                    nonleaf_weights = [
                        op.get_weight(
                            target_spec=target_spec,
                            depth=depth,
                            stack_size=stack_size,
                            template=template,
                        )
                        for _, op in non_leaf_ops
                    ]
                    idx = random.choices(
                        range(len(non_leaf_ops)), weights=nonleaf_weights, k=1
                    )[0]
                    chosen_op_name, chosen_operator = non_leaf_ops[idx]
        else:
            # Normal probability distribution over all ops
            all_ops = non_leaf_ops + leaf_ops
            if all_ops:
                all_weights = [
                    op.get_weight(
                        target_spec=target_spec,
                        depth=depth,
                        stack_size=stack_size,
                        template=template,
                    )
                    for _, op in all_ops
                ]
                idx = random.choices(range(len(all_ops)), weights=all_weights, k=1)[0]
                chosen_op_name, chosen_operator = all_ops[idx]
            else:
                chosen_op_name, chosen_operator = ("arg", get_operator("arg"))

    if chosen_operator is None:
        # If no operator found, fallback to arg
        return _get_arg_args_specs(target_spec)

    input_specs = chosen_operator.fuzz_inputs_specs(target_spec)
    return chosen_op_name, input_specs


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
    seed: int | None = None,
    template: str = "default",
    supported_ops: list[str] | None = None,
) -> OperationGraph:
    """
    Generate a graph of operations that produces the target specification.

    The graph-based approach allows for better visualization, debugging, and
    potential optimizations like common subexpression elimination.

    Args:
        target_spec: The desired output specification (TensorSpec or ScalarSpec)
        max_depth: Maximum depth of operations. At depth 0, only leaf operations (constant, arg) are used.
        seed: Random seed for reproducible generation. If None, uses current random state.
        template: Template name to determine configuration

    Returns:
        OperationGraph with nodes organized in a DAG structure
    """

    # Set seed for reproducible generation
    if seed is not None:
        import random

        random.seed(seed)
        torch.manual_seed(seed)
        # Reset global arg counter for deterministic behavior
        global _next_arg_id
        _next_arg_id = 0

    # Global counter for unique node IDs - start from 0 for deterministic behavior
    node_counter = 0

    # Dictionary to store all nodes: node_id -> OperationNode
    nodes: dict[str, OperationNode] = {}

    def _generate_node(spec: Spec, depth: int, stack_size: int = 0) -> str:
        """
        Generate a node for the given spec and return its node_id.
        """
        nonlocal node_counter

        # Generate new operation
        op_name, input_specs = fuzz_op(spec, depth, stack_size, template, supported_ops)

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
