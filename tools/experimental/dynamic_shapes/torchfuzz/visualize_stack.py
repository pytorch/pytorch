# mypy: ignore-errors

"""
Visualization tools for operation stacks as DAGs.
"""

import subprocess

from ops_fuzzer import Operation
from tensor_fuzzer import specs_compatible


def operation_stack_to_dot(
    operation_stack: list[Operation], title: str = "Operation Stack"
) -> str:
    """
    Convert an operation stack to Graphviz DOT format for visualization.

    Args:
        operation_stack: List of Operation instances
        title: Title for the graph

    Returns:
        DOT format string
    """
    dot_lines = [
        "digraph OperationStack {",
        f'    label="{title}";',
        "    rankdir=TB;",  # Top to bottom layout
        "    node [shape=box, style=filled, fontsize=10];",
        "    edge [fontsize=8];",
        "",
    ]

    # Track which operations we've already added as nodes (for argument deduplication)
    added_nodes = set()

    # Add nodes with styling based on operation type
    for i, op in enumerate(operation_stack):
        # For arguments, use the arg name as the node ID to deduplicate
        if op.op_name.startswith("arg_"):
            node_id = op.op_name  # Use arg_0, arg_1, etc. as node ID
            if node_id in added_nodes:
                continue  # Skip if we already added this argument node
            added_nodes.add(node_id)
        else:
            node_id = f"op_{i}"  # Use original op_i format for non-arguments

        # Choose color and shape based on operation type
        if op.op_name.startswith("arg_"):
            color = "lightblue"
            shape = "ellipse"
        elif op.op_name == "constant":
            color = "lightgreen"
            shape = "ellipse"
        elif "aten" in op.op_name:
            color = "lightyellow"
            shape = "box"
        else:
            color = "lightgray"
            shape = "box"

        # Create comprehensive label
        if op.op_name.startswith("arg_"):
            label_parts = [op.op_name]
        else:
            label_parts = [f"Op {i}", op.op_name, f"depth {op.depth}"]

        if hasattr(op.output_spec, "dtype"):
            dtype_str = str(op.output_spec.dtype).replace("torch.", "")
            label_parts.append(dtype_str)

        # Only add size for TensorSpec, not ScalarSpec
        try:
            from tensor_fuzzer import TensorSpec

            if isinstance(op.output_spec, TensorSpec) and op.output_spec.size:
                size_str = "x".join(map(str, op.output_spec.size))
                label_parts.append(f"size {size_str}")
        except (ImportError, AttributeError):
            if (
                hasattr(op.output_spec, "size")
                and hasattr(op.output_spec, "__class__")
                and "TensorSpec" in op.output_spec.__class__.__name__
            ):
                try:
                    size_attr = getattr(op.output_spec, "size", None)
                    if size_attr:
                        size_str = "x".join(map(str, size_attr))
                        label_parts.append(f"size {size_str}")
                except (AttributeError, TypeError):
                    pass

        label = "\\n".join(label_parts)

        # Special highlighting for target operation (index 0)
        extra_style = ""
        if i == 0:
            extra_style = ", penwidth=3, color=red"

        dot_lines.append(
            f'    {node_id} [label="{label}", fillcolor="{color}", shape="{shape}"{extra_style}];'
        )

    dot_lines.append("")

    # Add edges based on correct dependency calculation
    dependencies = calculate_actual_dependencies(operation_stack)
    for op_idx, deps in dependencies.items():
        # Determine the target node ID
        target_op = operation_stack[op_idx]
        if target_op.op_name.startswith("arg_"):
            target_node_id = target_op.op_name
        else:
            target_node_id = f"op_{op_idx}"

        for dep_idx, input_pos in deps:
            # Determine the source node ID
            source_op = operation_stack[dep_idx]
            if source_op.op_name.startswith("arg_"):
                source_node_id = source_op.op_name
            else:
                source_node_id = f"op_{dep_idx}"

            # Add edge from dependency to current operation with input label
            edge_label = f"input_{input_pos}"
            dot_lines.append(
                f'    {source_node_id} -> {target_node_id} [label="{edge_label}"];'
            )

    dot_lines.extend(
        [
            "",
            "    // Legend",
            "    subgraph cluster_legend {",
            '        label="Legend";',
            "        style=filled;",
            "        fillcolor=white;",
            '        legend_arg [label="arg", fillcolor=lightblue, shape=ellipse];',
            '        legend_const [label="constant", fillcolor=lightgreen, shape=ellipse];',
            '        legend_aten [label="aten ops", fillcolor=lightyellow, shape=box];',
            '        legend_target [label="target", fillcolor=orange, shape=box, penwidth=3, color=red];',
            "    }",
            "}",
        ]
    )

    return "\n".join(dot_lines)


def save_and_render_dot(dot_content: str, filename: str = "operation_stack"):
    """
    Save DOT content to file and render as PNG/PDF.

    Args:
        dot_content: DOT format string
        filename: Base filename (without extension)
    """
    import os

    dot_file = f"{filename}.dot"
    png_file = f"{filename}.png"

    # Get absolute path for clickable link
    abs_png = os.path.abspath(png_file)

    # Save DOT file
    with open(dot_file, "w") as f:
        f.write(dot_content)

    # Render to PNG
    try:
        subprocess.run(["dot", "-Tpng", dot_file, "-o", png_file], check=True)
        print(f"🖼️  View: file://{abs_png}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass


def calculate_subtree_size(op_idx: int, operation_stack: list[Operation]) -> int:
    """
    Recursively calculate the total number of operations in the subtree rooted at op_idx.

    Args:
        op_idx: Index of the root operation
        operation_stack: List of all operations

    Returns:
        Total number of operations in this subtree (including the root)
    """
    if op_idx >= len(operation_stack):
        return 0

    operation = operation_stack[op_idx]
    total_size = 1  # Count this operation itself

    if operation.input_specs:
        current_dep_idx = op_idx + 1
        for input_spec in operation.input_specs:
            if current_dep_idx < len(operation_stack):
                # Recursively calculate subtree size of this dependency
                dep_subtree_size = calculate_subtree_size(
                    current_dep_idx, operation_stack
                )
                total_size += dep_subtree_size
                current_dep_idx += dep_subtree_size

    return total_size


def calculate_actual_dependencies(operation_stack: list[Operation]) -> dict:
    """
    Calculate the actual dependency graph using the same logic as code generation.

    Args:
        operation_stack: List of Operation instances

    Returns:
        Dict mapping operation_idx -> list of (dependency_idx, input_position)
    """

    dependencies = {}  # op_idx -> list of (dependency_idx, input_position)

    # Calculate dependencies using the same logic as code generation
    for i, op in enumerate(operation_stack):
        dependencies[i] = []

        if op.input_specs:
            current_dep_idx = i + 1
            for j, input_spec in enumerate(op.input_specs):
                if current_dep_idx < len(operation_stack):
                    # Verify this dependency is compatible
                    dep_operation = operation_stack[current_dep_idx]
                    if specs_compatible(dep_operation.output_spec, input_spec):
                        dependencies[i].append((current_dep_idx, j))

                        # Calculate subtree size and advance pointer
                        dep_subtree_size = calculate_subtree_size(
                            current_dep_idx, operation_stack
                        )
                        current_dep_idx += dep_subtree_size
                    else:
                        raise RuntimeError("not expected")
                        print(
                            f"⚠️  Dependency mismatch: Op {i} input {j} expects {input_spec}, "
                            f"but op {current_dep_idx} provides {dep_operation.output_spec}"
                        )
                        current_dep_idx += 1

    return dependencies


def visualize_operation_stack(
    operation_stack: list[Operation],
    title: str = "Operation Stack",
    output_folder: str = ".",
):
    """
    Complete visualization pipeline for an operation stack.

    Args:
        operation_stack: List of Operation instances
        title: Title for the visualization
        output_folder: Folder where to save the visualization files
    """
    # Generate DOT content
    dot_content = operation_stack_to_dot(operation_stack, title)

    # Save and render in the specified folder
    import os

    filename = os.path.join(output_folder, "operation_stack")
    save_and_render_dot(dot_content, filename)


# NetworkX alternative for Python-native visualization
def operation_stack_to_networkx(operation_stack: list[Operation]):
    """
    Convert operation stack to NetworkX graph for Python visualization.
    Requires: pip install networkx matplotlib
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print(
            "⚠️  NetworkX/Matplotlib not installed. Run: pip install networkx matplotlib"
        )
        return

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes
    for i, op in enumerate(operation_stack):
        label = f"Op {i}\n{op.op_name}\ndepth {op.depth}"
        G.add_node(i, label=label, operation=op)

    # Add edges (simplified dependency calculation)
    for i, op in enumerate(operation_stack):
        if op.input_specs:
            current_dep_idx = i + 1
            for j, input_spec in enumerate(op.input_specs):
                if current_dep_idx < len(operation_stack):
                    G.add_edge(current_dep_idx, i)
                    current_dep_idx += 1

    # Plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Draw nodes
    node_colors = []
    for i, op in enumerate(operation_stack):
        if op.op_name.startswith("arg_"):
            node_colors.append("lightblue")
        elif op.op_name == "constant":
            node_colors.append("lightgreen")
        elif "aten" in op.op_name:
            node_colors.append("lightyellow")
        else:
            node_colors.append("lightgray")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500)
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20)

    # Draw labels
    labels = {i: f"Op {i}\n{op.op_name}" for i, op in enumerate(operation_stack)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.title("Operation Stack Dependency Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("operation_stack_networkx.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("✓ NetworkX visualization saved as operation_stack_networkx.png")
