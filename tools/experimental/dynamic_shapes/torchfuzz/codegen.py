# mypy: ignore-errors
import os
from typing import Optional

import torch

from torchfuzz.operators import get_operator
from torchfuzz.ops_fuzzer import OperationGraph
from torchfuzz.tensor_descriptor import format_tensor_descriptor
from torchfuzz.tensor_fuzzer import ScalarSpec, Spec, TensorSpec


def convert_graph_to_python_code(
    operation_graph: OperationGraph, seed: Optional[int] = None
) -> str:
    """
    Convert an operation graph to executable Python code using topological ordering.

    The graph-based approach generates code by:
    1. Getting the topological order of nodes (dependencies before dependents)
    2. Generating code for each node in that order
    3. Properly handling input dependencies through node connections

    Args:
        operation_graph: OperationGraph instance containing the operation DAG
        seed: Random seed for reproducible code generation. If None, uses current random state.

    Returns:
        String containing the complete Python code that executes the operations
    """

    # Set seed for reproducible code generation
    if seed is not None:
        import random

        random.seed(seed + 1000)  # Offset to avoid conflicts with graph generation
        torch.manual_seed(seed + 1000)

    if not operation_graph.nodes:
        raise ValueError("Empty operation graph")

    # Get topological order - this ensures dependencies are processed before dependents
    topo_order = operation_graph.get_topological_order()

    # Track generated variables and arg operations
    generated_code_lines = []
    node_variables: dict[str, tuple[str, Spec]] = {}  # Maps node_id to (var_name, spec)
    arg_operations: list[
        tuple[str, Spec]
    ] = []  # List of (node_id, spec) for arg operations

    # Process nodes in topological order
    for node_id in topo_order:
        node = operation_graph.nodes[node_id]
        op_name = node.op_name
        output_spec = node.output_spec

        # Generate output variable name
        output_var_name = f"var_{node_id}"

        # Generate input variable names from input nodes
        input_var_names = []
        for input_node_id in node.input_nodes:
            if input_node_id in node_variables:
                input_var_name, _ = node_variables[input_node_id]
                input_var_names.append(input_var_name)
            else:
                raise ValueError(
                    f"Node {node_id} depends on {input_node_id}, but {input_node_id} "
                    f"was not processed yet. Topological order may be incorrect."
                )

        # Handle different operation types
        if op_name == "arg" or op_name.startswith("arg_"):
            # Track arg operations for later function signature generation
            arg_operations.append((node_id, output_spec))
            arg_name = f"arg_{len(arg_operations) - 1}"
            # Add tensor descriptor comment for arg operations too
            descriptor_comment = f"# {format_tensor_descriptor(output_spec)}"
            operation_lines = [f"{output_var_name} = {arg_name} " + descriptor_comment]
        else:
            # Generate operation execution code
            operation_lines = generate_simple_operation_code(
                output_var_name, input_var_names, op_name, output_spec
            )

        # Add proper indentation for function body
        generated_code_lines.extend(["    " + line for line in operation_lines])

        # Track this node's variable
        node_variables[node_id] = (output_var_name, output_spec)

    # The final result comes from the root node
    root_node_id = operation_graph.root_node_id
    if root_node_id not in node_variables:
        raise ValueError(f"Root node {root_node_id} was not processed")

    final_var_name, _ = node_variables[root_node_id]

    # Generate function signature based on discovered arg operations
    if arg_operations:
        arg_names = [f"arg_{i}" for i in range(len(arg_operations))]
        function_signature = f"def fuzzed_program({', '.join(arg_names)})"
    else:
        function_signature = "def fuzzed_program()"

    # Build the complete code - all imports at the top
    code_lines = [
        "import torch",
        "torch._dynamo.config.capture_scalar_outputs = True",
        "",
    ]

    # Add single seed at the top if seed is provided
    if seed is not None:
        code_lines.append(f"torch.manual_seed({seed})")
        code_lines.append("")

    code_lines.append(function_signature + ":")

    # Add the generated operation code
    code_lines.extend(generated_code_lines)

    # Add return statement
    code_lines.extend(
        [
            f"    return {final_var_name}",
            "",
        ]
    )

    # Generate argument creation code without individual seeds
    if arg_operations:
        for i, (node_id, spec) in enumerate(arg_operations):
            arg_name = f"arg_{i}"

            if isinstance(spec, ScalarSpec):
                dtype_str = f"torch.{spec.dtype}".replace("torch.torch.", "torch.")
                code_lines.append(
                    f"{arg_name} = torch.tensor(torch.randn(()), dtype={dtype_str}).item()"
                )

            elif isinstance(spec, TensorSpec):
                size_str = str(spec.size)
                stride_str = str(spec.stride)
                dtype_str = f"torch.{spec.dtype}".replace("torch.torch.", "torch.")

                # Calculate storage size needed for the strided tensor
                if spec.size:
                    storage_size = 1
                    for dim_size, stride in zip(spec.size, spec.stride):
                        if dim_size > 1:
                            storage_size = max(
                                storage_size, (dim_size - 1) * abs(stride) + 1
                            )
                else:
                    storage_size = 1

                code_lines.append(
                    f"{arg_name} = torch.as_strided(torch.randn({storage_size}).to({dtype_str}), {size_str}, {stride_str})"
                )

    # Generate the final execution with both normal and compiled versions
    if arg_operations:
        arg_names = [f"arg_{i}" for i in range(len(arg_operations))]
        if len(arg_names) == 1:
            args_tuple = (
                f"({arg_names[0]},)"  # Single element tuple needs trailing comma
            )
        else:
            args_tuple = f"({', '.join(arg_names)})"
    else:
        args_tuple = "()"

    code_lines.extend(
        [
            "",
            f"args = {args_tuple}",
            "result_original = fuzzed_program(*args)",
            "print('✅ eager success')",
            "compiled_program = torch.compile(fuzzed_program, fullgraph=False, dynamic=True)",
            "result_compiled = compiled_program(*args)",
            "print('✅ compile success')",
        ]
    )

    return "\n".join(code_lines)


def generate_simple_operation_code(
    output_var: str,
    input_vars: list,
    op_name: str,
    output_spec,
) -> list:
    """
    Generate code lines for executing a single operation using class-based operators.

    Args:
        output_var: Name of the output variable
        input_vars: List of input variable names
        op_name: Name of the operation
        output_spec: Output specification for the operation
    """
    # Try to get the operator from the registry
    operator = get_operator(op_name)

    if operator is not None:
        # Use the class-based operator to generate code
        code_line = operator.codegen(output_var, input_vars, output_spec)
        # Add tensor descriptor comment
        descriptor_comment = f"# {format_tensor_descriptor(output_spec)}"
        return [code_line + " " + descriptor_comment]
    else:
        # Fallback for unknown operations
        return [f"# Unknown operation: {op_name}"]


def create_program_file(python_code: str) -> str:
    """
    Create a temporary Python file from the generated code.

    Args:
        python_code: String containing Python code to write

    Returns:
        Path to the created temporary file
    """
    import random

    # Generate a random nonce for the filename
    nonce = random.randint(0, 1_000_000_000)
    tmp_dir = "/tmp/torchfuzz"
    os.makedirs(tmp_dir, exist_ok=True)
    generated_file_path = os.path.join(tmp_dir, f"fuzz_{nonce}.py")

    # Write the generated code to the specified file
    with open(generated_file_path, "w") as f:
        f.write(python_code)

    return generated_file_path
