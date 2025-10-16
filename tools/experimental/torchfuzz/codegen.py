# mypy: ignore-errors
import os
from typing import Optional

import torch

from torchfuzz.operators import get_operator
from torchfuzz.ops_fuzzer import OperationGraph
from torchfuzz.tensor_descriptor import format_tensor_descriptor
from torchfuzz.tensor_fuzzer import ScalarSpec, Spec, TensorSpec


class FuzzTemplate:
    def __init__(self, supported_ops, check):
        self.supported_ops = supported_ops
        self.check = check

    def supported_dtypes(self):
        """Return list of supported dtypes for this template."""
        return [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.bool,
        ]

    def spec_distribution(self):
        """
        Define the distribution for generating random Specs.

        Returns:
            Dict with keys:
            - 'tensor_prob': Probability of generating TensorSpec (0.0 to 1.0)
            - 'scalar_prob': Probability of generating ScalarSpec (0.0 to 1.0)
            - 'allow_tensors': Whether TensorSpec generation is allowed (boolean)
            - 'allow_scalars': Whether ScalarSpec generation is allowed (boolean)
        """
        return {
            "tensor_prob": 0.8,
            "scalar_prob": 0.2,
            "allow_tensors": True,
            "allow_scalars": True,
        }

    def fuzz_spec_custom(self):
        """
        Generate a random Spec based on this template's distribution preferences.

        Returns:
            Spec: Either a TensorSpec or ScalarSpec according to template's distribution
        """
        import random

        from torchfuzz.tensor_fuzzer import fuzz_torch_tensor_type

        # Get template's distribution configuration
        distribution = self.spec_distribution()

        # Get random dtype based on template
        dtype = fuzz_torch_tensor_type("default")

        # Validate distribution configuration
        allow_tensors = distribution.get("allow_tensors", True)
        allow_scalars = distribution.get("allow_scalars", True)

        if not allow_tensors and not allow_scalars:
            raise ValueError("Template must allow at least one of tensors or scalars")

        # Determine which type to generate
        if not allow_scalars:
            # Only tensors allowed
            return self._generate_tensor_spec(dtype)
        elif not allow_tensors:
            # Only scalars allowed
            return self._generate_scalar_spec(dtype)
        else:
            # Both allowed, use probability distribution
            tensor_prob = distribution.get("tensor_prob", 0.8)
            if random.random() < tensor_prob:
                return self._generate_tensor_spec(dtype)
            else:
                return self._generate_scalar_spec(dtype)

    def _generate_tensor_spec(self, dtype):
        """Generate a TensorSpec with the given dtype."""
        from torchfuzz.tensor_fuzzer import (
            fuzz_tensor_size,
            fuzz_valid_stride,
            TensorSpec,
        )

        size = fuzz_tensor_size()
        stride = fuzz_valid_stride(size)
        return TensorSpec(size=size, stride=stride, dtype=dtype)

    def _generate_scalar_spec(self, dtype):
        """Generate a ScalarSpec with the given dtype."""
        from torchfuzz.tensor_fuzzer import ScalarSpec

        return ScalarSpec(dtype=dtype)

    def args_codegen(self, arg_operations):
        """Generate argument creation code for default template."""
        code_lines = []

        # Add sentinel tensor that ensures gradient computation
        code_lines.extend(
            [
                "# Sentinel tensor to ensure gradient computation",
                "sentinel = torch.tensor(1.0, requires_grad=True)",
                "",
            ]
        )

        if arg_operations:
            for i, (node_id, spec) in enumerate(arg_operations):
                arg_name = f"arg_{i}"

                if isinstance(spec, ScalarSpec):
                    dtype_str = f"torch.{spec.dtype}".replace("torch.torch.", "torch.")
                    if spec.dtype in [
                        torch.int8,
                        torch.int16,
                        torch.int32,
                        torch.int64,
                    ]:
                        # For integer scalars, use randint to avoid always getting 0
                        code_lines.append(
                            f"{arg_name} = int(torch.randint(5, 30, ()).item())"
                        )
                    elif spec.dtype == torch.bool:
                        # For boolean scalars, use randint and cast to bool
                        code_lines.append(
                            f"{arg_name} = bool(torch.randint(0, 2, ()).item())"
                        )
                    else:
                        # For float scalars, use randn
                        code_lines.append(
                            f"{arg_name} = float(torch.randn((), dtype={dtype_str}).item())"
                        )

                elif isinstance(spec, TensorSpec):
                    size_str = str(spec.size)
                    dtype_str = f"torch.{spec.dtype}".replace("torch.torch.", "torch.")

                    # Calculate storage size needed for the strided tensor
                    if spec.size:
                        # Calculate the maximum index that will be accessed
                        max_offset = 0
                        for dim_size, stride in zip(spec.size, spec.stride):
                            if dim_size > 1:
                                max_offset += (dim_size - 1) * abs(stride)
                        storage_size = max_offset + 1
                    else:
                        storage_size = 1

                    stride_str = str(spec.stride)

                    # Special handling for integer tensors which might be used as indices
                    if spec.dtype in [
                        torch.int8,
                        torch.int16,
                        torch.int32,
                        torch.int64,
                    ]:
                        # For integer tensors, generate valid indices with headroom for arithmetic
                        # Use smaller range [5, 30] to allow for multiplication and other operations
                        # This prevents indices from becoming too large after arithmetic
                        min_val = (
                            5  # Minimum to avoid negative results after subtraction
                        )
                        max_val = (
                            30  # Maximum to avoid out-of-bounds after multiplication
                        )
                        code_lines.append(
                            f"{arg_name} = torch.as_strided(torch.randint({min_val}, {max_val}, ({storage_size},)).to({dtype_str}), {size_str}, {stride_str})"
                        )
                    elif spec.dtype == torch.bool:
                        # For boolean tensors, use randint to generate True/False values
                        # Using randn().to(bool) would yield almost all True due to non-zero floats
                        code_lines.append(
                            f"{arg_name} = torch.as_strided(torch.randint(0, 2, ({storage_size},), dtype=torch.int8).bool(), {size_str}, {stride_str})"
                        )
                    else:
                        code_lines.append(
                            f"{arg_name} = torch.as_strided(torch.randn({storage_size}).to({dtype_str}), {size_str}, {stride_str})"
                        )

        return code_lines


class DefaultFuzzTemplate(FuzzTemplate):
    def __init__(self):
        from torchfuzz.checks import EagerVsFullGraphDynamicCompileWithNumericsCheck

        super().__init__(
            supported_ops=[
                # Basic arithmetic operations
                "torch.add",
                "torch.sub",
                "torch.mul",
                "torch.div",
                # Tensor shape operations
                "torch.Tensor.view",
                "torch.reshape",
                "torch.flatten",
                "torch.squeeze",
                "torch.unsqueeze",
                # Matrix operations
                "torch.mm",
                "torch.addmm",
                "torch.bmm",
                "torch.matmul",
                # Neural network operations
                "torch.nn.functional.embedding",
                "torch.nn.functional.linear",
                # Activation functions
                "torch.nn.functional.relu",
                "torch.nn.functional.leaky_relu",
                "torch.nn.functional.elu",
                "torch.nn.functional.gelu",
                "torch.nn.functional.silu",
                "torch.sigmoid",
                "torch.tanh",
                "torch.nn.functional.softmax",
                # Normalization layers
                "torch.nn.functional.layer_norm",
                "torch.nn.functional.rms_norm",
                "torch.nn.functional.batch_norm",
                "torch.nn.functional.group_norm",
                # Regularization
                "torch.nn.functional.dropout",
            ],
            check=EagerVsFullGraphDynamicCompileWithNumericsCheck(),
        )

    def spec_distribution(self):
        """Default template: tensor-only (no scalars)."""
        return {
            "tensor_prob": 1.0,
            "scalar_prob": 0.0,
            "allow_tensors": True,
            "allow_scalars": False,
        }

    def imports_codegen(self):
        return [
            "import torch",
        ]

    def flags_codegen(self):
        return ["torch._dynamo.config.capture_scalar_outputs = True"]

    def epilogue_codegen(self):
        return []


class DTensorFuzzTemplate(FuzzTemplate):
    def __init__(self):
        from torchfuzz.checks import EagerVsFullGraphDynamicCompileCheck

        super().__init__(
            supported_ops=[
                "torch.add",
                "torch.sub",
                "torch.mul",
                "torch.div",
                "torch.mm",
                "torch.addmm",
                "torch.bmm",
                "torch.matmul",
            ],
            check=EagerVsFullGraphDynamicCompileCheck(),
        )

    def supported_dtypes(self):
        """Return list of DTensor-compatible dtypes (no complex types)."""
        return [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.bool,
        ]

    def spec_distribution(self):
        """DTensor template: tensor-only (no scalars)."""
        return {
            "tensor_prob": 1.0,
            "scalar_prob": 0.0,
            "allow_tensors": True,
            "allow_scalars": False,
        }

    def imports_codegen(self):
        return [
            "import torch",
            "from torch.distributed.tensor.placement_types import Replicate, Shard",
            "from torch.testing._internal.distributed.fake_pg import FakeStore",
            "from torch.distributed.tensor import DTensor",
        ]

    def flags_codegen(self):
        return [
            "torch._dynamo.config.capture_scalar_outputs = True",
            "torch._dynamo.config.capture_dynamic_output_shape_ops = True",
            "torch._inductor.config.emulate_precision_casts = True",
        ]

    def args_codegen(self, arg_operations):
        """Generate DTensor argument creation code with proper mesh setup."""
        code_lines = []

        # Add DTensor setup code first
        code_lines.extend(
            [
                "world_size = 1024",
                "fake_store = FakeStore()",
                "torch.distributed.init_process_group(",
                '    "fake", store=fake_store, rank=0, world_size=world_size',
                ")",
                "",
                "mesh = torch.distributed.device_mesh.init_device_mesh(",
                '    "cuda",',
                "    (2, 8),",
                "    mesh_dim_names=(",
                '        "dim1", "dim2",',
                "    ),",
                ")",
                "",
                "placements = (Replicate(), Replicate())",
                "",
                "# Sentinel tensor to ensure gradient computation",
                "sentinel_local = torch.tensor(1.0, device='cuda', requires_grad=True)",
                "sentinel = DTensor.from_local(sentinel_local, mesh, placements)",
                "",
            ]
        )

        if arg_operations:
            for i, (node_id, spec) in enumerate(arg_operations):
                arg_name = f"arg_{i}"

                if isinstance(spec, ScalarSpec):
                    # For scalars in DTensor, create a 0-dim tensor
                    dtype_str = f"torch.{spec.dtype}".replace("torch.torch.", "torch.")
                    code_lines.extend(
                        [
                            f"{arg_name}_local = torch.randn((), dtype={dtype_str}, device='cuda', requires_grad=True)",
                            f"{arg_name} = DTensor.from_local({arg_name}_local, mesh, placements)",
                        ]
                    )

                elif isinstance(spec, TensorSpec):
                    size_str = str(spec.size)
                    dtype_str = f"torch.{spec.dtype}".replace("torch.torch.", "torch.")

                    # Handle different dtypes appropriately for DTensor
                    if spec.dtype in [
                        torch.int32,
                        torch.int64,
                        torch.int8,
                        torch.int16,
                    ]:
                        # Integer dtypes: use randint and no requires_grad
                        code_lines.extend(
                            [
                                f"{arg_name}_local = torch.randint(1, 10, {size_str}, dtype={dtype_str}, device='cuda')",
                                f"{arg_name} = DTensor.from_local({arg_name}_local, mesh, placements)",
                            ]
                        )
                    elif spec.dtype == torch.bool:
                        # Boolean dtype: use randint and cast to bool
                        code_lines.extend(
                            [
                                f"{arg_name}_local = torch.randint(0, 2, {size_str}, device='cuda').bool()",
                                f"{arg_name} = DTensor.from_local({arg_name}_local, mesh, placements)",
                            ]
                        )
                    else:
                        # Float dtypes: use randn and requires_grad
                        code_lines.extend(
                            [
                                f"{arg_name}_local = torch.randn({size_str}, dtype={dtype_str}, device='cuda', requires_grad=True)",
                                f"{arg_name} = DTensor.from_local({arg_name}_local, mesh, placements)",
                            ]
                        )

        return code_lines

    def epilogue_codegen(self):
        return ["torch.distributed.destroy_process_group()"]


class UnbackedFuzzTemplate(FuzzTemplate):
    def __init__(self):
        from torchfuzz.checks import EagerVsFullGraphDynamicCompileCheck

        super().__init__(
            supported_ops=[
                "torch.ops.aten.item",
                "torch.ops.aten.nonzero",
                "torch.ops.aten.masked_select",
                "torch.ops.aten.unique",
                # Basic arithmetic operations
                "torch.add",
                "torch.sub",
                "torch.mul",
                "torch.div",
                # Tensor shape operations
                "torch.Tensor.view",
                "torch.reshape",
                "torch.flatten",
                "torch.squeeze",
                "torch.unsqueeze",
                # Matrix operations
                "torch.mm",
                "torch.addmm",
                "torch.bmm",
                "torch.matmul",
                # Neural network operations
                "torch.nn.functional.embedding",
                "torch.nn.functional.linear",
                # Activation functions
                "torch.nn.functional.relu",
                "torch.nn.functional.leaky_relu",
                "torch.nn.functional.elu",
                "torch.nn.functional.gelu",
                "torch.nn.functional.silu",
                "torch.sigmoid",
                "torch.tanh",
                "torch.nn.functional.softmax",
                # Normalization layers
                "torch.nn.functional.layer_norm",
                "torch.nn.functional.rms_norm",
                "torch.nn.functional.batch_norm",
                "torch.nn.functional.group_norm",
                # Regularization
                "torch.nn.functional.dropout",
            ],
            check=EagerVsFullGraphDynamicCompileCheck(),
        )

    def supported_dtypes(self):
        """Return list of dtypes good for data-dependent operations."""
        # Focus on dtypes that work well with data-dependent ops and arithmetic
        # Exclude bool since arithmetic operations don't work with boolean tensors
        return [
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
        ]

    def spec_distribution(self):
        """Unbacked template: 50% tensors, 50% scalars."""
        return {
            "tensor_prob": 0.5,
            "scalar_prob": 0.5,
            "allow_tensors": True,
            "allow_scalars": True,
        }

    def imports_codegen(self):
        return [
            "import torch",
        ]

    def flags_codegen(self):
        return [
            "torch._dynamo.config.capture_scalar_outputs = True",
            "torch._dynamo.config.capture_dynamic_output_shape_ops = True",
        ]

    def epilogue_codegen(self):
        return []


def convert_graph_to_python_code(
    operation_graph: OperationGraph,
    seed: Optional[int] = None,
    template: str = "default",
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

    # Instantiate template
    if template == "dtensor":
        fuzz_template = DTensorFuzzTemplate()
    elif template == "unbacked":
        fuzz_template = UnbackedFuzzTemplate()
    else:
        fuzz_template = DefaultFuzzTemplate()

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
        function_signature = f"def fuzzed_program({', '.join(arg_names)}, sentinel)"
    else:
        function_signature = "def fuzzed_program(sentinel)"

    # Build the complete code - all imports at the top
    code_lines = []

    # Add template imports
    code_lines.extend(fuzz_template.imports_codegen())

    # Add template flags
    code_lines.extend(fuzz_template.flags_codegen())
    code_lines.append("")

    # Add single seed at the top if seed is provided
    if seed is not None:
        code_lines.append(f"torch.manual_seed({seed})")
        code_lines.append("")

    code_lines.append(function_signature + ":")

    # Add the generated operation code
    code_lines.extend(generated_code_lines)

    # Add return statement with sentinel multiplication to ensure gradient computation
    # Handle complex tensors appropriately based on template
    if template == "dtensor":
        # For DTensor, avoid .real operation which doesn't work with sharding
        # Instead use abs() for complex tensors to get a real result
        code_lines.extend(
            [
                "    # Ensure gradient computation by multiplying with sentinel",
                f"    result = {final_var_name} * sentinel",
                "    if result.is_complex():",
                "        result = result.abs()  # Use abs() instead of .real for DTensor compatibility",
                "    return result",
                "",
            ]
        )
    else:
        code_lines.extend(
            [
                "    # Ensure gradient computation by multiplying with sentinel and taking real part",
                f"    result = {final_var_name} * sentinel",
                "    if result.is_complex():",
                "        result = result.real",
                "    return result",
                "",
            ]
        )

    # Generate argument creation code using template
    arg_code_lines = fuzz_template.args_codegen(arg_operations)
    code_lines.extend(arg_code_lines)

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

    # Generate execution code using template check
    check_lines = fuzz_template.check.codegen(f"{args_tuple} + (sentinel,)")
    code_lines.extend([""] + check_lines)

    # Add template epilogue
    epilogue_lines = fuzz_template.epilogue_codegen()
    if epilogue_lines:
        code_lines.append("")
        code_lines.extend(epilogue_lines)

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
        code = operator.codegen(output_var, input_vars, output_spec)
        # Add tensor descriptor comment to the last emitted line
        descriptor_comment = f"# {format_tensor_descriptor(output_spec)}"
        if "\n" in code:
            lines = code.split("\n")
            # Attach comment to the last non-empty line
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip():
                    lines[i] = lines[i] + " " + descriptor_comment
                    break
            return lines
        else:
            return [code + " " + descriptor_comment]
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
    import hashlib

    # Generate a deterministic filename based on code content hash
    code_hash = hashlib.md5(python_code.encode()).hexdigest()[:8]  # noqa: S324
    tmp_dir = "/tmp/torchfuzz"
    os.makedirs(tmp_dir, exist_ok=True)
    generated_file_path = os.path.join(tmp_dir, f"fuzz_{code_hash}.py")

    # Write the generated code to the specified file
    with open(generated_file_path, "w") as f:
        f.write(python_code)

    return generated_file_path
