# mypy: ignore-errors
import hashlib
import importlib
import os
import random
from collections.abc import Callable
from dataclasses import dataclass

import torch

from torchfuzz.operators import get_operator
from torchfuzz.ops_fuzzer import OperationGraph
from torchfuzz.tensor_descriptor import format_tensor_descriptor
from torchfuzz.tensor_fuzzer import ScalarSpec, Spec, TensorSpec


_DEFAULT_DEVICE_MODULE = "torchfuzz.cuda"


@dataclass
class DeviceInfo:
    """Per-device metadata returned by a torchfuzz device plugin.

    Attributes:
        device_name: Short device name (e.g. "cuda", "xpu", "mtia").  Used by
            ``tensor_descriptor`` to label the device in emitted comments.
        select_runtime_env: Optional callback that customizes the subprocess
            environment passed to ``runner.py``.  Signature::

                def select_runtime_env(
                    env: dict[str, str],
                    *,
                    exclude_primary_device: bool = False,
                ) -> dict[str, str]: ...

            *env* is the current environment (``PYTHONPATH`` is already set by
            the runner).  When *exclude_primary_device* is ``True`` the plugin
            should avoid selecting device 0 (or the equivalent primary device)
            to prevent contention with the orchestrating process.  Return a
            (new) ``env`` dict; when the callback is ``None`` the runner uses
            the environment as-is.
    """

    device_name: str
    select_runtime_env: Callable[..., dict[str, str]] | None = None


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
        # Get template's distribution configuration
        distribution = self.spec_distribution()

        # Get random dtype based on this template's supported dtypes
        dtype = random.choice(self.supported_dtypes())

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

    def imports_codegen(self):
        """Return import lines emitted at the top of the generated file."""
        return []

    def flags_codegen(self):
        """Return flag/setup lines emitted at the top of the generated file."""
        return []

    def epilogue_codegen(self):
        """Return lines emitted at the very end of the generated file."""
        return []

    def treat_constant_as_global(self) -> bool:
        """Whether ``constant`` ops should be lifted to function arguments.

        When True, ``constant`` ops are appended to ``constant_operations`` and
        passed into ``args_codegen``; the function signature receives them as
        explicit parameters.  Templates that materialize constants outside the
        traced function (e.g. DTensor placements with random sharding) should
        return True.  Default is False.
        """
        return False

    def wrap_body(
        self, generated_code_lines: list[str], graph: OperationGraph
    ) -> list[str]:
        """Optionally rewrite the per-node body lines.

        Default is a passthrough.  Used by ``StreamFuzzTemplate`` to partition
        operations across CUDA streams.
        """
        return generated_code_lines

    def return_codegen(self, final_var_name: str) -> list[str]:
        """Return the lines that emit the ``return`` statement.

        Default uses ``.real`` to drop imaginary parts of complex outputs.
        """
        return [
            "    # Ensure gradient computation by multiplying with sentinel and taking real part",
            f"    result = {final_var_name} * sentinel",
            "    if result.is_complex():",
            "        result = result.real",
            "    return result",
            "",
        ]

    def codegen_constant(self, output_name: str, tensor_creation_expr: str) -> str:
        """Wrap a tensor-creation expression for use as a ``constant`` op.

        Default just assigns the expression to ``output_name``.  Templates that
        need to wrap constants (e.g. DTensor's ``DTensor.from_local``) should
        override this hook.
        """
        return f"{output_name} = {tensor_creation_expr}"

    def args_codegen(self, arg_operations, constant_operations=None):
        """Generate argument creation code for default template.

        ``constant_operations`` is only consulted by templates that opt in via
        :meth:`treat_constant_as_global`.  The base implementation ignores it.
        """
        del constant_operations  # unused in the base implementation

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


# ---------------------------------------------------------------------------
# Plugin registry
# ---------------------------------------------------------------------------


_TEMPLATE_REGISTRY: dict[str, type[FuzzTemplate]] | None = None
_DEVICE_INFO: DeviceInfo | None = None


def initialize_codegen() -> None:
    """Load the device plugin module and populate the template registry.

    Idempotent.  Called explicitly from ``fuzzer.py`` and lazily from
    :func:`make_template` / :func:`get_template_names` / :func:`get_device_info`
    so that library callers do not have to remember to invoke it themselves.

    The plugin module name comes from the ``TORCHFUZZ_DEVICE_MODULE`` environment
    variable; if unset, the default ``torchfuzz.cuda`` plugin is loaded.
    """
    global _TEMPLATE_REGISTRY, _DEVICE_INFO
    if _TEMPLATE_REGISTRY is not None:
        return
    module_name = os.environ.get("TORCHFUZZ_DEVICE_MODULE", _DEFAULT_DEVICE_MODULE)
    plugin = importlib.import_module(module_name)
    _TEMPLATE_REGISTRY = plugin.register_codegen()
    _DEVICE_INFO = plugin.get_device_info()


def get_template_names() -> list[str]:
    """Return the list of template names registered by the active plugin."""
    initialize_codegen()
    return list(_TEMPLATE_REGISTRY.keys())


def make_template(name: str) -> FuzzTemplate:
    """Instantiate the FuzzTemplate registered under ``name``."""
    initialize_codegen()
    if name not in _TEMPLATE_REGISTRY:
        raise KeyError(
            f"Unknown template '{name}'; available templates: "
            f"{sorted(_TEMPLATE_REGISTRY.keys())}"
        )
    return _TEMPLATE_REGISTRY[name]()


def get_device_info() -> DeviceInfo:
    """Return the active plugin's :class:`DeviceInfo`."""
    initialize_codegen()
    return _DEVICE_INFO


def convert_graph_to_python_code(
    operation_graph: OperationGraph,
    seed: int | None = None,
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

    # Instantiate template via the device plugin registry.
    fuzz_template = make_template(template)

    # Set seed for reproducible code generation
    if seed is not None:
        random.seed(seed + 1000)  # Offset to avoid conflicts with graph generation
        torch.manual_seed(seed + 1000)

    if not operation_graph.nodes:
        raise ValueError("Empty operation graph")

    # Get topological order - this ensures dependencies are processed before dependents
    topo_order = operation_graph.get_topological_order()

    constant_as_global = fuzz_template.treat_constant_as_global()

    # Track generated variables, arg operations, and constant operations
    generated_code_lines = []
    node_variables: dict[str, tuple[str, Spec]] = {}  # Maps node_id to (var_name, spec)
    arg_operations: list[
        tuple[str, Spec]
    ] = []  # List of (node_id, spec) for arg operations
    constant_operations: list[
        tuple[str, str, Spec]
    ] = []  # List of (node_id, var_name, spec) for templates that lift constants out

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
        elif op_name == "constant" and constant_as_global:
            # Track constants to create them outside the function
            constant_operations.append((node_id, output_var_name, output_spec))
            descriptor_comment = f"# {format_tensor_descriptor(output_spec)}"
            operation_lines = [
                f"{output_var_name} = {output_var_name} " + descriptor_comment
            ]
        else:
            # Generate operation execution code
            operation_lines = generate_simple_operation_code(
                output_var_name, input_var_names, op_name, output_spec
            )

        # Add proper indentation for function body
        generated_code_lines.extend(["    " + line for line in operation_lines])

        # Track this node's variable
        node_variables[node_id] = (output_var_name, output_spec)

    # Optional template-driven body rewrite (e.g. CUDA stream wrapping).
    generated_code_lines = fuzz_template.wrap_body(
        generated_code_lines, operation_graph
    )

    # The final result comes from the root node
    root_node_id = operation_graph.root_node_id
    if root_node_id not in node_variables:
        raise ValueError(f"Root node {root_node_id} was not processed")

    final_var_name, _ = node_variables[root_node_id]

    # Generate function signature based on discovered arg and constant operations
    param_names = []
    if arg_operations:
        param_names.extend([f"arg_{i}" for i in range(len(arg_operations))])
    if constant_as_global and constant_operations:
        param_names.extend([var_name for _, var_name, _ in constant_operations])
    param_names.append("sentinel")

    function_signature = f"def fuzzed_program({', '.join(param_names)})"

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

    # Add return statement (template-controlled to handle complex tensors etc.)
    code_lines.extend(fuzz_template.return_codegen(final_var_name))

    # Generate argument creation code using template (always pass constant_operations;
    # templates that don't opt in via treat_constant_as_global ignore the second arg).
    arg_code_lines = fuzz_template.args_codegen(arg_operations, constant_operations)
    code_lines.extend(arg_code_lines)

    # Generate the final execution with both normal and compiled versions
    param_values = []
    if arg_operations:
        param_values.extend([f"arg_{i}" for i in range(len(arg_operations))])
    if constant_as_global and constant_operations:
        param_values.extend([var_name for _, var_name, _ in constant_operations])
    param_values.append("sentinel")

    if len(param_values) == 1:
        args_tuple = (
            f"({param_values[0]},)"  # Single element tuple needs trailing comma
        )
    else:
        args_tuple = f"({', '.join(param_values)})"

    # Generate execution code using template check
    check_lines = fuzz_template.check.codegen(args_tuple)
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
    # Generate a deterministic filename based on code content hash
    code_hash = hashlib.md5(python_code.encode()).hexdigest()[:8]  # noqa: S324
    tmp_dir = "/tmp/torchfuzz"
    os.makedirs(tmp_dir, exist_ok=True)
    generated_file_path = os.path.join(tmp_dir, f"fuzz_{code_hash}.py")

    # Write the generated code to the specified file
    with open(generated_file_path, "w") as f:
        f.write(python_code)

    return generated_file_path
