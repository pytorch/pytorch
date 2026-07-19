# mypy: ignore-errors
"""FuzzTemplate subclasses provided by the CUDA device plugin.

These templates are returned by ``torchfuzz.cuda.register_codegen()`` and
override the device-agnostic hooks defined on
:class:`torchfuzz.codegen.FuzzTemplate` (``treat_constant_as_global``,
``wrap_body``, ``return_codegen``, ``args_codegen``, ``codegen_constant``)
to inject CUDA-specific behaviour into the code that
``convert_graph_to_python_code`` emits.
"""

import random

import torch

from torchfuzz.codegen import FuzzTemplate
from torchfuzz.ops_fuzzer import OperationGraph
from torchfuzz.tensor_fuzzer import ScalarSpec, TensorSpec


class DefaultFuzzTemplate(FuzzTemplate):
    def __init__(self):
        from torchfuzz.cuda._checks import EagerVsFullGraphDynamicCompileCheck

        super().__init__(
            supported_ops=[
                # Basic arithmetic operations
                "torch.add",
                "torch.sub",
                "torch.mul",
                "torch.div",
                "torch.clamp",
                "torch.cumsum",
                # Tensor shape operations
                "torch.Tensor.view",
                "torch.reshape",
                "torch.flatten",
                "torch.squeeze",
                "torch.unsqueeze",
                "torch.split",
                "torch.chunk",
                "torch.expand",
                "torch.cat",
                "torch.stack",
                # Indexing operations
                "torch.gather",
                "torch.index_select",
                "torch.argsort",
                # Matrix operations
                "torch.mm",
                "torch.addmm",
                "torch.bmm",
                "torch.matmul",
                # Neural network operations
                "torch.nn.functional.embedding",
                "torch.nn.functional.linear",
                "torch.nn.functional.scaled_dot_product_attention",
                "torch.nn.functional.multi_head_attention_forward",
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
        return [
            "torch.set_default_device('cuda')",
            "torch._dynamo.config.capture_scalar_outputs = True",
        ]

    def epilogue_codegen(self):
        return []


class DTensorFuzzTemplate(FuzzTemplate):
    def __init__(self):
        from torchfuzz.cuda._checks import EagerVsFullGraphDynamicCompileCheck

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

    def args_codegen(self, arg_operations, constant_operations=None):
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

    def return_codegen(self, final_var_name: str) -> list[str]:
        """DTensor return: avoid .real (incompatible with sharding); use .abs() for complex."""
        return [
            "    # Ensure gradient computation by multiplying with sentinel",
            f"    result = {final_var_name} * sentinel",
            "    if result.is_complex():",
            "        result = result.abs()  # Use abs() instead of .real for DTensor compatibility",
            "    return result",
            "",
        ]

    def codegen_constant(self, output_name: str, tensor_creation_expr: str) -> str:
        """DTensor constants live on cuda and are wrapped via DTensor.from_local."""
        return (
            f"{output_name}_local = {tensor_creation_expr}.to('cuda')\n"
            f"{output_name} = DTensor.from_local({output_name}_local, mesh, placements)"
        )


class UnbackedFuzzTemplate(FuzzTemplate):
    def __init__(self):
        from torchfuzz.cuda._checks import EagerVsFullGraphDynamicCompileCheck

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
            "torch.set_default_device('cuda')",
            "torch._dynamo.config.capture_scalar_outputs = True",
            "torch._dynamo.config.capture_dynamic_output_shape_ops = True",
        ]

    def epilogue_codegen(self):
        return []


class StreamFuzzTemplate(DefaultFuzzTemplate):
    """Template that wraps operations in random CUDA stream contexts.

    Reuses the same operator set as DefaultFuzzTemplate but partitions non-leaf
    operations across 2-3 CUDA streams, inserting proper wait_stream
    synchronization between dependent operations on different streams.
    """

    def __init__(self):
        super().__init__()
        from torchfuzz.cuda._checks import (
            EagerVsFullGraphDynamicCompileWithBackwardCheck,
        )

        self.check = EagerVsFullGraphDynamicCompileWithBackwardCheck()

    def imports_codegen(self):
        return [
            "import torch",
        ]

    def flags_codegen(self):
        return [
            "torch.set_default_device('cuda')",
            "torch._dynamo.config.capture_scalar_outputs = True",
        ]

    def args_codegen(self, arg_operations, constant_operations=None):
        """Generate args with requires_grad=True on float tensors.

        This ensures the backward pass traces through stream-wrapped operations,
        exercising Inductor's stream handling in the backward graph.
        """
        code_lines = super().args_codegen(arg_operations, constant_operations)
        if arg_operations:
            for i, (node_id, spec) in enumerate(arg_operations):
                if isinstance(spec, TensorSpec) and spec.dtype in [
                    torch.float32,
                    torch.float64,
                    torch.float16,
                    torch.bfloat16,
                ]:
                    code_lines.append(f"arg_{i} = arg_{i}.requires_grad_(True)")
        return code_lines

    def wrap_body(
        self, generated_code_lines: list[str], graph: OperationGraph
    ) -> list[str]:
        return self.wrap_body_with_streams(generated_code_lines, graph)

    @staticmethod
    def wrap_body_with_streams(
        generated_code_lines: list[str],
        graph: OperationGraph,
    ) -> list[str]:
        """Wrap generated function body lines with CUDA stream contexts.

        Assigns each non-leaf operation to one of 2-3 random streams, wraps each
        in ``with torch.cuda.stream(sN):``, and inserts ``wait_stream`` calls
        between dependent operations on different streams.
        """
        topo_order = graph.get_topological_order()

        # Identify leaf vs non-leaf node ids
        leaf_ids = set()
        non_leaf_ids = []
        for nid in topo_order:
            node = graph.nodes[nid]
            if (
                node.op_name == "arg"
                or node.op_name.startswith("arg_")
                or node.op_name == "constant"
            ):
                leaf_ids.add(nid)
            else:
                non_leaf_ids.append(nid)

        if not non_leaf_ids:
            return generated_code_lines

        num_streams = random.randint(2, 3)
        stream_names = [f"s{i + 1}" for i in range(num_streams)]

        # Decide sync strategy: wait_stream or event-based (record + wait_event)
        use_events = random.choice([True, False])
        event_counter = 0

        # Assign each non-leaf node to a random stream
        node_stream: dict[str, str] = {}
        for nid in non_leaf_ids:
            node_stream[nid] = random.choice(stream_names)

        # Build a mapping from node_id -> the original code lines for that node.
        # Each node produces lines prefixed with "    " (4-space indent for the
        # function body).  We identify nodes by their ``var_{node_id} =`` pattern.
        node_lines: dict[str, list[str]] = {}
        current_node: str | None = None
        current_buf: list[str] = []

        for line in generated_code_lines:
            stripped = line.strip()
            # Detect lines like "var_node_3 = ..." or "var_node_3, _ = ..."
            matched_node = None
            for nid in topo_order:
                if stripped.startswith((f"var_{nid} =", f"var_{nid},")):
                    matched_node = nid
                    break

            if matched_node is not None:
                # Flush previous node buffer
                if current_node is not None:
                    node_lines[current_node] = current_buf
                current_node = matched_node
                current_buf = [line]
            else:
                current_buf.append(line)

        # Flush last node
        if current_node is not None:
            node_lines[current_node] = current_buf

        # Rebuild the body with stream contexts and synchronization
        new_lines: list[str] = []

        # Stream variable declarations at the top of the function body
        for sname in stream_names:
            new_lines.append(f"    {sname} = torch.cuda.Stream()")

        for nid in topo_order:
            lines_for_node = node_lines.get(nid, [])
            if nid in leaf_ids:
                # Leaf nodes (args) stay on the default stream
                new_lines.extend(lines_for_node)
                continue

            stream = node_stream[nid]
            node = graph.nodes[nid]

            # Insert synchronization for cross-stream dependencies
            waited: set[str] = set()
            for dep_id in node.input_nodes:
                if dep_id in node_stream and node_stream[dep_id] != stream:
                    dep_stream = node_stream[dep_id]
                    if dep_stream not in waited:
                        if use_events:
                            ename = f"e{event_counter}"
                            event_counter += 1
                            new_lines.append(f"    {ename} = torch.cuda.Event()")
                            new_lines.append(f"    {ename}.record({dep_stream})")
                            new_lines.append(f"    {stream}.wait_event({ename})")
                        else:
                            new_lines.append(f"    {stream}.wait_stream({dep_stream})")
                        waited.add(dep_stream)

            # Wrap the operation in a stream context
            new_lines.append(f"    with torch.cuda.stream({stream}):")
            for code_line in lines_for_node:
                # Each line already has 4-space indent; add 4 more for the with block
                new_lines.append("    " + code_line)

        # Synchronize all streams before the return statement
        if use_events:
            for sname in stream_names:
                ename = f"e{event_counter}"
                event_counter += 1
                new_lines.append(f"    {ename} = torch.cuda.Event()")
                new_lines.append(f"    {ename}.record({sname})")
                new_lines.append(f"    torch.cuda.current_stream().wait_event({ename})")
        else:
            for sname in stream_names:
                new_lines.append(
                    f"    torch.cuda.current_stream().wait_stream({sname})"
                )

        return new_lines


class DTensorFuzzPlacementsTemplate(DTensorFuzzTemplate):
    """DTensor template with randomized placements (Replicate, Shard, Partial).

    Extends DTensorFuzzTemplate to randomize placement strategies instead of
    using fixed (Replicate(), Replicate()) for all tensors.
    """

    def fuzz_spec_custom(self):
        """Generate tensor specs with minimum 1 dimension for proper DTensor sharding."""
        from torchfuzz.tensor_fuzzer import fuzz_valid_stride

        # Get random dtype
        dtype = random.choice(self.supported_dtypes())

        # Generate tensor size with minimum 1 dimension (avoid 0-dim scalars)
        # Prefer 2D-3D tensors for interesting sharding patterns
        ndim = random.choices([1, 2, 3, 4], weights=[0.1, 0.5, 0.3, 0.1])[0]
        size = tuple(random.randint(2, 32) for _ in range(ndim))
        stride = fuzz_valid_stride(size)

        return TensorSpec(size=size, stride=stride, dtype=dtype)

    def imports_codegen(self):
        """Add Partial to imports."""
        base_imports = super().imports_codegen()
        # Update the placement imports to include Partial
        for i, imp in enumerate(base_imports):
            if "placement_types import" in imp:
                base_imports[i] = (
                    "from torch.distributed.tensor.placement_types import Replicate, Shard, Partial"
                )
                break
        base_imports.append("import torch.distributed.tensor as dist_tensor")
        return base_imports

    def treat_constant_as_global(self) -> bool:
        """Constants are created outside the function and passed in as args."""
        return True

    def codegen_constant(self, output_name: str, tensor_creation_expr: str) -> str:
        """Constants are materialized in args_codegen for this template."""
        return f"# {output_name} is created globally"

    def _generate_random_placement(self, tensor_size):
        """Generate random placement tuple (Replicate, Shard, or Partial)."""
        placements = []
        for _ in range(2):  # 2D mesh
            placement_type = random.randint(0, 2)
            if placement_type == 0:
                placements.append("Replicate()")
            elif placement_type == 1 and len(tensor_size) > 0:
                shard_dim = random.randint(0, len(tensor_size) - 1)
                placements.append(f"Shard({shard_dim})")
            else:
                placements.append("Partial()" if placement_type == 2 else "Replicate()")
        return f"({', '.join(placements)})"

    def args_codegen(self, arg_operations, constant_operations=None):
        """Generate args with randomized placements using dist_tensor API."""

        code_lines = []

        # DTensor setup (same as parent)
        code_lines.extend(
            [
                "world_size = 1024",
                "fake_store = FakeStore()",
                "torch.distributed.init_process_group(",
                '    "fake", store=fake_store, rank=0, world_size=world_size',
                ")",
                "",
                "mesh = torch.distributed.device_mesh.init_device_mesh(",
                '    "cuda", (2, 8), mesh_dim_names=("dim1", "dim2")',
                ")",
                "",
            ]
        )

        # Sentinel with random placement
        sentinel_placements = self._generate_random_placement((1,))
        code_lines.extend(
            [
                f"sentinel = dist_tensor.ones((1,), device_mesh=mesh, placements={sentinel_placements}, dtype=torch.float32, requires_grad=True)",
                "",
            ]
        )

        # Args with random placements using dist_tensor API
        if arg_operations:
            for i, (node_id, spec) in enumerate(arg_operations):
                if isinstance(spec, TensorSpec):
                    size_str = str(spec.size)
                    dtype_str = f"torch.{spec.dtype}".replace("torch.torch.", "torch.")
                    placements = self._generate_random_placement(spec.size)

                    if spec.dtype in [
                        torch.int32,
                        torch.int64,
                        torch.int8,
                        torch.int16,
                    ]:
                        code_lines.append(
                            f"arg_{i} = dist_tensor.ones({size_str}, device_mesh=mesh, placements={placements}, dtype={dtype_str}) * 5"
                        )
                    elif spec.dtype == torch.bool:
                        code_lines.append(
                            f"arg_{i} = dist_tensor.ones({size_str}, device_mesh=mesh, placements={placements}, dtype=torch.int8).bool()"
                        )
                    else:
                        code_lines.append(
                            f"arg_{i} = dist_tensor.randn({size_str}, device_mesh=mesh, placements={placements}, dtype={dtype_str}, requires_grad=True)"
                        )

        # Constants (if any) - use same dist_tensor approach
        if constant_operations:
            for node_id, var_name, spec in constant_operations:
                if isinstance(spec, TensorSpec):
                    size_str = str(spec.size)
                    dtype_str = f"torch.{spec.dtype}".replace("torch.torch.", "torch.")
                    placements = self._generate_random_placement(spec.size)
                    # Use dist_tensor.full with a simple fill value
                    code_lines.append(
                        f"{var_name} = dist_tensor.full({size_str}, 1.0, device_mesh=mesh, placements={placements}, dtype={dtype_str})"
                    )

        code_lines.append("")
        return code_lines
