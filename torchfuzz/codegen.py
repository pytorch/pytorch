"""
Code generation utilities for PyTorch fuzzer.
This module handles generating Python code from fuzzing operations.
"""

import os


class CodeGenerator:
    """Generates Python code from tensor operations and their dependencies."""

    def __init__(self, use_dtensor=False, mesh_dims=(8, 8, 4), world_size=256, placements=None, test_compile=False):
        self.use_dtensor = use_dtensor
        self.mesh_dims = mesh_dims
        self.world_size = world_size
        self.placements = placements or ("Shard(0)", "Shard(1)", "Shard(1)")
        self.test_compile = test_compile  # Whether to test compilation for DTensor

    def tensor_repr(self, tensor):
        """Generate tensor creation code representation."""
        if tensor.dtype == "bool":
            # torch.rand does not support bool, use torch.randint for boolean tensors
            return f"torch.randint(0, 2, {tuple(tensor.size)}, dtype=torch.bool, device='{tensor.device}')"
        elif tensor.dtype in ["int8", "int16", "int32", "int64", "uint8"]:
            # Integer tensors don't support requires_grad, use torch.randint instead
            # Use a reasonable range for integer tensors (0 to 1000 for most cases)
            max_val = 1000 if tensor.dtype == "int64" else 100
            return f"torch.randint(0, {max_val}, {list(tensor.size)}, dtype=torch.{tensor.dtype}, device='{tensor.device}')"
        return f"torch.rand({list(tensor.size)}, dtype=torch.{tensor.dtype}, device='{tensor.device}', requires_grad=True)"

    def generate_code(self, target_tensor, all_nodes, output_path):
        """
        Generate complete Python code from nodes and write to file.

        Args:
            target_tensor: The final output tensor
            all_nodes: List of (tensor, op, inputs) tuples in topological order
            output_path: Path to write the generated code
        """
        # Generate Python code
        code_lines = []

        if self.use_dtensor:
            # DTensor imports and setup
            code_lines.extend([
                "import torch",
                "import sys",
                "from torch.distributed.tensor.placement_types import Replicate, Shard",
                "from torch.testing._internal.distributed.fake_pg import FakeStore",
                "from torch.distributed.tensor import DTensor",
                "torch._dynamo.config.capture_scalar_outputs = True",
                "torch._dynamo.config.capture_dynamic_output_shape_ops = True",
                "torch._inductor.config.emulate_precision_casts = True",
                ""
            ])
        else:
            code_lines.extend([
                "import torch",
                "import sys",
                "torch._dynamo.config.capture_scalar_outputs = True",
                "torch._dynamo.config.capture_dynamic_output_shape_ops = True",
                "torch._inductor.config.emulate_precision_casts = True",
                ""
            ])

        tensor_names = {}

        # Identify leaf tensors (those with op is None and not produced by any op)
        leaf_nodes = [node for node in all_nodes if node[1] is None]
        leaf_tensors = [node[0] for node in leaf_nodes]

        # Assign names to tensors
        for idx, (tensor, op, inputs) in enumerate(all_nodes):
            name = f"t{idx}"
            tensor_names[id(tensor)] = name

        # Assign argument names for leaf tensors
        leaf_tensor_names = {}
        for i, tensor in enumerate(leaf_tensors):
            leaf_tensor_names[id(tensor)] = f"arg{i}"

        # Emit code for each node inside foo(), with leaf tensors as arguments
        arg_list = ", ".join([leaf_tensor_names[id(t)] for t in leaf_tensors])
        code_lines.append(f"def foo({arg_list}):")
        for idx, (tensor, op, inputs) in enumerate(all_nodes):
            name = tensor_names[id(tensor)]
            # Add tensor descriptor comment
            desc = f"# size={tensor.size}, stride={tensor.stride}, dtype={tensor.dtype}, device={tensor.device}"
            if op is None:
                # Leaf tensor: use argument
                code_lines.append(f"    {name} = {leaf_tensor_names[id(tensor)]} {desc}")
            else:
                input_names = [tensor_names[id(t)] for t in inputs]
                op_code = op.codegen(name, input_names, tensor)
                # Handle multi-line code generation properly
                if '\n' in op_code:
                    # Split into multiple lines and indent each one
                    op_lines = op_code.split('\n')
                    for i, line in enumerate(op_lines):
                        if i == 0:
                            # First line gets the comment
                            code_lines.append(f"    {line} {desc}")
                        else:
                            # Subsequent lines just get indented
                            code_lines.append(f"    {line}")
                else:
                    code_lines.append(f"    {op_code} {desc}")
        # The output is the original target tensor
        output_name = tensor_names[id(target_tensor)]
        code_lines.append(f"    output = {output_name}  # output tensor")
        code_lines.append(f"    return output")
        code_lines.append("")

        if self.use_dtensor:
            # DTensor setup
            code_lines.extend([
                "# FakeStore will mock collective results so that it can be ran on a single rank",
                "# =============================================================================",
                "",
                f"world_size = {self.world_size}",
                "fake_store = FakeStore()",
                "torch.distributed.init_process_group(",
                "    \"fake\", store=fake_store, rank=0, world_size=world_size",
                ")",
                "mesh = torch.distributed.device_mesh.init_device_mesh(",
                "    \"cuda\",",
                f"    {self.mesh_dims},",
                "    mesh_dim_names=(",
                "        " + ", ".join([f"\"dim{i+1}\"" for i in range(len(self.mesh_dims))]) + ",",
                "    ),",
                ")",
                f"placements = ({', '.join(self.placements)})",
                ""
            ])

        # Emit code to create the leaf tensors before calling foo
        for i, tensor in enumerate(leaf_tensors):
            desc = f"# size={tensor.size}, stride={tensor.stride}, dtype={tensor.dtype}, device={tensor.device}"
            code_lines.append(f"arg{i} = {self.tensor_repr(tensor)} {desc}")
            if self.use_dtensor:
                code_lines.append(f"arg{i} = DTensor.from_local(arg{i}, mesh, placements)")

        # Harness: run foo in eager and with torch.compile, then do a realistic backward (sum().backward())
        code_lines.append("if __name__ == '__main__':")
        code_lines.append(f"    out_eager = foo({', '.join([f'arg{i}' for i in range(len(leaf_tensors))])})")
        code_lines.append("    out_eager.sum().backward()")
        code_lines.append("    print('Eager Success! ✅')")

        # Zero out grads before running compiled version
        for i in range(len(leaf_tensors)):
            code_lines.append(f"    arg{i}.grad = None")

        # For DTensor, optionally test compilation (often fails, which is valuable for fuzzing)
        if self.use_dtensor:
            if self.test_compile:
                code_lines.extend([
                    "    # DTensor compilation often fails - this is expected and valuable for fuzzing",
                    "    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)",
                    f"    out_compiled = compiled_foo({', '.join([f'arg{i}' for i in range(len(leaf_tensors))])})",
                    "    out_compiled.sum().backward()",
                    "    print('Compile Success! ✅')",
                    "    ",
                    "    # Compare outputs (forward)",
                    "    out_eager_sum = out_eager.sum()",
                    "    out_compiled_sum = out_compiled.sum()",
                    "    diff = (out_eager_sum - out_compiled_sum).abs().item()",
                    "    rel_diff = diff / (out_eager_sum.abs().item() + 1e-12) * 100",
                    "    print(f'Relative diff (sum): {rel_diff:.6f}%')",
                    "    if rel_diff > 5:",
                    "        print(f'❌ Forward output sums differ significantly (relative)!')",
                    "        print('out_eager_sum:', out_eager_sum.item())",
                    "        print('out_compiled_sum:', out_compiled_sum.item())",
                    "        print('Absolute diff:', diff)",
                    "        print('Relative diff (%):', rel_diff)",
                    "        sys.exit(1)",
                    "    torch.distributed.destroy_process_group()"
                ])
            else:
                code_lines.extend([
                    "    # Skipping compilation test for DTensor (use --test-compile to enable)",
                    "    # This generates valid eager programs that may fail compilation - perfect for fuzzing!",
                    "    torch.distributed.destroy_process_group()"
                ])
        else:

            code_lines.extend([
                "    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)",
                f"    out_compiled = compiled_foo({', '.join([f'arg{i}' for i in range(len(leaf_tensors))])})",
                "    out_compiled.sum().backward()",
                "    print('Compile Success! ✅')"
            ])

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write to disk
        with open(output_path, "w") as f:
            f.write("\n".join(code_lines))

        return os.path.abspath(output_path)
