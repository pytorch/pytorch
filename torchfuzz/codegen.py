"""
Code generation utilities for PyTorch fuzzer.
This module handles generating Python code from fuzzing operations.
"""

import os


class CodeGenerator:
    """Generates Python code from tensor operations and their dependencies."""

    def __init__(self):
        pass

    def tensor_repr(self, tensor):
        """Generate tensor creation code representation."""
        return f"torch.empty({list(tensor.size)}, dtype=torch.{tensor.dtype}, device='{tensor.device}')"

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
        code_lines.append("import torch")
        code_lines.append("import sys")
        code_lines.append("")
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
                code_lines.append(f"    {op.codegen(name, input_names, tensor)} {desc}")
        # The output is the original target tensor
        output_name = tensor_names[id(target_tensor)]
        code_lines.append(f"    output = {output_name}  # output tensor")
        code_lines.append(f"    return output")
        code_lines.append("")

        # Emit code to create the leaf tensors before calling foo
        for i, tensor in enumerate(leaf_tensors):
            desc = f"# size={tensor.size}, stride={tensor.stride}, dtype={tensor.dtype}, device={tensor.device}"
            code_lines.append(f"arg{i} = {self.tensor_repr(tensor)} {desc}")

        # Harness: run foo in eager and with torch.compile, exit(1) if either fails
        code_lines.append("if __name__ == '__main__':")
        code_lines.append(f"    out_eager = foo({', '.join([f'arg{i}' for i in range(len(leaf_tensors))])})")
        code_lines.append("    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)")
        code_lines.append(f"    out_compiled = compiled_foo({', '.join([f'arg{i}' for i in range(len(leaf_tensors))])})")
        code_lines.append("    print('Success!')")

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write to disk
        with open(output_path, "w") as f:
            f.write("\n".join(code_lines))

        return os.path.abspath(output_path)
