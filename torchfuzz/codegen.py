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
        # Use actual size for batch_norm buffer tensors
        if hasattr(tensor, '_actual_size'):
            tensor_size = tensor._actual_size
        else:
            tensor_size = tensor.size
            
        if tensor.dtype == "bool":
            # torch.rand does not support bool, use torch.randint for boolean tensors
            return f"torch.randint(0, 2, {tuple(tensor_size)}, dtype=torch.bool, device='{tensor.device}')"
        elif tensor.dtype in ["int8", "int16", "int32", "int64", "uint8"]:
            # Integer tensors don't support requires_grad, use torch.randint instead
            # Use a reasonable range for integer tensors (0 to 1000 for most cases)
            max_val = 1000 if tensor.dtype == "int64" else 100
            return f"torch.randint(0, {max_val}, {list(tensor_size)}, dtype=torch.{tensor.dtype}, device='{tensor.device}')"
        
        # Handle requires_grad attribute
        if hasattr(tensor, 'requires_grad') and tensor.requires_grad is not None:
            requires_grad = tensor.requires_grad
        else:
            requires_grad = True  # Default behavior
            
        return f"torch.rand({list(tensor_size)}, dtype=torch.{tensor.dtype}, device='{tensor.device}', requires_grad={requires_grad})"

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
        code_lines.append("torch._dynamo.config.capture_scalar_outputs = True")
        code_lines.append("torch._dynamo.config.capture_dynamic_output_shape_ops = True")
        code_lines.append("torch._inductor.config.emulate_precision_casts = True")
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

        # Emit code to create the leaf tensors before calling foo
        for i, tensor in enumerate(leaf_tensors):
            desc = f"# size={tensor.size}, stride={tensor.stride}, dtype={tensor.dtype}, device={tensor.device}"
            code_lines.append(f"arg{i} = {self.tensor_repr(tensor)} {desc}")

        # Harness: run foo in eager and with torch.compile, then do a realistic backward (sum().backward())
        code_lines.append("if __name__ == '__main__':")
        code_lines.append(f"    out_eager = foo({', '.join([f'arg{i}' for i in range(len(leaf_tensors))])})")
        code_lines.append("    out_eager.sum().backward()")
        code_lines.append("    print('Eager Success! ✅')")
        code_lines.append("    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)")
        code_lines.append(f"    out_compiled = compiled_foo({', '.join([f'arg{i}' for i in range(len(leaf_tensors))])})")
        code_lines.append("    out_compiled.sum().backward()")
        code_lines.append("    print('Compile Success! ✅')")
        # code_lines.append("    # Compare outputs (forward)")
        # code_lines.append("    out_eager_sum = out_eager.sum()")
        # code_lines.append("    out_compiled_sum = out_compiled.sum()")
        # code_lines.append("    diff = (out_eager_sum - out_compiled_sum).abs().item()")
        # code_lines.append("    rel_diff = diff / (out_eager_sum.abs().item() + 1e-12) * 100")
        # code_lines.append("    print(f'Relative diff (sum): {rel_diff:.6f}%')")
        # code_lines.append("    if rel_diff > 5:")  # 5% threshold, adjust as needed
        # code_lines.append("        print(f'❌ Forward output sums differ significantly (relative)!')")
        # code_lines.append("        print('out_eager_sum:', out_eager_sum.item())")
        # code_lines.append("        print('out_compiled_sum:', out_compiled_sum.item())")
        # code_lines.append("        print('Absolute diff:', diff)")
        # code_lines.append("        print('Relative diff (%):', rel_diff)")
        # code_lines.append("        sys.exit(1)")


        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write to disk
        with open(output_path, "w") as f:
            f.write("\n".join(code_lines))

        return os.path.abspath(output_path)
