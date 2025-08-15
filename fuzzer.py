import torch
from typing import Union, List, Optional
from dataclasses import dataclass
from tensor_fuzzer import (
    test_fuzzing_tensors, test_fuzz_spec, test_fuzz_op, test_fuzz_spec_with_fuzz_op,
    fuzz_tensor, fuzz_tensor_simple, fuzz_scalar, TensorSpec, ScalarSpec, Spec
)
from ops_fuzzer import fuzz_spec, fuzz_op
import random

# Global variable counter for generating unique variable names
_var_name_counters = {}


@dataclass
class Operation:
    """
    Represents a single operation in the fuzzed operation stack.
    
    Attributes:
        op_name: Name of the operation (e.g., 'torch.ops.aten.add', 'scalar_add', 'arg')
        input_specs: List of input specifications required by this operation
        output_spec: Output specification produced by this operation
        depth: Depth level of this operation in the generation tree
        reuse_target: Optional index of operation being reused (for reuse operations)
    """
    op_name: str
    input_specs: List[Spec]
    output_spec: Spec
    depth: int
    reuse_target: Optional[int] = None  # Index of operation being reused
    
    def __str__(self) -> str:
        """String representation for debugging."""
        if self.reuse_target is not None:
            return f"{self.op_name} -> {self.output_spec} (depth {self.depth}, reuses op {self.reuse_target})"
        return f"{self.op_name} -> {self.output_spec} (depth {self.depth})"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"Operation(op_name='{self.op_name}', input_specs={self.input_specs}, output_spec={self.output_spec}, depth={self.depth}, reuse_target={self.reuse_target})"

def generate_argument_creation_code(arg_names: list, arg_specs: list) -> list:
    """
    Generate Python code lines for creating arguments from specifications.
    
    Args:
        arg_names: List of argument variable names (e.g., ['arg_0', 'arg_1'])
        arg_specs: List of argument specifications (TensorSpec or ScalarSpec)
    
    Returns:
        List of code lines as strings
    """
    code_lines = []
    
    for i, (arg_name, arg_spec) in enumerate(zip(arg_names, arg_specs)):
        if isinstance(arg_spec, ScalarSpec):
            # Generate scalar creation code using fuzz_scalar
            dtype_str = f"torch.{arg_spec.dtype}".replace("torch.torch.", "torch.")
            
            code_lines.extend([
                f"# Create scalar argument {i}",
                f"from tensor_fuzzer import fuzz_scalar, ScalarSpec",
                f"scalar_spec_{i} = ScalarSpec(dtype={dtype_str})",
                f"{arg_name} = fuzz_scalar(scalar_spec_{i})"
            ])
            
        elif isinstance(arg_spec, TensorSpec):
            # Generate tensor creation code using fuzz_tensor_simple
            size_str = str(arg_spec.size)
            stride_str = str(arg_spec.stride)
            dtype_str = f"torch.{arg_spec.dtype}".replace("torch.torch.", "torch.")
            
            code_lines.extend([
                f"# Create tensor argument {i}",
                f"from tensor_fuzzer import fuzz_tensor_simple",
                f"{arg_name} = fuzz_tensor_simple({size_str}, {stride_str}, {dtype_str})"
            ])
    
    return code_lines


def generate_random_var_name(prefix: str = "var") -> str:
    """Generate a variable name with the given prefix and incremental number suffix."""
    global _var_name_counters
    
    if prefix not in _var_name_counters:
        _var_name_counters[prefix] = 0
    else:
        _var_name_counters[prefix] += 1
    
    return f"{prefix}_{_var_name_counters[prefix]}"


def fuzz_operation_stack(target_spec: Spec, max_depth: int = 3, seed: Optional[int] = None, 
                        enable_reuse: bool = False, reuse_probability: float = 0.25) -> List[Operation]:
    """
    Recursively generate a stack of operations that produces the target specification.
    
    The returned stack has the target-producing operation at index 0 (top of stack).
    
    Args:
        target_spec: The desired output specification (TensorSpec or ScalarSpec)
        max_depth: Maximum depth of operations. At depth 0, only leaf operations (constant, arg) are used.
        seed: Random seed for reproducible generation. If None, uses current random state.
        enable_reuse: Whether to enable reusing existing compatible operations (disabled by default)
        reuse_probability: Probability of reusing an existing operation when possible
    
    Returns:
        List of Operation dataclass instances with target-producing operation at index 0
    """
    
    # Set seed for reproducible generation
    if seed is not None:
        import random
        random.seed(seed)
        torch.manual_seed(seed)
    
    def _generate_recursive(spec: Spec, depth: int, stack_size: int = 0) -> List[Operation]:
        """
        Recursively generate operations for the given spec at the given depth.
        Returns list of operations with the spec-producing operation at index 0.
        """
        
        # Generate new operation normally
        op_name, input_specs = fuzz_op(spec, depth, stack_size)
        
        # Create operation entry using dataclass
        operation = Operation(
            op_name=op_name,
            input_specs=input_specs,
            output_spec=spec,
            depth=depth
        )
        
        # Start with empty dependency list
        all_dependencies = []
        
        # If this operation requires inputs, recursively generate them
        if input_specs:  # Non-leaf operations (not constant or arg)
            for input_spec in input_specs:
                # Generate operations for each input at depth-1
                input_ops = _generate_recursive(input_spec, max(0, depth - 1), stack_size + len(all_dependencies) + 1)
                # Add all input operations to dependencies
                all_dependencies.extend(input_ops)
        
        # Return list with the target operation at index 0, followed by all dependencies
        return [operation] + all_dependencies
    
    # Generate the operation stack
    operation_stack = _generate_recursive(target_spec, max_depth, 0)
    
    # Verify that the operation at index 0 produces the target spec
    if operation_stack and not specs_compatible(operation_stack[0].output_spec, target_spec):
        raise ValueError(f"Generated stack top operation produces {operation_stack[0].output_spec}, "
                        f"but target spec is {target_spec}")
    
    return operation_stack


def fuzz_and_execute(seed: Optional[int] = None, max_depth: Optional[int] = None, log_at_faluire=False):
    """
    Generate a fuzzed operation stack, convert it to Python code, and execute it.
    
    Args:
        seed: Random seed for reproducible generation. If None, uses a random seed.
        max_depth: Maximum depth for operation stack (1-10). If None, uses a random depth.
    
    Returns:
        tuple: (seed_used, success_status)
            - seed_used: The actual seed that was used for generation
            - success_status: True if execution succeeded, False if it failed
    
    This function:
    1. Generates a random target specification 
    2. Creates a stack of operations to produce that target
    3. Converts the stack into executable Python code
    4. Executes the generated Python code
    5. Validates the final result matches the target spec
    """
    
    import random
    
    # Generate seed if not provided
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    
    # Generate max_depth if not provided (range 1-10)
    if max_depth is None:
        random.seed(seed + 999)  # Use seed offset for consistent depth selection
        max_depth = random.randint(1, 20)
    else:
        # Clamp max_depth to valid range
        max_depth = max(1,  max_depth)
    
    print(f"Using seed: {seed}")
    print(f"Using max_depth: {max_depth}")
    
    # Set seed for reproducible generation
    random.seed(seed)
    torch.manual_seed(seed)
    operation_stack = None
    python_code = None
    result = None
    target_spec = None

    def log(success):
        import tempfile
        import os
        import time
        
        # Create a unique folder for this iteration
        timestamp = int(time.time() * 1000)  # milliseconds
        folder_name = f"fuzzing_seed_{seed}_{timestamp}_{'success' if success else 'failed'}"
        iteration_folder = os.path.join("/tmp", folder_name)
        os.makedirs(iteration_folder, exist_ok=True)
        
        if success:
            print(f"✅ SUCCESS - artifacts saved to: {iteration_folder}")
        else:
            print(f"❌ FAILED - artifacts saved to: {iteration_folder}")

        # Write summary file
        summary_path = os.path.join(iteration_folder, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Fuzzing Session Summary\n")
            f.write(f"======================\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Max depth: {max_depth}\n")
            f.write(f"Success: {success}\n")
            f.write(f"Target specification: {target_spec}\n")
            if operation_stack:
                f.write(f"Operations count: {len(operation_stack)}\n")

        if operation_stack:
            # Write operation stack to file in iteration folder
            stack_file_path = os.path.join(iteration_folder, "operation_stack.txt")
            with open(stack_file_path, 'w') as f:
                f.write(f"Target specification: {target_spec}\n")
                f.write(f"Generated {len(operation_stack)} operations in stack\n\n")
                f.write("Operation stack (in reverse order - dependencies first):\n")
                for i in range(len(operation_stack) - 1, -1, -1):
                    op = operation_stack[i]
                    f.write(f"  {i}: {op.op_name} -> {op.output_spec} (depth {op.depth})\n")
            
            # Generate visualization in the iteration folder
            from visualize_stack import visualize_operation_stack
            visualize_operation_stack(operation_stack, "Operation Stack", iteration_folder)
      
        if python_code:
            # Write Python code to file in iteration folder
            code_file_path = os.path.join(iteration_folder, "generated_code.py")
            with open(code_file_path, 'w') as f:
                f.write(python_code)
            
            print(f"📁 Code saved in : {code_file_path}")

        print(f"📁 All files saved to: {iteration_folder}")
    try:
        # Generate target specification and operation stack
        target_spec = fuzz_spec()
        operation_stack = fuzz_operation_stack(target_spec, max_depth=max_depth, seed=seed)

        # Convert operation stack to Python code
        python_code = convert_stack_to_python_code(operation_stack, target_spec, seed=seed)

        # Execute the generated Python code
        result = execute_python_code(python_code, target_spec)

        # # Validate the result matches target specification
        # validate_result_against_spec(result, target_spec)
        if not log_at_faluire:
            log(True)
        return seed, result
        
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        # from visualize_stack import visualize_operation_stack
        log(False)
        import traceback
        traceback.print_exc()
        return seed, False


def specs_compatible(spec1, spec2) -> bool:
    """
    Check if two specifications are compatible (one can be used where the other is expected).
    """
    if type(spec1) != type(spec2):
        return False
        
    if isinstance(spec1, ScalarSpec):
        # For scalars, require exact dtype match for simplicity
        return spec1.dtype == spec2.dtype
    elif isinstance(spec1, TensorSpec):
        # For tensors, shape and dtype should match exactly
        return (spec1.size == spec2.size and 
                spec1.dtype == spec2.dtype)
    
    return False


def validate_result_against_spec(result, target_spec) -> None:
    """
    Validate that the result matches the target specification.
    
    Args:
        result: The actual result from execution
        target_spec: Expected specification
        
    Raises:
        AssertionError: If validation fails
    """
    
    if isinstance(target_spec, ScalarSpec):
        # Check that result is a Python scalar of compatible type
        expected_python_types = {
            torch.float16: (float,),
            torch.float32: (float,), 
            torch.float64: (float,),
            torch.bfloat16: (float,),
            torch.int8: (int,),
            torch.int16: (int,),
            torch.int32: (int,),
            torch.int64: (int,),
            torch.bool: (bool,),
            torch.complex64: (complex,),
            torch.complex128: (complex,),
        }
        
        expected_types = expected_python_types.get(target_spec.dtype, (type(result),))
        
        if not isinstance(result, expected_types):
            raise AssertionError(f"Expected Python type {expected_types} for dtype {target_spec.dtype}, got {type(result)}")
                
    elif isinstance(target_spec, TensorSpec):
        # Check that result is a tensor with correct properties
        if not isinstance(result, torch.Tensor):
            raise AssertionError(f"Expected torch.Tensor, got {type(result)}")
        
        if result.shape != target_spec.size:
            raise AssertionError(f"Expected shape {target_spec.size}, got {result.shape}")
        
        if result.dtype != target_spec.dtype:
            raise AssertionError(f"Expected dtype {target_spec.dtype}, got {result.dtype}")
                
    else:
        raise ValueError(f"Unknown target spec type: {type(target_spec)}")


def convert_stack_to_python_code(operation_stack: List[Operation], target_spec, seed: Optional[int] = None) -> str:
    """
    Convert an operation stack to executable Python code using backward recursion.
    
    The stack represents operations in LIFO order:
    - operation_stack[0] is the TOP of the stack (final result we want)
    - operation_stack[-1] is the BOTTOM of the stack (foundational dependencies)
    
    Code generation uses backward recursion:
    1. Start from the top operation (index 0) - what we want to compute
    2. Recursively generate code for its dependencies
    3. Generate code in proper execution order (dependencies first)
    
    Args:
        operation_stack: List of Operation dataclass instances in stack order (top to bottom)
        target_spec: Expected output specification
        seed: Random seed for reproducible code generation. If None, uses current random state.
        
    Returns:
        String containing the complete Python code that executes the operations
    """
    
    # Set seed for reproducible code generation
    if seed is not None:
        import random
        random.seed(seed + 1000)  # Offset to avoid conflicts with operation_stack generation
        torch.manual_seed(seed + 1000)
    
    if not operation_stack:
        raise ValueError("Empty operation stack")
    
    # Track generated operations to avoid duplicates
    generated_operations = set()
    generated_code_lines = []
    operation_variables = {}  # Maps operation index to (var_name, spec)
    arg_operations = []  # List of (operation_index, spec) for arg operations
    
    def generate_operation_recursive(op_idx: int) -> tuple[str, int]:
        """
        Recursively generate code for operation at op_idx and its dependencies.
        Returns (variable_name, subtree_size) where subtree_size is the number of operations processed.
        """
        # If already generated, return the variable name and size 1
        if op_idx in generated_operations:
            return operation_variables[op_idx][0], 1
        
        operation = operation_stack[op_idx]
        op_name = operation.op_name
        input_specs = operation.input_specs
        output_spec = operation.output_spec
        
        # Track total subtree size starting with this operation
        total_subtree_size = 1
        
        # Generate input variables by recursively processing dependencies FIRST
        input_var_names = []
        if input_specs:
            # Calculate dependency indices based on stack generation pattern
            current_dep_idx = op_idx + 1
            
            for j, input_spec in enumerate(input_specs):
                # Find the dependency that produces this input
                dep_idx = current_dep_idx
                
                if dep_idx >= len(operation_stack):
                    raise ValueError(f"Operation {op_idx} ({op_name}) requires input {j} at index {dep_idx}, "
                                   f"but stack only has {len(operation_stack)} operations")
                
                # Verify the dependency produces the expected spec
                dep_operation = operation_stack[dep_idx]
                if not specs_compatible(dep_operation.output_spec, input_spec):
                    raise ValueError(f"Operation {op_idx} ({op_name}) requires input {input_spec} at position {j}, "
                                   f"but operation {dep_idx} produces {dep_operation.output_spec}")
                
                # Recursively generate this dependency
                dep_var_name, dependency_subtree_size = generate_operation_recursive(dep_idx)
                input_var_names.append(dep_var_name)
                
                # Update indices and total size
                current_dep_idx += dependency_subtree_size
                total_subtree_size += dependency_subtree_size
        
        # NOW add the comment for this operation (after dependencies are processed)
        generated_code_lines.append(f"    # Operation {op_idx}: {op_name} (stack position {op_idx})")
        
        # Generate output variable name
        output_var_name = f"tmp_{op_idx}"
        
        # Handle different operation types
        if op_name == "arg":
            # Track arg operations for later function signature generation
            arg_operations.append((op_idx, output_spec))
            arg_name = f"arg_{len(arg_operations) - 1}"
            operation_lines = [f"{output_var_name} = {arg_name}"]
        elif op_name.startswith("reuse_"):
            # Handle reuse operations - reference the variable from the reused operation
            if operation.reuse_target is not None:
                # Generate variable name for the target operation (even if not generated yet)
                target_var_name = f"tmp_{operation.reuse_target}"
                operation_lines = [f"{output_var_name} = {target_var_name}  # Reusing operation {operation.reuse_target}"]
            else:
                # Fallback if reuse target is invalid - this shouldn't happen
                operation_lines = [f"# ERROR: Invalid reuse operation {op_name}"]
        else:
            # Generate operation execution code
            operation_lines = generate_simple_operation_code(
                output_var_name, input_var_names, op_name, output_spec, {}
            )
        
        # Add proper indentation for function body
        generated_code_lines.extend(["    " + line for line in operation_lines])
        generated_code_lines.append("")
        
        # Track this operation as generated
        generated_operations.add(op_idx)
        operation_variables[op_idx] = (output_var_name, output_spec)
        
        return output_var_name, total_subtree_size
    
    # Start backward recursion from the top operation (index 0)
    final_var_name, _ = generate_operation_recursive(0)
    
    # Generate function signature based on discovered arg operations
    if arg_operations:
        arg_names = [f"arg_{i}" for i in range(len(arg_operations))]
        function_signature = f"def fuzzed_program({', '.join(arg_names)})"
    else:
        function_signature = "def fuzzed_program()"
    
    # Build the complete code
    code_lines = [
        "import torch",
        "from tensor_fuzzer import fuzz_scalar, fuzz_tensor_simple, ScalarSpec, TensorSpec",
        "",
        "# Generated fuzzed program code (backward recursion from stack top)",
        f"# Stack has {len(operation_stack)} operations",
        "",
        function_signature + ":",
    ]
    
    # Add the generated operation code
    code_lines.extend(generated_code_lines)
    
    # Add return statement
    code_lines.extend([
        f"    # Final result from top of stack (operation 0)",
        f"    return {final_var_name}",
        ""
    ])
    
    # Generate argument creation code with deterministic seeds
    if arg_operations:
        code_lines.append("# Create arguments for the fuzzed program")
        for i, (_, spec) in enumerate(arg_operations):
            arg_name = f"arg_{i}"
            # Use a deterministic seed based on the argument index and main seed
            arg_seed = (seed + 10000 + i) if seed is not None else None
            
            if isinstance(spec, ScalarSpec):
                dtype_str = f"torch.{spec.dtype}".replace("torch.torch.", "torch.")
                if arg_seed is not None:
                    code_lines.extend([
                        f"scalar_spec = ScalarSpec(dtype={dtype_str})",
                        f"{arg_name} = fuzz_scalar(scalar_spec, seed={arg_seed})"
                    ])
                else:
                    code_lines.extend([
                        f"scalar_spec = ScalarSpec(dtype={dtype_str})",
                        f"{arg_name} = fuzz_scalar(scalar_spec)"
                    ])
            elif isinstance(spec, TensorSpec):
                size_str = str(spec.size)
                stride_str = str(spec.stride)
                dtype_str = f"torch.{spec.dtype}".replace("torch.torch.", "torch.")
                if arg_seed is not None:
                    code_lines.append(f"{arg_name} = fuzz_tensor_simple({size_str}, {stride_str}, {dtype_str}, seed={arg_seed})")
                else:
                    code_lines.append(f"{arg_name} = fuzz_tensor_simple({size_str}, {stride_str}, {dtype_str})")
    
    # Generate the final execution with both normal and compiled versions
    if arg_operations:
        arg_names = [f"arg_{i}" for i in range(len(arg_operations))]
        if len(arg_names) == 1:
            args_tuple = f"({arg_names[0]},)"  # Single element tuple needs trailing comma
        else:
            args_tuple = f"({', '.join(arg_names)})"
    else:
        args_tuple = "()"
    
    code_lines.extend([
        "",
        "# Execute the fuzzed program both normally and with torch.compile",
        "import torch",
        "import tempfile",
        "import os",
        "import sys",
        "import contextlib",
        "from io import StringIO",
        "",
        f"# Create arguments",
        f"args = {args_tuple}",
        "",   
        "# Execute original version",
        "print('=== Executing Original Program ===')",
        "try:",
        "    result_original = fuzzed_program(*args)",
        "    print('✅ Original execution successful')",
        "except Exception as e:",
        "    print(f'❌ Original execution failed: {e}')",
        "    raise",
        "",
        "# Execute compiled version",
        "print('\\n=== Executing Compiled Program  fullgraph=False')",
        "try:",
        "    compiled_program = torch.compile(fuzzed_program, fullgraph=False)",
        "    result_compiled = compiled_program(*args)",
        "    print('✅ Compiled execution successful')",
        "    print(f'Compiled result type: {type(result_compiled)}')",
        "except Exception as e:",
        "    print(f'❌ Compiled execution failed: {e}')",
        "    # Exit with non-zero code to signal compile failure",
        "    import sys",
        "    sys.exit(1)",
        "",
        "# Execute compiled version 2",
        "print('\\n=== Executing Compiled Program  fullgraph=False dynamic=True')",
        "try:",
        "    compiled_program = torch.compile(fuzzed_program, fullgraph=False, dynamic=True)",
        "    result_compiled = compiled_program(*args)",
        "    print('✅ Compiled execution successful')",
        "    print(f'Compiled result type: {type(result_compiled)}')",
        "except Exception as e:",
        "    print(f'❌ Compiled execution failed: {e}')",
        "    # Exit with non-zero code to signal compile failure",
        "    import sys",
        "    sys.exit(1)",
        "",
        "# Execute compiled version 3",
        "print('\\n=== Executing Compiled Program  fullgraph=True dynamic=True')",
        "try:",
        "    with torch._dynamo.config.patch(capture_scalar_outputs=True):", 
        "       compiled_program = torch.compile(fuzzed_program, fullgraph=False, dynamic=True)",
        "       result_compiled = compiled_program(*args)",
        "       print('✅ Compiled execution successful')",
        "       print(f'Compiled result type: {type(result_compiled)}')",
        "except Exception as e:",
        "    print(f'❌ Compiled execution failed: {e}')",
        "    # Exit with non-zero code to signal compile failure",
        "    import sys",
        "    sys.exit(1)",
        "",
    ])
    
    return "\n".join(code_lines)


def generate_simple_operation_code(output_var: str, input_vars: list, op_name: str, output_spec, available_variables: Optional[dict] = None) -> list:
    """
    Generate code lines for executing a single operation (simplified version without arg_tracker).
    
    Args:
        output_var: Name of the output variable
        input_vars: List of input variable names
        op_name: Name of the operation
        output_spec: Output specification for the operation
        available_variables: Dict mapping variable names to their specs, for potential reuse
    """
    if op_name == "scalar_add":
        return [f"{output_var} = {input_vars[0]} + {input_vars[1]}"]
    
    elif op_name == "scalar_multiply":
        return [f"{output_var} = {input_vars[0]} * {input_vars[1]}"]
    
    elif op_name == "torch.ops.aten.item":
        return [f"{output_var} = {input_vars[0]}.item()"]
    
    elif op_name == "torch.ops.aten.add":
        return [f"{output_var} = torch.ops.aten.add({input_vars[0]}, {input_vars[1]})"]
    
    elif op_name == "torch.ops.aten.mul":
        return [f"{output_var} = torch.ops.aten.mul({input_vars[0]}, {input_vars[1]})"]
    
    elif op_name == "constant":
        # Create constant by calling fuzzing functions during codegen with deterministic seed
        # Use a deterministic seed based on the variable name to ensure reproducibility
        var_seed = hash(output_var) % (2**31)
        
        if isinstance(output_spec, ScalarSpec):
            # Call fuzz_scalar during codegen and embed the result
            actual_value = fuzz_scalar(output_spec, seed=var_seed)
            
            # Format the value for embedding in code
            if isinstance(actual_value, bool):
                value_str = str(actual_value)
            elif isinstance(actual_value, (int, float)):
                value_str = repr(actual_value)
            elif isinstance(actual_value, complex):
                value_str = f"complex({actual_value.real}, {actual_value.imag})"
            else:
                value_str = repr(actual_value)
                
            return [f"{output_var} = {value_str}"]
            
        elif isinstance(output_spec, TensorSpec):
            # Call fuzz_tensor_simple during codegen and embed the result
            actual_tensor = fuzz_tensor_simple(output_spec.size, output_spec.stride, output_spec.dtype, seed=var_seed)
            
            # Convert tensor to code representation
            size_str = str(output_spec.size)
            dtype_str = f"torch.{output_spec.dtype}".replace("torch.torch.", "torch.")
            
            # Handle empty tensors (with 0 elements)
            if actual_tensor.numel() == 0:
                # For empty tensors, use a default fill value based on dtype
                default_values = {
                    torch.float16: 0.0,
                    torch.float32: 0.0, 
                    torch.float64: 0.0,
                    torch.bfloat16: 0.0,
                    torch.int8: 0,
                    torch.int16: 0,
                    torch.int32: 0,
                    torch.int64: 0,
                    torch.bool: False,
                    torch.complex64: 0.0,
                    torch.complex128: 0.0,
                }
                fill_value = default_values.get(output_spec.dtype, 0)
                return [f"{output_var} = torch.full({size_str}, {fill_value}, dtype={dtype_str})"]
            else:
                # For non-empty tensors, use the first element as fill value
                fill_value = actual_tensor.flatten()[0].item()
                return [f"{output_var} = torch.full({size_str}, {fill_value}, dtype={dtype_str})"]
            
        else:
            return [f"# Unknown output spec type for constant: {type(output_spec)}"]
    
    else:
        return [f"# Unknown operation: {op_name}"]


def execute_python_code(python_code: str, target_spec) -> Union[torch.Tensor, float, int, bool, complex]:
    """
    Execute the generated Python code by writing it to a file and running it.
    Also execute it in-process to get the actual result for validation.
    
    Args:
        python_code: String containing Python code to execute
        target_spec: Expected output specification for validation
        
    Returns:
        The actual result from executing the generated code
    """
    import tempfile
    import subprocess
    import sys
    import os
    
    # Write the generated code to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='_generated.py', delete=False) as f:
        f.write(python_code)
        generated_file_path = f.name
    
    try:
        # Execute the generated file (for console output)
        subprocess.run(
            [sys.executable, generated_file_path],
            check=True
        )
        
        # If we get here, both original and compiled execution succeeded
        # Return a dummy result since the actual execution output is shown in console
        return True
        
    except subprocess.CalledProcessError as e:
        if e.returncode == 1:
            # This indicates torch.compile failed - raise a specific exception
            raise RuntimeError("torch.compile execution failed")
        else:
            # Other execution errors
            print(f"❌ Generated file execution failed with return code {e.returncode}")
            raise
    finally:
        # Clean up the temporary file
        try:
            os.unlink(generated_file_path)
        except:
            pass


def compare_results(result1, result2) -> bool:
    """
    Compare two results for equality, handling different types appropriately.
    
    Args:
        result1: First result
        result2: Second result
        
    Returns:
        True if results are considered equal, False otherwise
    """
    
    # Check if types match
    if type(result1) != type(result2):
        print(f"Type mismatch: {type(result1)} vs {type(result2)}")
        return False
    
    if isinstance(result1, torch.Tensor):
        # For tensors, check shape, dtype, and values
        if result1.shape != result2.shape:
            print(f"Shape mismatch: {result1.shape} vs {result2.shape}")
            return False
        
        if result1.dtype != result2.dtype:
            print(f"Dtype mismatch: {result1.dtype} vs {result2.dtype}")
            return False
        
        # Use torch.allclose for floating point comparison
        try:
            if result1.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.complex64, torch.complex128]:
                if not torch.allclose(result1, result2, rtol=1e-5, atol=1e-8, equal_nan=True):
                    print(f"Values differ (allclose failed)")
                    print(f"Max absolute difference: {torch.max(torch.abs(result1 - result2)).item()}")
                    return False
            else:
                # For integer/bool tensors, use exact equality
                if not torch.equal(result1, result2):
                    print(f"Values differ (exact comparison)")
                    return False
        except Exception as e:
            print(f"Error comparing tensor values: {e}")
            return False
            
        return True
        
    elif isinstance(result1, (int, float, bool, complex)):
        # For scalars, use appropriate comparison
        if isinstance(result1, float):
            # Use relative tolerance for floats
            import math
            if math.isnan(result1) and math.isnan(result2):
                return True
            elif math.isnan(result1) or math.isnan(result2):
                return False
            else:
                return abs(result1 - result2) <= 1e-8 * max(abs(result1), abs(result2), 1.0)
        elif isinstance(result1, complex):
            # Use tolerance for complex numbers
            return abs(result1 - result2) <= 1e-8 * max(abs(result1), abs(result2), 1.0)
        else:
            # Exact equality for int and bool
            return result1 == result2
    
    else:
        # For other types, use regular equality
        return result1 == result2


def generate_code_only(seed: Optional[int] = None):
    """
    Generate operation stack and Python code without executing it.
    
    Args:
        seed: Random seed for reproducible generation
        
    Returns:
        tuple: (operation_stack, python_code, target_spec)
    """
    # Set seed for reproducible generation
    if seed is not None:
        import random
        random.seed(seed)
        torch.manual_seed(seed)
    
    # Generate target specification and operation stack
    target_spec = fuzz_spec()
    operation_stack = fuzz_operation_stack(target_spec, max_depth=2, seed=seed)
    
    # Convert operation stack to Python code
    python_code = convert_stack_to_python_code(operation_stack, target_spec, seed=seed)
    
    return operation_stack, python_code, target_spec


def test_reproducible_generation():
    """
    Test that using the same seed produces identical code generation.
    """
    print("=== Testing Reproducible Generation ===")
    
    test_seed = 42
    
    # Generate code twice with the same seed
    print(f"🔄 First generation with seed {test_seed}:")
    try:
        stack1, code1, spec1 = generate_code_only(seed=test_seed)
        print(f"Target spec: {spec1}")
        print(f"Operations: {len(stack1)}")
        for i, op in enumerate(stack1):
            print(f"  {i}: {op.op_name}")
        print(f"Code length: {len(code1)} characters")
        print("\n📄 Generated Python Code:")
        print("=" * 50)
        print(code1)
        print("=" * 50)
    except Exception as e:
        print(f"❌ First generation failed: {e}")
        return
    
    print(f"\n🔄 Second generation with seed {test_seed}:")
    try:
        stack2, code2, spec2 = generate_code_only(seed=test_seed)
        print(f"Target spec: {spec2}")
        print(f"Operations: {len(stack2)}")
        for i, op in enumerate(stack2):
            print(f"  {i}: {op.op_name}")
        print(f"Code length: {len(code2)} characters")
        print("\n📄 Generated Python Code (should be identical):")
        print("=" * 50)
        print(code2)
        print("=" * 50)
    except Exception as e:
        print(f"❌ Second generation failed: {e}")
        return
    
    # Compare results
    print(f"\n🔍 Comparing generations:")
    
    # Compare target specs
    if spec1 == spec2:
        print("✅ Target specs are identical")
    else:
        print(f"❌ Target specs differ: {spec1} vs {spec2}")
    
    # Compare operation stacks
    if len(stack1) == len(stack2):
        print(f"✅ Operation stack lengths match ({len(stack1)})")
        
        all_ops_match = True
        for i, (op1, op2) in enumerate(zip(stack1, stack2)):
            if (op1.op_name == op2.op_name and 
                op1.input_specs == op2.input_specs and 
                op1.output_spec == op2.output_spec and 
                op1.depth == op2.depth):
                print(f"  ✅ Operation {i}: {op1.op_name} matches")
            else:
                print(f"  ❌ Operation {i} differs:")
                print(f"    First:  {op1}")
                print(f"    Second: {op2}")
                all_ops_match = False
        
        if all_ops_match:
            print("✅ All operations are identical")
    else:
        print(f"❌ Operation stack lengths differ: {len(stack1)} vs {len(stack2)}")
    
    # Compare generated code
    if code1 == code2:
        print("✅ Generated code is identical")
        print("🎉 Reproducible generation test PASSED!")
    else:
        print("❌ Generated code differs")
        print("First few lines of difference:")
        
        lines1 = code1.split('\n')
        lines2 = code2.split('\n')
        
        max_lines = max(len(lines1), len(lines2))
        differences_shown = 0
        
        for i in range(max_lines):
            line1 = lines1[i] if i < len(lines1) else "<missing>"
            line2 = lines2[i] if i < len(lines2) else "<missing>"
            
            if line1 != line2:
                print(f"  Line {i+1}:")
                print(f"    First:  '{line1}'")
                print(f"    Second: '{line2}'")
                differences_shown += 1
                
                if differences_shown >= 5:
                    print("    ... (showing first 5 differences)")
                    break
        
        print("❌ Reproducible generation test FAILED!")
    
    # Test with different seed to ensure it produces different results
    print(f"\n🔄 Third generation with different seed {test_seed + 1}:")
    try:
        stack3, code3, spec3 = generate_code_only(seed=test_seed + 1)
        print(f"Target spec: {spec3}")
        print(f"Operations: {len(stack3)}")
        print(f"Code length: {len(code3)} characters")
        
        if code1 != code3:
            print("✅ Different seed produces different code (as expected)")
        else:
            print("⚠️  Different seed produced identical code (unusual but possible)")
            
    except Exception as e:
        print(f"❌ Third generation failed: {e}")


def fuzz_and_test():
    """
    Test the new fuzz_and_execute function with seed and max_depth arguments.
    """
    print("=== Testing fuzz_and_execute with arguments ===")
   
    for i in range(100):
        print(f"------------------ TEST itteration {i} ---------------")
        seed, success = fuzz_and_execute()
        if not success:
            return 



def quick_visualize_test(seed: Optional[int] = None, max_depth: int = 3, title: Optional[str] = None):
    """
    Quick helper to generate and visualize an operation stack.
    
    Args:
        seed: Random seed for reproducible generation
        max_depth: Maximum operation depth  
        title: Optional title for the visualization
    """
    try:
        from visualize_stack import visualize_operation_stack
    except ImportError:
        print("⚠️  visualize_stack.py not found in current directory")
        return
    
    print(f"🎲 Generating operation stack (seed={seed}, max_depth={max_depth})...")
    
    # Generate random target spec and operation stack
    target_spec = fuzz_spec()
    operation_stack = fuzz_operation_stack(target_spec, max_depth=max_depth, seed=seed)
    
    print(f"🎯 Target: {target_spec}")
    print(f"📚 Generated {len(operation_stack)} operations")
    
    # Create descriptive title
    if title is None:
        title = f"Stack (seed={seed}, depth={max_depth}, {len(operation_stack)} ops)"
    
    # Visualize
    visualize_operation_stack(operation_stack, title)
    
    return operation_stack


if __name__ == "__main__":
    test_fuzzing_tensors()
    
    # Test the new function interface
    fuzz_and_test()
    
    # print("\n" + "="*60)
    # Test reproducible generation
    # test_reproducible_generation()
    
    # Quick visualization demo
    # print("\n" + "="*60)
    # print("🎨 VISUALIZATION DEMO")
    # print("="*60)
    # quick_visualize_test(seed=42, max_depth=2, title="Demo Stack")
    