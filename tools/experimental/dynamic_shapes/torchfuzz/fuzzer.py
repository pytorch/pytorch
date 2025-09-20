from __future__ import annotations

import logging
import random
from typing import Any, Optional, Union

from codegen import convert_stack_to_python_code, execute_python_code, Operation
from ops_fuzzer import fuzz_op, fuzz_spec
from tensor_fuzzer import ScalarSpec, Spec, TensorSpec
from visualize_stack import visualize_operation_stack

import torch


# Global variable counter for generating unique variable names
_var_name_counters: dict[str, int] = {}


def generate_argument_creation_code(
    arg_names: list[str], arg_specs: list[Spec]
) -> list[str]:
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

            code_lines.extend(
                [
                    f"# Create scalar argument {i}",
                    "from tensor_fuzzer import fuzz_scalar, ScalarSpec",
                    f"scalar_spec_{i} = ScalarSpec(dtype={dtype_str})",
                    f"{arg_name} = fuzz_scalar(scalar_spec_{i})",
                ]
            )

        elif isinstance(arg_spec, TensorSpec):
            # Generate tensor creation code using fuzz_tensor_simple
            size_str = str(arg_spec.size)
            stride_str = str(arg_spec.stride)
            dtype_str = f"torch.{arg_spec.dtype}".replace("torch.torch.", "torch.")

            code_lines.extend(
                [
                    f"# Create tensor argument {i}",
                    "from tensor_fuzzer import fuzz_tensor_simple",
                    f"{arg_name} = fuzz_tensor_simple({size_str}, {stride_str}, {dtype_str})",
                ]
            )

    return code_lines


def generate_random_var_name(prefix: str = "var") -> str:
    """Generate a variable name with the given prefix and incremental number suffix."""
    global _var_name_counters

    if prefix not in _var_name_counters:
        _var_name_counters[prefix] = 0
    else:
        _var_name_counters[prefix] += 1

    return f"{prefix}_{_var_name_counters[prefix]}"


def fuzz_operation_stack(
    target_spec: Spec,
    max_depth: int = 3,
    seed: Optional[int] = None,
) -> list[Operation]:
    """
    Recursively generate a stack of operations that produces the target specification.

    The returned stack has the target-producing operation at index 0 (top of stack).

    Args:
        target_spec: The desired output specification (TensorSpec or ScalarSpec)
        max_depth: Maximum depth of operations. At depth 0, only leaf operations (constant, arg) are used.
        seed: Random seed for reproducible generation. If None, uses current random state.

    Returns:
        List of Operation dataclass instances with target-producing operation at index 0
    """

    # Set seed for reproducible generation
    if seed is not None:
        import random

        random.seed(seed)
        torch.manual_seed(seed)

    def _generate_recursive(
        spec: Spec, depth: int, stack_size: int = 0
    ) -> list[Operation]:
        """
        Recursively generate operations for the given spec at the given depth.
        Returns list of operations with the spec-producing operation at index 0.
        """

        # Generate new operation normally
        op_name, input_specs = fuzz_op(spec, depth, stack_size)

        # Create operation entry using dataclass
        operation = Operation(
            op_name=op_name, input_specs=input_specs, output_spec=spec, depth=depth
        )

        # Start with empty dependency list
        all_dependencies: list[Operation] = []

        # If this operation requires input_var_name_counterss, recursively generate them
        if input_specs:  # Non-leaf operations (not constant or arg)
            for input_spec in input_specs:
                # Generate operations for each input at depth-1
                input_ops = _generate_recursive(
                    input_spec,
                    max(0, depth - 1),
                    stack_size + len(all_dependencies) + 1,
                )
                # Add all input operations to dependencies
                all_dependencies.extend(input_ops)

        # Return list with the target operation at index 0, followed by all dependencies
        return [operation] + all_dependencies

    # Generate the operation stack
    operation_stack = _generate_recursive(target_spec, max_depth, 0)

    # Verify that the operation at index 0 produces the target spec
    if operation_stack and not specs_compatible(
        operation_stack[0].output_spec, target_spec
    ):
        raise ValueError(
            f"Generated stack top operation produces {operation_stack[0].output_spec}, "
            f"but target spec is {target_spec}"
        )

    return operation_stack


def fuzz_and_execute(
    seed: Optional[int] = None,
    max_depth: Optional[int] = None,
    log_at_faluire: bool = False,
) -> tuple[int, Union[bool, Any], Optional[str]]:
    """
    Generate a fuzzed operation stack, convert it to Python code, and execute it.

    Args:
        seed: Random seed for reproducible generation. If None, uses a random seed.
        max_depth: Maximum depth for operation stack (1-10). If None, uses a random depth.

    Returns:
        tuple: (seed_used, success_status, error_message)
            - seed_used: The actual seed that was used for generation
            - success_status: True if execution succeeded, False if it failed
            - error_message: Error message if failed, None if succeeded

    This function:
    1. Generates a random target specification
    2. Creates a stack of operations to produce that target
    3. Converts the stack into executable Python code
    4. Executes the generated Python code
    5. Validates the final result matches the target spec
    """

    # Generate seed if not provided
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    # Generate max_depth if not provided (range 1-10)
    if max_depth is None:
        random.seed(seed + 999)  # Use seed offset for consistent depth selection
        max_depth = random.randint(1, 20)
    else:
        # Clamp max_depth to valid range
        max_depth = max(1, max_depth)

    print(f"Using seed: {seed}")
    print(f"Using max_depth: {max_depth}")

    # Set seed for reproducible generation
    random.seed(seed)
    torch.manual_seed(seed)
    operation_stack = None
    python_code = None
    result = None
    target_spec = None

    def log(success: bool) -> None:
        import os
        import time

        # Create a unique folder for this iteration
        timestamp = int(time.time() * 1000)  # milliseconds
        folder_name = (
            f"fuzzing_seed_{seed}_{timestamp}_{'success' if success else 'failed'}"
        )
        iteration_folder = os.path.join("/tmp", folder_name)
        os.makedirs(iteration_folder, exist_ok=True)

        if success:
            print(f"✅ SUCCESS - artifacts saved to: {iteration_folder}")
        else:
            print(f"❌ FAILED - artifacts saved to: {iteration_folder}")

        # Write summary file
        summary_path = os.path.join(iteration_folder, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("Fuzzing Session Summary\n")
            f.write("======================\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Max depth: {max_depth}\n")
            f.write(f"Success: {success}\n")
            f.write(f"Target specification: {target_spec}\n")
            if operation_stack:
                f.write(f"Operations count: {len(operation_stack)}\n")

        if operation_stack:
            # Write operation stack to file in iteration folder
            stack_file_path = os.path.join(iteration_folder, "operation_stack.txt")
            with open(stack_file_path, "w") as f:
                f.write(f"Target specification: {target_spec}\n")
                f.write(f"Generated {len(operation_stack)} operations in stack\n\n")
                f.write("Operation stack (in reverse order - dependencies first):\n")
                for i in range(len(operation_stack) - 1, -1, -1):
                    op = operation_stack[i]
                    f.write(
                        f"  {i}: {op.op_name} -> {op.output_spec} (depth {op.depth})\n"
                    )

            # Generate visualization in the iteration folder

            visualize_operation_stack(
                operation_stack, "Operation Stack", iteration_folder
            )

        if python_code:
            # Write Python code to file in iteration folder
            code_file_path = os.path.join(iteration_folder, "generated_code.py")
            with open(code_file_path, "w") as f:
                f.write(python_code)

            print(f"📁 Code saved in : {code_file_path}")

        print(f"📁 All files saved to: {iteration_folder}")

    import time

    try:
        logger = logging.getLogger(__name__)

        # Generate target specification first
        logger.debug("⏱️  Step 1: Generating target spec...")
        start_time = time.time()
        target_spec = fuzz_spec()
        logger.debug(
            "   Completed in %.3fs - %s", time.time() - start_time, target_spec
        )

        logger.debug("⏱️  Step 2: Generating operation stack...")
        start_time = time.time()
        operation_stack = fuzz_operation_stack(
            target_spec, max_depth=max_depth, seed=seed
        )
        logger.debug(
            "   Completed in %.3fs - %d operations",
            time.time() - start_time,
            len(operation_stack),
        )

        logger.debug("⏱️  Step 3: Converting to Python code...")
        start_time = time.time()
        python_code = convert_stack_to_python_code(
            operation_stack, target_spec, seed=seed
        )
        logger.debug(
            "   Completed in %.3fs - %d chars",
            time.time() - start_time,
            len(python_code),
        )

        logger.debug("⏱️  Step 4: Executing Python code...")
        start_time = time.time()
        # Enable temporary file preservation in debug mode for easier debugging
        preserve_temp = logger.isEnabledFor(logging.DEBUG)
        # Use a 60-second timeout for execution
        result = execute_python_code(
            python_code, target_spec, preserve_temp_file=preserve_temp, timeout=300
        )
        logger.debug("   Completed in %.3fs", time.time() - start_time)

        # # Validate the result matches target specification
        # validate_result_against_spec(result, target_spec)
        if not log_at_faluire:
            log(True)
        return seed, result, None

    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        # from visualize_stack import visualize_operation_stack
        log(False)
        import traceback

        traceback.print_exc()
        error_message = str(e)
        return seed, False, error_message


def specs_compatible(spec1: Spec, spec2: Spec) -> bool:
    """
    Check if two specifications are compatible (one can be used where the other is expected).
    """
    if type(spec1) != type(spec2):
        return False

    if isinstance(spec1, ScalarSpec):
        # For scalars, require exact dtype match for simplicity
        return spec1.dtype == spec2.dtype
    elif isinstance(spec1, TensorSpec):
        assert isinstance(spec2, TensorSpec)
        # For tensors, shape and dtype should match exactly
        return spec1.size == spec2.size and spec1.dtype == spec2.dtype

    return False


def validate_result_against_spec(result: Any, target_spec: Spec) -> None:
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
            raise AssertionError(
                f"Expected Python type {expected_types} for dtype {target_spec.dtype}, got {type(result)}"
            )

    elif isinstance(target_spec, TensorSpec):
        # Check that result is a tensor with correct properties
        if not isinstance(result, torch.Tensor):
            raise AssertionError(f"Expected torch.Tensor, got {type(result)}")

        if result.shape != target_spec.size:
            raise AssertionError(
                f"Expected shape {target_spec.size}, got {result.shape}"
            )

        if result.dtype != target_spec.dtype:
            raise AssertionError(
                f"Expected dtype {target_spec.dtype}, got {result.dtype}"
            )

    else:
        raise ValueError(f"Unknown target spec type: {type(target_spec)}")


def generate_code_only(seed: Optional[int] = None) -> tuple[list[Operation], str, Spec]:
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


def test_reproducible_generation() -> None:
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
    print("\n🔍 Comparing generations:")

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
            if (
                op1.op_name == op2.op_name
                and op1.input_specs == op2.input_specs
                and op1.output_spec == op2.output_spec
                and op1.depth == op2.depth
            ):
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

        lines1 = code1.split("\n")
        lines2 = code2.split("\n")

        max_lines = max(len(lines1), len(lines2))
        differences_shown = 0

        for i in range(max_lines):
            line1 = lines1[i] if i < len(lines1) else "<missing>"
            line2 = lines2[i] if i < len(lines2) else "<missing>"

            if line1 != line2:
                print(f"  Line {i + 1}:")
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


def fuzz_and_test(seed: Optional[int] = None, max_depth: Optional[int] = None) -> None:
    """
    Test the new fuzz_and_execute function with seed and max_depth arguments.

    Args:
        seed: Starting seed for the test loop. If provided, each iteration uses seed + i
        max_depth: Maximum depth for operation stack to use in all iterations
    """
    known_issues = {
        "RuntimeError: self.stride(-1) must be 1 to view ComplexDouble as": "https://github.com/pytorch/pytorch/issues/162561",
        "BooleanAtom not allowed in this context": "https://github.com/pytorch/pytorch/issues/160726",
    }

    def known_issue(error_message: str) -> bool:
        return any(issue in error_message for issue in known_issues.keys())

    print("=== Testing fuzz_and_execute with arguments ===")
    if seed is not None:
        print(f"Using starting seed: {seed}")
    if max_depth is not None:
        print(f"Using max_depth: {max_depth}")

    for i in range(1000):
        print(f"------------------ TEST iteration {i} ---------------")

        # Use starting seed + iteration number for reproducible but varied results
        iteration_seed = seed + i if seed is not None else None

        iteration_seed, success, error_message = fuzz_and_execute(
            seed=iteration_seed, max_depth=max_depth
        )
        if not success:
            assert error_message is not None
            if known_issue(error_message):
                print("Known issue skipped")
                continue

            print(f"Test failed with error: {error_message}")
            return


def tests() -> None:
    """
    Run all tests.
    """
    print("=== Running all tests ===")
    test_reproducible_generation()


if __name__ == "__main__":
    import argparse

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="PyTorch Fuzzer - Generate and test random PyTorch operations"
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for reproducible generation"
    )
    parser.add_argument(
        "--max-depth", type=int, help="Maximum depth for operation stack (1-20)"
    )
    parser.add_argument("--test", action="store_true", help="Run the fuzzing test loop")
    parser.add_argument(
        "--single", action="store_true", help="Run a single fuzz_and_execute"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

    if args.single:
        # Run a single execution with optional seed and max_depth
        print("Running single fuzz_and_execute...")
        seed, success, error_message = fuzz_and_execute(
            seed=args.seed, max_depth=args.max_depth
        )
        print(f"Result: seed={seed}, success={success}")
        if not success:
            print(f"Error: {error_message}")
    else:
        # Default behavior - run the test loop (--test is now the default)
        fuzz_and_test(seed=args.seed, max_depth=args.max_depth)
