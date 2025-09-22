# mypy: ignore-errors
import logging
import os
import random
import sys
from typing import Any, Optional, Union


# Add parent directory to path so we can import torchfuzz as a module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from torchfuzz.codegen import convert_graph_to_python_code, execute_python_code
from torchfuzz.ops_fuzzer import fuzz_operation_graph, fuzz_spec
from torchfuzz.visualize_graph import visualize_operation_graph


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

    # Generate max_depth if not provided (range 3-12)
    if max_depth is None:
        random.seed(seed + 999)  # Use seed offset for consistent depth selection
        max_depth = random.randint(3, 12)
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
            visualize_operation_graph(
                operation_graph, "Operation Graph", iteration_folder
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

        logger.debug("⏱️  Step 2: Generating operation graph...")
        start_time = time.time()
        operation_graph = fuzz_operation_graph(
            target_spec, max_depth=max_depth, seed=seed
        )
        logger.debug("⏱️  Step 3: Converting to Python code...")
        start_time = time.time()
        python_code = convert_graph_to_python_code(operation_graph, seed=seed)
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
