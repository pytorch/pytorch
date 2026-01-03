# mypy: ignore-errors
import logging
import multiprocessing as mp
import os
import random
import sys


# Add parent directory to path so we can import torchfuzz as a module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from torchfuzz.codegen import convert_graph_to_python_code, create_program_file
from torchfuzz.ops_fuzzer import fuzz_operation_graph, fuzz_spec
from torchfuzz.runner import ProgramRunner
from torchfuzz.visualize_graph import visualize_operation_graph


def _parse_supported_ops_with_weights(spec: str) -> tuple[list[str], dict[str, float]]:
    """Parse --supported-ops string.

    Format: comma-separated fully-qualified torch ops, each optionally with =weight.
    Example: "torch.matmul=5,torch.nn.functional.rms_norm=5,torch.add"
    Returns (ops_list, weights_dict)
    """
    ops: list[str] = []
    weights: dict[str, float] = {}
    if not spec:
        return ops, weights
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" in entry:
            name, w = entry.split("=", 1)
            name = name.strip()
            try:
                weight = float(w.strip())
            except ValueError:
                continue
            ops.append(name)
            weights[name] = weight
        else:
            ops.append(entry)
    return ops, weights


def fuzz_and_execute(
    seed: int | None = None,
    max_depth: int | None = None,
    log_at_faluire: bool = False,
    template: str = "default",
    supported_ops: list[str] | None = None,
    op_weights: dict[str, float] | None = None,
) -> None:
    """
    Generate a fuzzed operation stack, convert it to Python code, and execute it.

    Args:
        seed: Random seed for reproducible generation. If None, uses a random seed.
        max_depth: Maximum depth for operation stack (1-10). If None, uses a random depth.

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
        max_depth = random.randint(2, 4)
    else:
        # Clamp max_depth to valid range
        max_depth = max(1, max_depth)

    print(f"Using seed: {seed}, max_depth: {max_depth}")

    # Set seed for reproducible generation
    random.seed(seed)
    torch.manual_seed(seed)
    operation_stack = None
    python_code = None
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

    import time

    try:
        logger = logging.getLogger(__name__)

        # Generate target specification first
        logger.debug("⏱️  Step 1: Generating target spec...")
        start_time = time.time()
        target_spec = fuzz_spec(template)

        # Apply user-specified operator weights (if provided)
        if op_weights:
            from torchfuzz.operators import set_operator_weights

            set_operator_weights(op_weights)
        logger.debug(
            "   Completed in %.3fs - %s", time.time() - start_time, target_spec
        )

        logger.debug("⏱️  Step 2: Generating operation graph...")
        start_time = time.time()
        operation_graph = fuzz_operation_graph(
            target_spec,
            max_depth=max_depth,
            seed=seed,
            template=template,
            supported_ops=supported_ops,
        )

        # Extract and print operation statistics
        operation_counts = {}
        for node in operation_graph.nodes.values():
            # Use the fully qualified torch operation name if available
            from torchfuzz.operators import get_operator

            # Try to get the fully qualified torch operation name
            torch_op_name = None

            # Extract the base operation name (without arg_X suffixes)
            base_op_name = node.op_name
            if node.op_name.startswith("arg_"):
                # For arg operations, use just "arg" to look up in registry
                base_op_name = "arg"

            try:
                operator = get_operator(base_op_name)
                if (
                    operator
                    and hasattr(operator, "torch_op_name")
                    and operator.torch_op_name
                ):
                    torch_op_name = operator.torch_op_name
            except (KeyError, ValueError):
                # If the operator doesn't exist in registry, use the node's op_name
                pass

            # Use fully qualified name if available, otherwise use the node's op_name
            display_name = torch_op_name if torch_op_name else node.op_name
            operation_counts[display_name] = operation_counts.get(display_name, 0) + 1

        # Print operation statistics in a parseable format
        print("OPERATION_STATS:")
        for op_name, count in sorted(operation_counts.items()):
            print(f"  {op_name}: {count}")

        logger.debug("⏱️  Step 3: Converting to Python code...")
        start_time = time.time()
        python_code = convert_graph_to_python_code(
            operation_graph, seed=seed, template=template
        )
        logger.debug(
            "   Completed in %.3fs - %d chars",
            time.time() - start_time,
            len(python_code),
        )

        logger.debug("⏱️  Step 4: Executing Python code...")
        start_time = time.time()

        # Create program file and run with new runner
        program_path = create_program_file(python_code)
        runner = ProgramRunner()
        runner.run_program(program_path)

        logger.debug("   Completed in %.3fs", time.time() - start_time)

        # # Validate the result matches target specification
        if not log_at_faluire:
            log(True)

    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        # from visualize_stack import visualize_operation_stack
        log(False)
        import traceback

        traceback.print_exc()
        error_message = str(e)
        print(f"Error: {error_message}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    try:
        from multi_process_fuzzer import run_multi_process_fuzzer, run_until_failure
    except ImportError:
        # If importing as a module fails, import from the same directory
        import os
        import sys

        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        from multi_process_fuzzer import run_multi_process_fuzzer, run_until_failure

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="PyTorch Fuzzer - Generate and test random PyTorch operations"
    )

    # Single seed execution arguments
    parser.add_argument("--seed", type=int, help="Random seed for single execution")
    parser.add_argument(
        "--max-depth", type=int, help="Maximum depth for operation stack (1-20)"
    )
    parser.add_argument(
        "--template",
        choices=["default", "dtensor", "dtensor_placements", "unbacked"],
        default="default",
        help="Template to use for code generation (default: default)",
    )
    parser.add_argument(
        "--supported-ops",
        type=str,
        help=(
            "Comma-separated fully-qualified torch ops to allow, each optionally with =weight. "
            "Examples: 'torch.matmul,torch.nn.functional.rms_norm' or "
            "'torch.matmul=5,torch.nn.functional.rms_norm=5'. Overrides template supported ops."
        ),
    )

    # Multi-process fuzzing arguments
    parser.add_argument(
        "--start", type=int, help="Starting seed value for multi-process fuzzing"
    )
    parser.add_argument(
        "--count", type=int, help="Number of seeds to run in multi-process fuzzing"
    )
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        help="Number of worker processes to use (default: auto-detected)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed output for all runs (not just failures)",
    )
    parser.add_argument(
        "--stop-at-first-failure",
        action="store_true",
        help="Pick a random seed and keep iterating until finding a failure (exits with non-zero code)",
    )

    # Legacy arguments
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run a single fuzz_and_execute (deprecated, use --seed)",
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

    # Determine execution mode
    if args.seed is not None or args.single:
        # Single seed execution mode
        print("Running single fuzz_and_execute...")
        # Parse supported ops and any inline weights from that flag
        parsed_supported_ops: list[str] | None = None
        parsed_weights: dict[str, float] = {}
        if args.supported_ops:
            parsed_supported_ops, parsed_weights = _parse_supported_ops_with_weights(
                args.supported_ops
            )

        fuzz_and_execute(
            seed=args.seed,
            max_depth=args.max_depth,
            template=args.template,
            supported_ops=parsed_supported_ops,
            op_weights=(parsed_weights if parsed_weights else None),
        )
    elif args.stop_at_first_failure:
        # Stop-at-first-failure mode
        # Default number of processes
        if args.processes is None:
            cpu_count = mp.cpu_count()
            args.processes = max(1, min(16, int(cpu_count * 0.75)))

        if args.processes < 1:
            print("❌ Error: Number of processes must be at least 1")
            sys.exit(1)

        try:
            run_until_failure(
                num_processes=args.processes,
                verbose=args.verbose,
                template=args.template,
                supported_ops=args.supported_ops,
            )
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
    elif args.start is not None or args.count is not None:
        # Multi-process fuzzing mode
        if args.start is None:
            print("❌ Error: --start is required when --count is specified")
            sys.exit(1)
        if args.count is None:
            print("❌ Error: --count is required when --start is specified")
            sys.exit(1)

        # Validate arguments
        if args.count < 1:
            print("❌ Error: --count must be at least 1")
            sys.exit(1)

        # Default number of processes
        if args.processes is None:
            cpu_count = mp.cpu_count()
            args.processes = max(1, min(16, int(cpu_count * 0.75)))

        if args.processes < 1:
            print("❌ Error: Number of processes must be at least 1")
            sys.exit(1)

        try:
            run_multi_process_fuzzer(
                num_processes=args.processes,
                seed_start=args.start,
                seed_count=args.count,
                verbose=args.verbose,
                template=args.template,
                supported_ops=args.supported_ops,
            )
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
    else:
        # Show help when no arguments are provided
        parser.print_help()
        print("\nExamples:")
        print("  python fuzzer.py --seed 42                    # Run single seed")
        print(
            "  python fuzzer.py --start 0 --count 1000       # Run multi-process fuzzing"
        )
        print("  python fuzzer.py --start 100 --count 50 -p 8  # Use 8 processes")
