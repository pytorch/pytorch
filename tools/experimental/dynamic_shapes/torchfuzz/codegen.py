# mypy: ignore-errors
import os
import signal
import subprocess
import sys
import tempfile
import time
from queue import Empty, Queue
from threading import Thread
from typing import Any, Optional, Union

import torch

from torchfuzz.operators import get_operator
from torchfuzz.ops_fuzzer import OperationGraph
from torchfuzz.tensor_fuzzer import ScalarSpec, Spec, TensorSpec


def convert_graph_to_python_code(
    operation_graph: OperationGraph, seed: Optional[int] = None
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

        # Generate comment for this operation
        generated_code_lines.append(
            f"    # Node {node_id}: {op_name} (depth {node.depth})"
        )

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
            operation_lines = [f"{output_var_name} = {arg_name}"]
        else:
            # Generate operation execution code
            operation_lines = generate_simple_operation_code(
                output_var_name, input_var_names, op_name, output_spec
            )

        # Add proper indentation for function body
        generated_code_lines.extend(["    " + line for line in operation_lines])
        generated_code_lines.append("")

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
        function_signature = f"def fuzzed_program({', '.join(arg_names)})"
    else:
        function_signature = "def fuzzed_program()"

    # Build the complete code
    fuzzer_dir = os.path.dirname(os.path.abspath(__file__))
    code_lines = [
        "import torch",
        "import sys",
        "import os",
        "# Add fuzzer directory to path so we can import tensor_fuzzer",
        f"fuzzer_dir = r'{fuzzer_dir}'",
        "if fuzzer_dir not in sys.path:",
        "    sys.path.insert(0, fuzzer_dir)",
        "from tensor_fuzzer import fuzz_scalar, fuzz_tensor_simple, ScalarSpec, TensorSpec",
        "",
        "# Generated fuzzed program code (topological order from operation graph)",
        f"# Graph has {len(operation_graph.nodes)} nodes",
        "",
        function_signature + ":",
    ]

    # Add the generated operation code
    code_lines.extend(generated_code_lines)

    # Add return statement
    code_lines.extend(
        [
            "    # Final result from root node",
            f"    return {final_var_name}",
            "",
        ]
    )

    # Generate argument creation code with deterministic seeds
    if arg_operations:
        code_lines.append("# Create arguments for the fuzzed program")
        for i, (node_id, spec) in enumerate(arg_operations):
            arg_name = f"arg_{i}"
            # Use a deterministic seed based on the argument index and main seed
            arg_seed = (seed + 10000 + i) if seed is not None else None

            if isinstance(spec, ScalarSpec):
                dtype_str = f"torch.{spec.dtype}".replace("torch.torch.", "torch.")
                if arg_seed is not None:
                    code_lines.extend(
                        [
                            f"scalar_spec = ScalarSpec(dtype={dtype_str})",
                            f"{arg_name} = fuzz_scalar(scalar_spec, seed={arg_seed})",
                        ]
                    )
                else:
                    code_lines.extend(
                        [
                            f"scalar_spec = ScalarSpec(dtype={dtype_str})",
                            f"{arg_name} = fuzz_scalar(scalar_spec)",
                        ]
                    )
            elif isinstance(spec, TensorSpec):
                size_str = str(spec.size)
                stride_str = str(spec.stride)
                dtype_str = f"torch.{spec.dtype}".replace("torch.torch.", "torch.")
                if arg_seed is not None:
                    code_lines.append(
                        f"{arg_name} = fuzz_tensor_simple({size_str}, {stride_str}, {dtype_str}, seed={arg_seed})"
                    )
                else:
                    code_lines.append(
                        f"{arg_name} = fuzz_tensor_simple({size_str}, {stride_str}, {dtype_str})"
                    )

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

    code_lines.extend(
        [
            "",
            "# Execute the fuzzed program both normally and with torch.compile",
            "import torch",
            "import tempfile",
            "import os",
            "import sys",
            "import contextlib",
            "from io import StringIO",
            "",
            "# Create arguments",
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
        ]
    )

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
        code_line = operator.codegen(output_var, input_vars, output_spec)
        return [code_line]
    else:
        # Fallback for unknown operations
        return [f"# Unknown operation: {op_name}"]


def execute_python_code(
    python_code: str, target_spec, preserve_temp_file: bool = False, timeout: int = 60
) -> Union[torch.Tensor, float, int, bool, complex]:
    """
    Execute the generated Python code by writing it to a file and running it.
    Supports both real-time output printing and output capturing with proper process termination.

    Args:
        python_code: String containing Python code to execute
        target_spec: Expected output specification for validation
        preserve_temp_file: If True, don't delete the temporary file after execution
        timeout: Maximum time in seconds to wait for execution (default: 60)

    Returns:
        The actual result from executing the generated code

    Raises:
        RuntimeError: With full stdout/stderr output if execution fails
        TimeoutError: If execution exceeds the timeout
    """

    # Write the generated code to a temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_generated.py", delete=False
    ) as f:
        f.write(python_code)
        generated_file_path = f.name

    print(f"📄 Generated code written to: {generated_file_path}")

    process = None
    stdout_thread = None
    stderr_thread = None

    def stream_reader(
        stream: Any, queue: "Queue[tuple[str, str]]", stream_name: str
    ) -> None:
        """Read from stream and put lines in queue with stream identifier"""
        try:
            for line in iter(stream.readline, ""):
                if line:
                    queue.put((stream_name, line.rstrip("\n")))
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def kill_process_tree(process):
        """Kill the process and all its children"""
        try:
            # Try to terminate gracefully first
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                    print("🔄 Process terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate gracefully
                    print("💀 Force killing process...")
                    process.kill()
                    try:
                        process.wait(timeout=5)
                        print("💀 Process force killed")
                    except subprocess.TimeoutExpired:
                        print("⚠️  Process may still be running after force kill")

            # Also try to kill process group if it was created
            try:
                pid = process.pid
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                time.sleep(2)
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except OSError:
                pass  # Process group might not exist or already killed

        except Exception as e:
            print(f"⚠️  Error killing process: {e}")

    try:
        # Execute the generated file with real-time output streaming
        print(f"🚀 Executing: python {generated_file_path} (timeout: {timeout}s)")
        print("=" * 50)

        # Start process with new process group to enable killing child processes
        process = subprocess.Popen(
            [sys.executable, generated_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
            preexec_fn=os.setsid,  # Create new process group  # noqa: PLW1509
        )

        # Create queues and threads for reading stdout and stderr
        output_queue = Queue()
        stdout_thread = Thread(
            target=stream_reader, args=(process.stdout, output_queue, "stdout")
        )
        stderr_thread = Thread(
            target=stream_reader, args=(process.stderr, output_queue, "stderr")
        )

        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()

        # Collect output while printing in real-time
        captured_stdout = []
        captured_stderr = []
        start_time = time.time()

        # Read output until process finishes or timeout
        while process.poll() is None:
            # Check for timeout
            if time.time() - start_time > timeout:
                print(f"⏰ Execution timeout ({timeout}s) reached, killing process...")
                kill_process_tree(process)
                raise TimeoutError(f"Execution exceeded {timeout} seconds timeout")

            try:
                stream_name, line = output_queue.get(timeout=0.1)
                if stream_name == "stdout":
                    print(line)  # Print to console in real-time
                    captured_stdout.append(line)
                elif stream_name == "stderr":
                    print(line, file=sys.stderr)  # Print to stderr in real-time
                    captured_stderr.append(line)
            except Empty:
                continue

        # Process has finished, collect any remaining output
        timeout_remaining = max(0, timeout - (time.time() - start_time))
        output_timeout = min(5, timeout_remaining)  # Max 5 seconds for remaining output

        end_time = time.time() + output_timeout
        while not output_queue.empty() and time.time() < end_time:
            try:
                stream_name, line = output_queue.get(timeout=0.1)
                if stream_name == "stdout":
                    print(line)
                    captured_stdout.append(line)
                elif stream_name == "stderr":
                    print(line, file=sys.stderr)
                    captured_stderr.append(line)
            except Empty:
                break

        # Wait for threads to finish with timeout
        if stdout_thread.is_alive():
            stdout_thread.join(timeout=2)
        if stderr_thread.is_alive():
            stderr_thread.join(timeout=2)

        # Get the return code
        return_code = process.returncode

        print("=" * 50)
        print(f"🏁 Process finished with return code: {return_code}")

        if return_code == 0:
            # Success - we already printed output in real-time
            if preserve_temp_file:
                print(f"📁 Temporary file preserved at: {generated_file_path}")
            return True
        else:
            # Failed execution
            full_output = ""
            if captured_stdout:
                full_output += "STDOUT:\n" + "\n".join(captured_stdout) + "\n"
            if captured_stderr:
                full_output += "STDERR:\n" + "\n".join(captured_stderr) + "\n"
            full_output += f"Return code: {return_code}\n"

            print(f"❌ Generated file execution failed with return code {return_code}")
            if preserve_temp_file:
                print(f"📁 Failed execution file preserved at: {generated_file_path}")
            raise RuntimeError(full_output)

    except TimeoutError:
        # Re-raise timeout error as-is
        raise
    except Exception as e:
        if hasattr(e, "returncode"):
            # This was a CalledProcessError-like exception
            raise e
        else:
            # Some other error occurred
            print(f"❌ Execution error: {e}")
            if preserve_temp_file:
                print(f"📁 Error execution file preserved at: {generated_file_path}")
            raise RuntimeError(f"Execution failed: {e}") from e
    finally:
        # Ensure process and threads are properly cleaned up
        try:
            if process is not None:
                kill_process_tree(process)
        except Exception:
            pass

        # Force cleanup threads if they're still running
        try:
            if stdout_thread is not None and stdout_thread.is_alive():
                stdout_thread.join(timeout=1)
            if stderr_thread is not None and stderr_thread.is_alive():
                stderr_thread.join(timeout=1)
        except Exception:
            pass

        # Clean up the temporary file unless preservation is requested
        if not preserve_temp_file:
            try:
                os.unlink(generated_file_path)
                print(f"🗑️  Temporary file cleaned up: {generated_file_path}")
            except Exception:
                pass
