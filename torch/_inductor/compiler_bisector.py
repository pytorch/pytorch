import atexit
import collections
import dataclasses
import functools
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

from torch._inductor.runtime.cache_dir_utils import cache_dir


# Set the subdirectory name
SUBDIR_NAME = "bisect"


@dataclass
class Subsystem:
    name: str


@dataclass
class BisectSubsystem(Subsystem):
    pass


@dataclass
class BinarySubsystem(Subsystem):
    pass


@dataclass
class ConfigChange(BinarySubsystem):
    name: str = field(init=False)
    config_name: str
    config_field: str
    config_value: object

    def __post_init__(self) -> None:
        self.name = f"{self.config_name}_{self.config_field}"


# Dictionary of backend -> subsystems
BACKENDS: dict[str, list[Subsystem]] = {
    # run dynamo without aot_autograd
    "eager": [],
    # run dynamo with aot_autograd, but no partitioner or decomps
    "aot_eager": [],
    # run dynamo with aot autograd, decompositions and partitioner
    "aot_eager_decomp_partition": [
        ConfigChange("aot_eager_decomp_partition", "cse", False),
        BisectSubsystem(
            "decomposition"
        ),  # number of decompositions we apply in tracing
    ],  # TODO - add cse ?
    # applies CrossRefFakeMode on invocation
    "aot_eager_decomp_partition_crossref": [],
    "inductor": [
        BisectSubsystem("pre_grad_passes"),  # passes applied on pre-grad IR
        BisectSubsystem("joint_graph_passes"),  # passes applied on joint graph
        BisectSubsystem(
            "post_grad_passes"
        ),  # passes applied individually on forward, and backward in inductor
        ConfigChange("inductor", "fallback_random", True),
        ConfigChange("inductor", "emulate_precision_casts", True),
        ConfigChange("inductor", "layout_optimization", False),
        ConfigChange("inductor", "comprehensive_padding", False),
        BisectSubsystem("lowerings"),  # lowering aten operators to inductor
    ],  # TODO - add more - fusions ?
}

subsystem_call_counter: dict[str, int] = collections.Counter()
call_counter_debug_info: dict[int, str] = {}

# For range bisection (minimizing both start and end for lowerings)
# Stores: (start_min, end_max) - the range of nodes to enable for inductor
lowering_range_bisect_state: dict[str, tuple[int, int]] = {}
# Stores all node info during final capture: list of (index, repr)
lowering_all_nodes_info: list[tuple[int, str]] = []
# The current FX graph module being lowered (for capture)
lowering_fx_graph: Optional["torch.fx.GraphModule"] = None
# Example inputs (for capture)
lowering_example_inputs: Optional[list] = None


def reset_counters() -> None:
    subsystem_call_counter.clear()
    call_counter_debug_info.clear()
    lowering_all_nodes_info.clear()


@functools.cache
def get_env_val(env_str: str) -> Optional[str]:
    return os.environ.get(env_str, None)


@dataclasses.dataclass
class LoweringBisectInfo:
    """
    Information about a minimal range of lowerings that trigger a failure.
    This is populated when bisect_lowerings_range=True and subsystem='lowerings'.
    """

    # Start and end indices of the minimal range of nodes that need inductor lowering
    start_index: int
    end_index: int
    # List of (index, node_name, node_repr) for all nodes in the minimal range
    nodes_in_range: list[tuple[int, str, str]]
    # The FX graph module being lowered (if captured)
    fx_graph: Optional["torch.fx.GraphModule"] = None
    # Example inputs for the subgraph (if captured)
    example_inputs: Optional[list] = None

    def print_graph(self) -> str:
        """Print the minimal subgraph of nodes that trigger the issue."""
        lines = [
            f"Minimal lowering range: [{self.start_index}, {self.end_index}]",
            f"Number of nodes in range: {len(self.nodes_in_range)}",
            "",
            "Nodes in range:",
        ]
        for idx, node_name, node_repr in self.nodes_in_range:
            lines.append(f"  [{idx}] {node_name}: {node_repr}")
        return "\n".join(lines)

    def get_pruned_graph(self) -> Optional["torch.fx.GraphModule"]:
        """
        Extract a pruned FX graph containing only the offending nodes.
        Inputs from outside the range become placeholder inputs.
        """
        if self.fx_graph is None:
            return None

        import copy

        import torch.fx

        # Get the node names in our minimal range
        node_names_in_range = {name for _, name, _ in self.nodes_in_range}

        # Find the actual FX nodes
        original_graph = self.fx_graph.graph
        nodes_to_include = []
        for node in original_graph.nodes:
            if node.op == "call_function" and node.name in node_names_in_range:
                nodes_to_include.append(node)

        if not nodes_to_include:
            return None

        # Create a new graph
        new_graph = torch.fx.Graph()
        env: dict = {}  # Maps old nodes to new nodes
        new_placeholders: dict = {}  # Maps input nodes to their new placeholders

        def get_new_node(old_node):
            """Get or create the corresponding node in the new graph."""
            if old_node in env:
                return env[old_node]

            # If it's not in our range, create a placeholder for it
            if old_node not in new_placeholders:
                # Create a placeholder with metadata from the original node
                placeholder_name = f"input_{old_node.name}"
                new_placeholder = new_graph.placeholder(placeholder_name)
                if "val" in old_node.meta:
                    new_placeholder.meta["val"] = old_node.meta["val"]
                if "tensor_meta" in old_node.meta:
                    new_placeholder.meta["tensor_meta"] = old_node.meta["tensor_meta"]
                new_placeholders[old_node] = new_placeholder
            return new_placeholders[old_node]

        # Copy the nodes in range to the new graph
        for node in nodes_to_include:
            # Map arguments to new graph
            new_args = []
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    if arg.name in node_names_in_range and arg in env:
                        new_args.append(env[arg])
                    else:
                        new_args.append(get_new_node(arg))
                else:
                    new_args.append(arg)

            new_kwargs = {}
            for k, v in node.kwargs.items():
                if isinstance(v, torch.fx.Node):
                    if v.name in node_names_in_range and v in env:
                        new_kwargs[k] = env[v]
                    else:
                        new_kwargs[k] = get_new_node(v)
                else:
                    new_kwargs[k] = v

            # Create the new node
            new_node = new_graph.call_function(node.target, tuple(new_args), new_kwargs)
            new_node.meta = copy.copy(node.meta)
            env[node] = new_node

        # Add output node (output the last node in range)
        if nodes_to_include:
            last_node = nodes_to_include[-1]
            if last_node in env:
                new_graph.output(env[last_node])

        # Create a new GraphModule
        new_graph.lint()
        new_gm = torch.fx.GraphModule(self.fx_graph, new_graph)
        return new_gm

    def dump_repro(self, path: str) -> None:
        """Dump a reproduction script with the minimal graph and inputs."""
        import pickle

        os.makedirs(path, exist_ok=True)

        # Dump the graph info
        graph_info_path = os.path.join(path, "graph_info.txt")
        with open(graph_info_path, "w") as f:
            f.write(self.print_graph())
        print(f"Graph info written to {graph_info_path}")

        # Dump the FX graph if available
        if self.fx_graph is not None:
            graph_path = os.path.join(path, "fx_graph.py")
            with open(graph_path, "w") as f:
                f.write(self.fx_graph.print_readable(print_output=False))
            print(f"FX graph written to {graph_path}")

        # Dump example inputs if available
        if self.example_inputs is not None:
            inputs_path = os.path.join(path, "example_inputs.pkl")
            try:
                with open(inputs_path, "wb") as f:
                    pickle.dump(self.example_inputs, f)
                print(f"Example inputs written to {inputs_path}")
            except Exception as e:
                print(f"Failed to pickle example inputs: {e}")
                # Try to save tensor metadata instead
                metadata_path = os.path.join(path, "inputs_metadata.txt")
                with open(metadata_path, "w") as f:
                    for i, inp in enumerate(self.example_inputs):
                        if hasattr(inp, "shape"):
                            f.write(
                                f"Input {i}: shape={inp.shape}, dtype={inp.dtype}, device={inp.device}\n"
                            )
                        else:
                            f.write(f"Input {i}: {type(inp)} = {inp}\n")
                print(f"Input metadata written to {metadata_path}")


@dataclasses.dataclass
class BisectionResult:
    """
    backend: torch.compile backend responsible for failure
    subsystem: optional, registered component identified for failure
    bisect_number: optional, number of times the subsystem needed to be applied to trigger failure
    debug_info: associated info of the triggering bisect application of subsystem
    lowering_info: optional, detailed info about minimal lowering range (for lowerings subsystem)
    """

    backend: str
    subsystem: Optional[str] = None
    bisect_number: Optional[int] = None
    debug_info: Optional[str] = None
    lowering_info: Optional[LoweringBisectInfo] = None


class CompilerBisector:
    """
    This class iteratively runs torch.compile backends (eager, aot_eager, inductor) to find the
    first backend that can repro an issue.

    Once it discovers the offending backend it will iteratively disable subsystems within the backend.
    For subsystems which are applied repeatedly, such as the number of post grad passes or number
    of lowering of nodes to inductor ir, it will bisect to find the offending application.

    The idiomatic way to run it is with `do_bisect`. You can also use it by setting the env flags
    `TORCH_BISECT_BACKEND`, `TORCH_BISECT_SUBSYSTEM` and `TORCH_BISECT_MAX`.

    It also supports a CLI interface, although this is less well tested.

    You must run python compiler_bisector.py [start | good | bad | end]
    """

    bisection_enabled: bool = False

    in_process_cache: Optional[str] = None

    @classmethod
    def get_dir(cls) -> str:
        return f"{cache_dir() if not cls.in_process_cache else cls.in_process_cache}/{SUBDIR_NAME}"

    @classmethod
    def write_lines_to_file(cls, file_path: str, lines: list[str]) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            file.writelines(lines)

    @classmethod
    def read_lines_from_file(cls, file_path: str) -> list[str]:
        if os.path.exists(file_path):
            with open(file_path) as file:
                return file.readlines()
        return []

    @classmethod
    def update_run_state(
        cls, backend_name: str, subsystem: Subsystem, run_state: str
    ) -> None:
        file_path = os.path.join(
            cls.get_dir(), backend_name, f"{subsystem.name}_run_state.txt"
        )
        if isinstance(subsystem, ConfigChange):
            assert run_state == "test_disable"
            cls.set_config_values(
                backend_name,
                subsystem.name,
                {subsystem.config_field: subsystem.config_value},
            )

        cls.write_lines_to_file(file_path, [run_state])

    @classmethod
    def set_config_values(
        cls, backend: str, subsystem: str, config_data: dict[str, object]
    ) -> None:
        file_path = os.path.join(cls.get_dir(), backend, f"{subsystem}_config.txt")
        lines = [f"{k}={v}\n" for k, v in config_data.items()]
        cls.write_lines_to_file(file_path, lines)

    @classmethod
    def update_bisect_status(cls, backend_name: str, subsystem_name: str) -> None:
        assert isinstance(subsystem_name, str)
        file_path = os.path.join(cls.get_dir(), "bisect_status.txt")
        lines = [f"backend={backend_name}\n", f"subsystem={subsystem_name}\n"]
        cls.write_lines_to_file(file_path, lines)

    @classmethod
    def update_bisect_range(
        cls, backend_name: str, subsystem_name: str, low: int, high: int
    ) -> None:
        assert isinstance(subsystem_name, str)
        file_path = os.path.join(
            cls.get_dir(), backend_name, f"{subsystem_name}_bisect_range.txt"
        )
        lines = [f"low={low}\n", f"high={high}\n"]
        cls.write_lines_to_file(file_path, lines)

    @classmethod
    def update_start_bisect_range(
        cls, backend_name: str, subsystem_name: str, low: int, high: int
    ) -> None:
        """Update the bisect range for minimizing from the start."""
        assert isinstance(subsystem_name, str)
        file_path = os.path.join(
            cls.get_dir(), backend_name, f"{subsystem_name}_start_bisect_range.txt"
        )
        lines = [f"low={low}\n", f"high={high}\n"]
        cls.write_lines_to_file(file_path, lines)

    @classmethod
    def get_start_bisect_range(
        cls, backend_name: str, subsystem_name: str
    ) -> tuple[int, int]:
        """Get the bisect range for minimizing from the start."""
        file_path = os.path.join(
            cls.get_dir(), backend_name, f"{subsystem_name}_start_bisect_range.txt"
        )
        lines = cls.read_lines_from_file(file_path)
        low = None
        high = None
        for line in reversed(lines):
            if line.startswith("low="):
                low = int(line.strip().split("=")[1])
            elif line.startswith("high="):
                high = int(line.strip().split("=")[1])

            if low is not None and high is not None:
                break

        if low is None or high is None:
            raise RuntimeError(
                f"Trying to get start bisect range when it is not set: subsystem {subsystem_name}"
            )

        return low, high

    @classmethod
    def get_backend(cls) -> Optional[str]:
        """
        Returns the active backend, if any
        """
        if val := get_env_val("TORCH_BISECT_BACKEND"):
            return val

        file_path = os.path.join(cls.get_dir(), "bisect_status.txt")
        lines = cls.read_lines_from_file(file_path)
        for line in lines:
            if line.startswith("backend="):
                return line.strip().split("=")[1]
        return None

    @classmethod
    def get_subsystem(cls) -> Optional[str]:
        """
        Returns the active subsystem, if any
        """

        if val := get_env_val("TORCH_BISECT_SUBSYSTEM"):
            return val

        file_path = os.path.join(cls.get_dir(), "bisect_status.txt")
        lines = cls.read_lines_from_file(file_path)
        for line in lines:
            if line.startswith("subsystem="):
                out = line.strip().split("=")[1]
                return out if out else None
        return None

    @classmethod
    def get_subsystem_object(cls, backend_name: str, subsystem_name: str) -> Subsystem:
        return next(obj for obj in BACKENDS[backend_name] if obj.name == subsystem_name)

    @classmethod
    def get_run_state(cls, backend_name: str, subsystem_name: str) -> Optional[str]:
        """
        Returns the current stage of bisecting, if Any.
        States:
          - test_disable: Disable subsystem completely to see if issue is fixed
          - find_max_bounds: Find upper bound of node count
          - bisect: Bisect to find the problematic node (from start)
          - bisect_start: (lowerings only) Bisect from start to minimize range
          - capture: (lowerings only) Final run to capture node info for reporting
        """

        file_path = os.path.join(
            cls.get_dir(), backend_name, f"{subsystem_name}_run_state.txt"
        )
        lines = cls.read_lines_from_file(file_path)
        if lines:
            out = lines[0].strip()
            assert out in (
                "test_disable",
                "find_max_bounds",
                "bisect",
                "bisect_start",
                "capture",
            )
            return out
        return None

    @classmethod
    def get_bisect_range(
        cls, backend_name: str, subsystem_name: str
    ) -> tuple[int, int]:
        file_path = os.path.join(
            cls.get_dir(), backend_name, f"{subsystem_name}_bisect_range.txt"
        )
        lines = cls.read_lines_from_file(file_path)
        low = None
        high = None
        # pyrefly: ignore [bad-assignment]
        for line in reversed(lines):
            if line.startswith("low="):
                low = int(line.strip().split("=")[1])
            elif line.startswith("high="):
                high = int(line.strip().split("=")[1])

            if low is not None and high is not None:
                break

        if low is None or high is None:
            raise RuntimeError(
                f"Trying to get bisect range when it is not set: subsystem {subsystem_name}"
            )

        return low, high

    @classmethod
    def update_config_change(cls, backend: str, subsystem: ConfigChange) -> None:
        file_path = os.path.join(cls.get_dir(), backend, f"{subsystem.name}_config.txt")
        lines = [
            f"config_name={subsystem.config_name}\n",
            f"config_field={subsystem.config_field}\n",
            f"config_value={subsystem.config_value}\n",
        ]
        cls.write_lines_to_file(file_path, lines)

    @classmethod
    def get_config_change(cls, config_name: str) -> Optional[dict[str, object]]:
        backend = cls.get_backend()
        subsystem = cls.get_subsystem()

        if not backend or not subsystem:
            return None

        file_path = os.path.join(cls.get_dir(), backend, f"{subsystem}_config.txt")

        if not os.path.exists(file_path):
            return None

        lines = cls.read_lines_from_file(file_path)
        config_data = {}
        for line in lines:
            key, value = line.strip().split("=", 1)
            config_data[key] = eval(value)

        return config_data

    @classmethod
    def delete_bisect_status(cls) -> None:
        # in process_cache we have created if it exists, just the subdirectory of non created dir
        dir_name = cls.in_process_cache if cls.in_process_cache else cls.get_dir()
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print("Bisection status deleted.")
        else:
            print("No bisection status found.")

    @classmethod
    def get_system_counter(cls, name: str, increment: bool = True) -> int:
        global subsystem_call_counter
        curr = subsystem_call_counter[name]
        if increment:
            subsystem_call_counter[name] += 1
        return curr

    @classmethod
    def set_lowering_graph_info(
        cls,
        fx_graph: "torch.fx.GraphModule",
        example_inputs: list,
    ) -> None:
        """
        Set the FX graph and example inputs for lowering bisection.
        Called from GraphLowering to capture the graph being lowered.
        """
        global lowering_fx_graph, lowering_example_inputs
        if not cls.bisection_enabled:
            return
        if cls.get_backend() != "inductor" or cls.get_subsystem() != "lowerings":
            return
        run_state = cls.get_run_state("inductor", "lowerings")
        if run_state == "capture":
            lowering_fx_graph = fx_graph
            lowering_example_inputs = example_inputs

    @classmethod
    def disable_subsystem(
        cls,
        backend: str,
        subsystem: str,
        debug_info: Optional[Callable[[], str]] = None,
    ) -> bool:
        if not cls.bisection_enabled:
            return False

        if cls.get_backend() != backend:
            return False

        if cls.get_subsystem() != subsystem:
            return False

        if val := get_env_val("TORCH_BISECT_MAX"):
            counter = cls.get_system_counter(subsystem, increment=True)
            return counter > int(val)

        run_state = cls.get_run_state(backend, subsystem)
        if run_state == "test_disable":
            # First run, disable completely
            return True
        elif run_state == "find_max_bounds":
            # Second run, update bisection range and return True to enable the subsystem
            cls.update_bisect_range(
                backend,
                subsystem,
                0,
                cls.get_system_counter(subsystem, increment=True),
            )
            return False
        elif run_state == "bisect":
            # Bisect to find the minimal end index (standard bisection)
            low, high = cls.get_bisect_range(backend, subsystem)
            midpoint = (low + high) // 2
            call_counter = cls.get_system_counter(subsystem)

            if (
                call_counter >= low
                and call_counter <= high
                and (high - low) <= 2
                and debug_info is not None
            ):
                call_counter_debug_info[call_counter] = debug_info()

            return call_counter > midpoint
        elif run_state == "bisect_start":
            # Bisect from the start to find the minimal start index
            # end_index is already known (stored in bisect_range as 'low' after bisect phase)
            end_index, _ = cls.get_bisect_range(backend, subsystem)
            start_low, start_high = cls.get_start_bisect_range(backend, subsystem)
            start_midpoint = (start_low + start_high) // 2
            call_counter = cls.get_system_counter(subsystem)

            if (
                call_counter >= start_low
                and call_counter <= end_index
                and (start_high - start_low) <= 2
                and debug_info is not None
            ):
                call_counter_debug_info[call_counter] = debug_info()

            # Disable node if it's before start_midpoint OR after end_index
            return call_counter < start_midpoint or call_counter > end_index
        elif run_state == "capture":
            # Final capture run: enable only nodes in the minimal range and record info
            end_index, _ = cls.get_bisect_range(backend, subsystem)
            start_index, _ = cls.get_start_bisect_range(backend, subsystem)
            call_counter = cls.get_system_counter(subsystem)

            # Record node info for all nodes in range
            if (
                call_counter >= start_index
                and call_counter <= end_index
                and debug_info is not None
            ):
                info = debug_info()
                # Handle both tuple (name, repr) and plain string formats
                if isinstance(info, tuple):
                    node_name, node_repr = info
                    lowering_all_nodes_info.append((call_counter, node_name, node_repr))
                else:
                    lowering_all_nodes_info.append((call_counter, str(call_counter), info))

            # Disable node if it's outside the minimal range
            return call_counter < start_index or call_counter > end_index
        else:
            raise RuntimeError(f"Unexpected run_state: {run_state}")

    @classmethod
    def advance_subsystem(
        cls, curr_backend: str, curr_subsystem: Subsystem
    ) -> Optional[Subsystem]:
        """
        Tries to move to the next subsystem within the current system.
        """
        print(f"Disabling {curr_subsystem.name} did not fix the issue.")

        current_subsystems = BACKENDS[curr_backend]
        current_subsystem_index = next(
            i
            for i, subsystem in enumerate(current_subsystems)
            if subsystem.name == curr_subsystem.name
        )

        if current_subsystem_index < len(current_subsystems) - 1:
            next_subsystem = current_subsystems[current_subsystem_index + 1]
            cls.update_bisect_status(curr_backend, next_subsystem.name)
            cls.update_run_state(curr_backend, next_subsystem, "test_disable")
            print(
                f"Moving to the next subsystem: {curr_backend} - {next_subsystem.name}"
            )
            return next_subsystem
        else:
            print(
                f"All subsystems in {curr_backend} have been checked. The issue is not in this system."
            )
            return None

    @classmethod
    def advance_backend(cls, curr_backend: str) -> Optional[str]:
        """
        Tries Move to the next backend.
        """
        current_system_index = list(BACKENDS.keys()).index(curr_backend)

        if current_system_index < len(BACKENDS) - 1:
            curr_backend = list(BACKENDS.keys())[current_system_index + 1]
            cls.update_bisect_status(curr_backend, "")
            print(f"Moving to the next system: {curr_backend}")
            return curr_backend
        else:
            return None

    @classmethod
    def process_subsystem(
        cls,
        curr_backend: str,
        curr_subsystem: Subsystem,
        fn: Callable[[], bool],
        cli_interface: bool = True,
    ) -> bool:
        """
        Process the current subsystem. Returns True if the issue is found, False otherwise.
        """
        assert isinstance(curr_subsystem, Subsystem)
        while True:
            run_state = cls.get_run_state(curr_backend, curr_subsystem.name)
            reset_counters()
            if run_state == "test_disable":
                if not fn():
                    next_subsystem = cls.advance_subsystem(curr_backend, curr_subsystem)
                    if not next_subsystem:
                        return False
                    curr_subsystem = next_subsystem
                else:
                    if isinstance(curr_subsystem, ConfigChange):
                        print(
                            f"Setting config {curr_subsystem.config_name} field {curr_subsystem.config_field} "
                            f"to {curr_subsystem.config_value} fixed the issue"
                        )
                    else:
                        print(f"Disabling {curr_subsystem.name} fixed the issue.")
                    if isinstance(curr_subsystem, BinarySubsystem):
                        return True
                    print("Starting bisect by getting upper bound.")
                    cls.update_run_state(
                        curr_backend, curr_subsystem, "find_max_bounds"
                    )
            elif run_state == "find_max_bounds":
                if fn():
                    raise RuntimeError(
                        f"Function succeeded with 'find_max_bounds' status for {curr_backend} - {curr_subsystem.name}."
                    )
                else:
                    _, high = cls.get_bisect_range(curr_backend, curr_subsystem.name)
                    print(f"Upper bound of {high} found for {curr_backend}.")
                    cls.update_run_state(curr_backend, curr_subsystem, "bisect")
            elif run_state == "bisect":
                low, high = cls.get_bisect_range(curr_backend, curr_subsystem.name)
                midpoint = (low + high) // 2
                print(
                    f"Bisecting {curr_backend} - {curr_subsystem.name} end (Range: [{low}, {high}], Midpoint: {midpoint})"
                )
                if fn():
                    cls.update_bisect_range(
                        curr_backend, curr_subsystem.name, midpoint + 1, high
                    )
                else:
                    cls.update_bisect_range(
                        curr_backend, curr_subsystem.name, low, midpoint
                    )
                low, high = cls.get_bisect_range(curr_backend, curr_subsystem.name)
                if low == high:
                    # For lowerings subsystem, proceed to bisect from start
                    if curr_subsystem.name == "lowerings":
                        print(
                            f"Found minimal end index: {low}. Now bisecting from start..."
                        )
                        # Initialize start bisect range: [0, low]
                        cls.update_start_bisect_range(
                            curr_backend, curr_subsystem.name, 0, low
                        )
                        cls.update_run_state(curr_backend, curr_subsystem, "bisect_start")
                    else:
                        print(
                            f"Binary search completed for {curr_backend} - {curr_subsystem.name}. The bisect number is {low}. "
                            f"Debug info: {call_counter_debug_info.get(low, 'not found')}"
                        )
                        return True
            elif run_state == "bisect_start":
                # Bisect from start to find minimal start index (highest start that still triggers bug)
                end_index, _ = cls.get_bisect_range(curr_backend, curr_subsystem.name)
                start_low, start_high = cls.get_start_bisect_range(
                    curr_backend, curr_subsystem.name
                )
                start_midpoint = (start_low + start_high) // 2
                print(
                    f"Bisecting {curr_backend} - {curr_subsystem.name} start "
                    f"(Start range: [{start_low}, {start_high}], End: {end_index}, Midpoint: {start_midpoint})"
                )
                if fn():
                    # Test passed (no bug) = start is too high, need more nodes
                    cls.update_start_bisect_range(
                        curr_backend, curr_subsystem.name, start_low, start_midpoint
                    )
                else:
                    # Bug still reproduces, can try higher start (fewer nodes)
                    cls.update_start_bisect_range(
                        curr_backend, curr_subsystem.name, start_midpoint + 1, start_high
                    )
                start_low, start_high = cls.get_start_bisect_range(
                    curr_backend, curr_subsystem.name
                )
                if start_low == start_high:
                    print(
                        f"Found minimal range: [{start_low}, {end_index}] "
                        f"({end_index - start_low + 1} nodes)"
                    )
                    # Proceed to capture phase to record all node info
                    cls.update_run_state(curr_backend, curr_subsystem, "capture")
            elif run_state == "capture":
                # Final capture run to record node info
                end_index, _ = cls.get_bisect_range(curr_backend, curr_subsystem.name)
                start_index, _ = cls.get_start_bisect_range(
                    curr_backend, curr_subsystem.name
                )
                print(
                    f"Capturing node info for range [{start_index}, {end_index}]..."
                )
                fn()  # Run to capture node info
                print(
                    f"Binary search completed for {curr_backend} - {curr_subsystem.name}. "
                    f"Minimal range: [{start_index}, {end_index}] ({end_index - start_index + 1} nodes)"
                )
                return True
            else:
                raise RuntimeError(f"Unexpected run_state {run_state}")

            if cli_interface:
                sys.exit(0)

    @classmethod
    def initialize_system(cls) -> None:
        curr_backend = next(iter(BACKENDS.keys()))
        curr_subsystem = ""
        cls.update_bisect_status(curr_backend, curr_subsystem)
        print(f"Starting bisection process with system: {curr_backend}")

    @classmethod
    def do_bisect(
        cls, fn: Callable[[], bool], cli_interface: bool = False
    ) -> Optional[BisectionResult]:
        """
        Run fn repeatedly attempting to bisect torch.compile. fn should return True on success and False on failure.
        """

        # TODO graph bisecting is not well composed with lowering
        # bisector so far. Use a config to opt-in
        import torch._inductor.config as inductor_config

        if inductor_config.test_configs.bisect_pre_grad_graph:
            BACKENDS["inductor"].insert(0, BisectSubsystem("pre_grad_graph"))

        if not cli_interface:
            bisection_enabled_orig = cls.bisection_enabled
            cls.delete_bisect_status()
            cls.bisection_enabled = True
            cls.in_process_cache = tempfile.mkdtemp()

            def cleanup() -> None:
                cls.bisection_enabled = bisection_enabled_orig
                cls.delete_bisect_status()
                cls.in_process_cache = None

                if BACKENDS["inductor"][0].name == "pre_grad_graph":
                    del BACKENDS["inductor"][0]

            cleanup_handler = atexit.register(cleanup)

            class DisableBisect:
                def __del__(self) -> None:
                    cleanup()
                    atexit.unregister(cleanup_handler)

            _cleanup = DisableBisect()

        curr_backend = cls.get_backend()
        curr_subsystem_name = cls.get_subsystem()

        if not curr_backend:
            cls.initialize_system()
            curr_backend = cls.get_backend()
            assert curr_backend is not None
            curr_subsystem_name = cls.get_subsystem()

        curr_subsystem = (
            cls.get_subsystem_object(curr_backend, curr_subsystem_name)
            if curr_subsystem_name is not None
            else None
        )
        while True:
            assert curr_backend is not None
            reset_counters()
            if curr_subsystem:
                result = cls.process_subsystem(
                    curr_backend, curr_subsystem, fn, cli_interface=cli_interface
                )
                if result:
                    curr_subsystem = cls.get_subsystem_object(
                        curr_backend,
                        cls.get_subsystem(),  # type: ignore[arg-type]
                    )

                    if isinstance(curr_subsystem, BinarySubsystem):
                        return BisectionResult(
                            curr_backend,
                            curr_subsystem.name,
                            0,
                            curr_subsystem.name,
                        )

                    low, _ = cls.get_bisect_range(curr_backend, curr_subsystem.name)

                    # For lowerings, include the LoweringBisectInfo
                    lowering_info = None
                    debug_info = call_counter_debug_info.get(low)
                    if curr_subsystem.name == "lowerings":
                        try:
                            start_index, _ = cls.get_start_bisect_range(
                                curr_backend, curr_subsystem.name
                            )
                            lowering_info = LoweringBisectInfo(
                                start_index=start_index,
                                end_index=low,
                                nodes_in_range=list(lowering_all_nodes_info),
                                fx_graph=lowering_fx_graph,
                                example_inputs=lowering_example_inputs,
                            )
                            # Set debug_info from the captured nodes if not already set
                            if debug_info is None and lowering_all_nodes_info:
                                # Use the node names as debug info
                                node_names = [
                                    node_name
                                    for _, node_name, _ in lowering_all_nodes_info
                                ]
                                debug_info = ", ".join(node_names)
                        except RuntimeError:
                            # start_bisect_range not set (old behavior fallback)
                            pass

                    return BisectionResult(
                        curr_backend,
                        curr_subsystem.name,
                        low,
                        debug_info,
                        lowering_info=lowering_info,
                    )

                next_subsystem = cls.advance_subsystem(curr_backend, curr_subsystem)
                if not next_subsystem:
                    print(
                        f"The issue is in the {curr_backend} system, but could not identify subsystem."
                    )
                    assert curr_backend is not None
                    return BisectionResult(curr_backend)

                curr_subsystem = next_subsystem
            else:
                if fn():
                    next_backend = cls.advance_backend(curr_backend)
                    if not next_backend:
                        print("All systems have been checked.")
                        return None

                    curr_backend = next_backend
                else:
                    current_subsystems = BACKENDS[curr_backend]
                    if current_subsystems:
                        curr_subsystem = current_subsystems[0]
                        cls.update_bisect_status(curr_backend, curr_subsystem.name)
                        cls.update_run_state(
                            curr_backend, curr_subsystem, "test_disable"
                        )
                        print(
                            f"The issue is in the {curr_backend} system. Moving to the first subsystem: {curr_subsystem}"
                        )
                    else:
                        print(f"The issue is in the {curr_backend} system.")
                        return BisectionResult(curr_backend)

            if cli_interface:
                sys.exit(0)


def command_line_usage() -> None:
    if len(sys.argv) < 2:
        print("Usage: python bisect_update.py <start|end|good|bad>")
        sys.exit(1)

    bisection_manager = CompilerBisector()
    command = sys.argv[1]

    if command == "end":
        bisection_manager.delete_bisect_status()
        sys.exit(0)

    if command == "start":
        bisection_manager.delete_bisect_status()
        bisection_manager.initialize_system()
        sys.exit(0)

    if command not in ["good", "bad"]:
        print("Invalid command. Must be 'good', 'bad', 'start', or 'end'.")
        sys.exit(1)

    def test_function() -> bool:
        return command == "good"

    if not bisection_manager.get_backend():
        raise ValueError("Must call start prior to good or bad")

    bisection_manager.do_bisect(test_function, cli_interface=True)


def get_is_bisection_enabled() -> bool:
    return (
        CompilerBisector.get_subsystem() is not None
        or CompilerBisector.get_backend() is not None
    )


CompilerBisector.bisection_enabled = get_is_bisection_enabled()

if __name__ == "__main__":
    command_line_usage()
