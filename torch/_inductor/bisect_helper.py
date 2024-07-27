import collections
import functools
import os
import shutil
import sys
from typing import Callable, Dict, List, Optional, Tuple

from torch._inductor.runtime.cache_dir_utils import cache_dir

# Set the subdirectory name
SUBDIR_NAME = "bisect"

# Dictionary of subsystems
SUBSYSTEMS: Dict[str, List[str]] = {
    "eager": [],
    "aot_eager": [],
    "aot_eager_decomp_partition": ["decomposition"],
    # TODO - add impls here "inductor": ["subtensor_fusion", "epilogue_fusion", "precompute_fusion"]
}

subsystem_call_counter: Dict[str, int] = collections.Counter()


@functools.lru_cache(None)
def get_env_val(env_str: str) -> Optional[str]:
    return os.environ.get(env_str, None)


class BisectionManager:
    bisection_enabled: bool = False

    @classmethod
    def get_dir(cls) -> str:
        return f"{cache_dir()}/{SUBDIR_NAME}"

    @classmethod
    def write_lines_to_file(cls, file_path: str, lines: List[str]) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            file.writelines(lines)

    @classmethod
    def read_lines_from_file(cls, file_path: str) -> List[str]:
        if os.path.exists(file_path):
            with open(file_path) as file:
                return file.readlines()
        return []

    @classmethod
    def update_run_state(
        cls, system_name: str, subsystem_name: str, run_state: str
    ) -> None:
        file_path = os.path.join(
            cls.get_dir(), system_name, f"{subsystem_name}_run_state.txt"
        )
        cls.write_lines_to_file(file_path, [run_state])

    @classmethod
    def update_bisect_status(cls, system_name: str, subsystem_name: str) -> None:
        file_path = os.path.join(cls.get_dir(), "bisect_status.txt")
        lines = [f"system={system_name}\n", f"subsystem={subsystem_name}\n"]
        cls.write_lines_to_file(file_path, lines)

    @classmethod
    def update_bisect_range(
        cls, system_name: str, subsystem_name: str, low: int, high: int
    ) -> None:
        file_path = os.path.join(
            cls.get_dir(), system_name, f"{subsystem_name}_bisect_range.txt"
        )
        lines = [f"low={low}\n", f"high={high}\n"]
        cls.write_lines_to_file(file_path, lines)

    @classmethod
    def get_current_system(cls) -> str:
        if val := get_env_val("TORCH_BISECT_SYSTEM"):
            return val

        file_path = os.path.join(cls.get_dir(), "bisect_status.txt")
        lines = cls.read_lines_from_file(file_path)
        for line in lines:
            if line.startswith("system="):
                return line.strip().split("=")[1]
        return ""

    @classmethod
    def get_current_subsystem(cls) -> str:
        if val := get_env_val("TORCH_BISECT_SUBSYSTEM"):
            return val

        file_path = os.path.join(cls.get_dir(), "bisect_status.txt")
        lines = cls.read_lines_from_file(file_path)
        for line in lines:
            if line.startswith("subsystem="):
                return line.strip().split("=")[1]
        return ""

    @classmethod
    def get_run_state(cls, system_name: str, subsystem_name: str) -> str:
        file_path = os.path.join(
            cls.get_dir(), system_name, f"{subsystem_name}_run_state.txt"
        )
        lines = cls.read_lines_from_file(file_path)
        if lines:
            return lines[0].strip()
        return ""

    @classmethod
    def get_bisect_range(cls, system_name: str, subsystem_name: str) -> Tuple[int, int]:
        file_path = os.path.join(
            cls.get_dir(), system_name, f"{subsystem_name}_bisect_range.txt"
        )
        lines = cls.read_lines_from_file(file_path)
        low = 0
        high = 0
        for line in lines:
            if line.startswith("low="):
                low = int(line.strip().split("=")[1])
            elif line.startswith("high="):
                high = int(line.strip().split("=")[1])
        return low, high

    @classmethod
    def delete_bisect_status(cls) -> None:
        if os.path.exists(cls.get_dir()):
            shutil.rmtree(cls.get_dir())
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
    def torch_bisect(cls, *systems: str) -> bool:
        if not cls.bisection_enabled:
            return True

        current_system = cls.get_current_system()

        if current_system != systems[0]:
            return True

        current_subsystem = cls.get_current_subsystem()

        subsystem_name = systems[1]
        if current_subsystem != subsystem_name:
            return True

        if val := get_env_val("TORCH_BISECT_MAX"):
            return cls.get_system_counter(subsystem_name) <= int(val)

        run_state = cls.get_run_state(current_system, current_subsystem)

        if run_state == "test_disable":
            # First run, disable completely
            # print(current_subsystem, subsystem_name, "disable")
            return False
        elif run_state == "find_max_bounds":
            # Second run, update bisection range and return True to enable the subsystem
            cls.update_bisect_range(
                current_system,
                current_subsystem,
                0,
                cls.get_system_counter(current_subsystem, increment=True),
            )
            return True
        else:
            # If the environment variable is not set, use the bisection range midpoint
            low, high = cls.get_bisect_range(current_system, current_subsystem)
            midpoint = (low + high) // 2
            call_counter = cls.get_system_counter(subsystem_name)
            return call_counter <= midpoint

    @classmethod
    def advance_subsystem(
        cls, current_system: str, current_subsystem: str
    ) -> Tuple[bool, str, str]:
        """
        Move to the next subsystem within the current system.

        :param current_system: The current system name.
        :param current_subsystem: The current subsystem name.
        :return: A tuple containing a boolean indicating if there are more subsystems, and the updated system and subsystem names.
        """
        print(f"Disabling {current_subsystem} did not fix the issue.")

        current_subsystems = SUBSYSTEMS[current_system]
        current_subsystem_index = current_subsystems.index(current_subsystem)

        if current_subsystem_index < len(current_subsystems) - 1:
            current_subsystem = current_subsystems[current_subsystem_index + 1]
            cls.update_bisect_status(current_system, current_subsystem)
            cls.update_run_state(current_system, current_subsystem, "test_disable")
            print(
                f"Moving to the next subsystem: {current_system} - {current_subsystem}"
            )
            return True, current_system, current_subsystem
        else:
            print(
                f"All subsystems in {current_system} have been checked. The issue is not in this system."
            )
            return False, current_system, current_subsystem

    @classmethod
    def advance_system(cls, current_system: str) -> tuple[str, str]:
        """
        Move to the next system.

        :param current_system: The current system name.
        :return: A tuple containing the updated system and subsystem names.
        """
        current_system_index = list(SUBSYSTEMS.keys()).index(current_system)

        if current_system_index < len(SUBSYSTEMS) - 1:
            current_system = list(SUBSYSTEMS.keys())[current_system_index + 1]
            current_subsystem = ""
            cls.update_bisect_status(current_system, current_subsystem)
            print(f"Moving to the next system: {current_system}")
        else:
            print("All systems have been checked.")
            current_system, current_subsystem = "", ""

        return current_system, current_subsystem

    @classmethod
    def perform_bisection(
        cls,
        current_system: str,
        current_subsystem: str,
        fn: Callable[[], bool],
        cli_interface: bool = True,
    ) -> bool:
        """
        Perform the bisection process for the current system and subsystem.

        :param current_system: The current system name.
        :param current_subsystem: The current subsystem name.
        :param fn: The test function to determine if the issue is fixed.
        :param cli_interface: Flag indicating whether the method is being called from the CLI interface.
        :return: True if the issue is found, False otherwise.
        """
        while True:
            run_state = cls.get_run_state(current_system, current_subsystem)
            subsystem_call_counter.clear()
            if run_state == "test_disable":
                if not fn():
                    (
                        has_more_subsystems,
                        current_system,
                        current_subsystem,
                    ) = cls.advance_subsystem(current_system, current_subsystem)
                    if not has_more_subsystems:
                        return False
                else:
                    print(
                        f"Disabling {current_subsystem} fixed the issue. Starting bisect by getting upper bound."
                    )
                    cls.update_run_state(
                        current_system, current_subsystem, "find_max_bounds"
                    )
            elif run_state == "find_max_bounds":
                if fn():
                    raise RuntimeError(
                        f"Function succeeded with 'find_max_bounds' status for {current_system} - {current_subsystem}."
                    )
                else:
                    _, high = cls.get_bisect_range(current_system, current_subsystem)
                    print(f"Upper bound of {high} found for {current_system}.")
                    cls.update_run_state(current_system, current_subsystem, "bisect")
            elif run_state == "bisect":
                low, high = cls.get_bisect_range(current_system, current_subsystem)
                midpoint = (low + high) // 2
                print(
                    f"Bisecting {current_system} - {current_subsystem} (Range: [{low}, {high}], Midpoint: {midpoint})"
                )
                if fn():
                    cls.update_bisect_range(
                        current_system, current_subsystem, midpoint + 1, high
                    )
                else:
                    cls.update_bisect_range(
                        current_system, current_subsystem, low, midpoint
                    )
                low, high = cls.get_bisect_range(current_system, current_subsystem)
                if low == high:
                    print(
                        f"Binary search completed for {current_system} - {current_subsystem}. The bad number is {low}."
                    )
                    return True
            else:
                raise RuntimeError(f"Unexpected run_state {run_state}")

            if cli_interface:
                sys.exit(0)

    @classmethod
    def initialize_system(cls) -> None:
        current_system = next(iter(SUBSYSTEMS.keys()))
        current_subsystem = ""
        cls.update_bisect_status(current_system, current_subsystem)
        print(f"Starting bisection process with system: {current_system}")

    @classmethod
    def do_bisect(
        cls, fn: Callable[[], bool], cli_interface: bool = False
    ) -> Tuple[List[str], int]:
        if not cli_interface:
            cls.delete_bisect_status()
            cls.bisection_enabled = True

            class DisableBisect:
                def __del__(self) -> None:
                    cls.bisection_enabled = False
                    cls.delete_bisect_status()

            cleanup = DisableBisect()

        current_system = cls.get_current_system()
        current_subsystem = cls.get_current_subsystem()

        if not current_system:
            cls.initialize_system()
            current_system = cls.get_current_system()
            current_subsystem = cls.get_current_subsystem()

        while True:
            subsystem_call_counter.clear()
            if current_subsystem:
                result = cls.perform_bisection(
                    current_system, current_subsystem, fn, cli_interface=cli_interface
                )
                if result:
                    current_subsystem = cls.get_current_subsystem()
                    low, _ = cls.get_bisect_range(current_system, current_subsystem)
                    return ([current_system, current_subsystem], low)

                (
                    has_more_subsystems,
                    next_system,
                    next_subsystem,
                ) = cls.advance_subsystem(current_system, current_subsystem)
                if not has_more_subsystems:
                    print(
                        f"The issue is in the {current_system} system, but could not identify subsystem."
                    )
                    return ([current_system], 0)

                if not next_system:
                    print("All systems have been checked.")
                    return ([], 0)

                current_system, current_subsystem = next_system, next_subsystem
            else:
                if fn():
                    current_system, current_subsystem = cls.advance_system(
                        current_system
                    )
                    if not current_system:
                        print("All systems have been checked.")
                        return ([], 0)
                else:
                    current_subsystems = SUBSYSTEMS[current_system]
                    if current_subsystems:
                        current_subsystem = current_subsystems[0]
                        cls.update_bisect_status(current_system, current_subsystem)
                        cls.update_run_state(
                            current_system, current_subsystem, "test_disable"
                        )
                        print(
                            f"The issue is in the {current_system} system. Moving to the first subsystem: {current_subsystem}"
                        )
                    else:
                        print(f"The issue is in the {current_system} system.")
                        return ([current_system], 0)

            if cli_interface:
                sys.exit(0)


def command_line_usage() -> None:
    if len(sys.argv) < 2:
        print("Usage: python bisect_update.py <start|end|good|bad>")
        sys.exit(1)

    bisection_manager = BisectionManager()
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

    if not bisection_manager.get_current_system():
        raise ValueError("Must call start prior to good or bad")

    bisection_manager.do_bisect(test_function, cli_interface=True)


def get_is_bisection_enabled() -> bool:
    return (
        BisectionManager.get_current_subsystem() is not None
        or BisectionManager.get_current_system() is not None
    )


BisectionManager.bisection_enabled = get_is_bisection_enabled()

if __name__ == "__main__":
    command_line_usage()
