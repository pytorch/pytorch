import itertools
import pickle
import random
import signal
import string
import traceback
from enum import Enum
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    Dict,
    get_args,
    get_origin,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass
from torch._inductor.scheduler import BaseSchedulerNode
from torch.utils._config_module import _ConfigEntry, ConfigModule


def is_optional_type(type_hint) -> bool:  # type: ignore[no-untyped-def]
    origin = get_origin(type_hint)

    if origin is Union:
        args = get_args(type_hint)
        return type(None) in args

    return False


# callable types are messed up
def is_callable_type(type_hint) -> bool:  # type: ignore[no-untyped-def]
    return type_hint.__name__ == "Callable"


def is_type(type_hint, comp_type) -> bool:  # type: ignore[no-untyped-def]
    return type_hint is comp_type or get_origin(type_hint) is comp_type


class DummyPass(CustomGraphPass):
    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        """
        Implementation of the custom pass.
        """
        return None

    def uuid(self) -> Optional[Any]:
        """
        Return an ID to uniquely identify your custom pass implementation. Return None
        to skip inductor code caching entirely.
        """
        return None


TYPE_EXEMPLARS: dict[str, Any] = {
    CustomGraphPass.__name__: DummyPass(),
    torch.fx.graph.Graph.__name__: torch.fx.graph.Graph(),
    BaseSchedulerNode.__name__: BaseSchedulerNode(None),  # type: ignore[arg-type]
}


class Status(Enum):
    SKIPPED = "skipped"
    PASSED = "passed"
    FAILED_RUN_EXCEPTION = "failed_run_exception"
    FAILED_RUN_RETURN = "failed_run_return"
    FAILED_COMPILE = "failed_compile"

    def failing(self) -> bool:
        return self == Status.FAILED_RUN or self == Status.FAILED_COMPILE


class SamplingMethod(Enum):
    """
    This class handles sampling values of a type assign to configs.
    """

    TOGGLE = "TOGGLE"  # toggle to the opposite value
    RANDOM = "RANDOM"  # randomly choose an option

    @staticmethod
    def _generate_value_for_type(
        random_sample: bool, type_hint: Type[Any], default: Any
    ) -> Any:
        """this setting will use randomness too, but if there's a sensible 'toggle', it will use that"""
        if type_hint == bool:
            return random.choice([True, False]) if random_sample else not default
        elif type_hint == int:
            # NOTE initially tried to use negation of the value, but it doesn't work because most types are ints
            # when they should be natural numbers + zero. Python types to cover these values aren't super convenient.
            return random.randint(0, 1000)
        elif type_hint == float:
            return random.uniform(0, 1000)
        elif type_hint == str:
            characters = string.ascii_letters + string.digits + string.punctuation
            return "".join(
                random.choice(characters) for _ in range(random.randint(1, 20))
            )
        elif is_type(type_hint, list):
            elem_type = getattr(
                type_hint,
                "__args__",
                [type(default[0])] if len(default) else [type(None)],
            )[0]
            new_default = default[0] if len(default) > 0 else None
            return [
                SamplingMethod._generate_value_for_type(
                    random_sample, elem_type, new_default
                )
                for _ in range(random.randint(1, 3))
            ]
        elif is_type(type_hint, set):
            indexable = list(default)
            elem_type = getattr(
                type_hint,
                "__args__",
                [type(indexable[0])] if len(default) else [type(None)],
            )[0]
            new_default = indexable[0] if len(default) > 0 else None
            return {
                SamplingMethod._generate_value_for_type(
                    random_sample, elem_type, new_default
                )
                for _ in range(random.randint(1, 3))
            }
        elif is_type(type_hint, dict):
            key_type, value_type = getattr(
                type_hint,
                "__args__",
                map(type, next(iter(default.items())))
                if len(default)
                else (type(None), type(None)),
            )
            items = list(default.items())
            if len(items) > 0:
                default_key, default_val = items[0]
            else:
                default_key, default_val = None, None
            return {
                SamplingMethod._generate_value_for_type(
                    random_sample, key_type, default_key
                ): SamplingMethod._generate_value_for_type(
                    random_sample, value_type, default_val
                )
                for _ in range(random.randint(0, 3))
            }
        elif is_type(type_hint, Union):
            # do whatever is not the type of default
            try:
                assert len(type_hint.__args__) > 1
            except AttributeError as err:
                raise ValueError("Union type with no args") from err
            if random_sample:
                new_type = random.choice(type_hint.__args__)
            else:
                new_type = random.choice(
                    [t for t in type_hint.__args__ if t != type(default)]
                )
            try:
                new_default = new_type()
            except Exception:  # noqa: E722
                # if default constructor doesn't work, try None
                new_default = None

            return SamplingMethod._generate_value_for_type(
                random_sample, new_type, new_default
            )
        elif is_type(type_hint, tuple):
            args = getattr(
                type_hint,
                "__args__",
                tuple(map(type, default)),
            )
            zipped = zip(args, default)
            return tuple(
                map(  # noqa: C417
                    lambda x: SamplingMethod._generate_value_for_type(
                        random_sample, x[0], x[1]
                    ),
                    zipped,
                )
            )
        elif is_type(type_hint, Literal):
            try:
                if random_sample:
                    return random.choice(type_hint.__args__)
                else:
                    return random.choice(
                        [t for t in type_hint.__args__ if t != default]
                    )
            except AttributeError as err:
                raise ValueError("Literal type with no args") from err
        elif is_optional_type(type_hint):
            try:
                elem_type = type_hint.__args__[0]
            except AttributeError as err:
                raise ValueError("Optional type with no args") from err
            if random_sample:
                return random.choice(
                    [
                        None,
                        SamplingMethod._generate_value_for_type(
                            random_sample, elem_type, default
                        ),
                    ]
                )
            else:
                if default is None:
                    return SamplingMethod._generate_value_for_type(
                        random_sample, elem_type, None
                    )
                else:
                    return None
        elif type_hint is type(None):
            return None
        elif is_callable_type(type_hint):
            try:
                input_args, return_type = (
                    list(type_hint.__args__)[:-1],
                    list(type_hint.__args__)[-1],
                )
            except AttributeError as err:
                raise ValueError("Callable type with no args") from err

            @wraps(lambda *args, **kwargs: None)
            def dummy_function(*args, **kwargs):  # type: ignore[no-untyped-def]
                return SamplingMethod._generate_value_for_type(
                    random_sample, return_type, None
                )

            return dummy_function
        elif type_hint.__name__ in TYPE_EXEMPLARS:
            return TYPE_EXEMPLARS[type_hint.__name__]
        elif type_hint == Any:
            return 1 if not default == 1 else 2
        else:
            raise ValueError(f"Unable to process type {type_hint}. PRs welcome :)")

    @staticmethod
    def dispatch(sm: "SamplingMethod") -> Callable[[Type[Any], Any], Any]:
        if sm == SamplingMethod.RANDOM:
            return partial(SamplingMethod._generate_value_for_type, True)
        elif sm == SamplingMethod.TOGGLE:
            return partial(SamplingMethod._generate_value_for_type, False)
        else:
            raise ValueError(f"malformed sampling method: {sm}")


class Default:
    pass


DEFAULT = Default()

ComboType = Tuple[str, ...]
ResultType = Dict[ComboType, Status]
ConfigType = Dict[str, Any]
FactoryOutputType = Callable[[], bool | Tuple[Any]]
FactoryType = Callable[[], FactoryOutputType]


class ConfigFuzzer:
    """
    This tool makes it easy to search through config state-space with a minimal reproduction or test, either for debugging or just bug hunting.
    It presents a similar interface to the config bisector by taking a test_function that should either raise on Exception or return False upon failure.
    It has two entry points:
     - fuzz_with_bisect, which randomly flips configs and tries to find the minimal reproduction upon failure.
     - fuzz_n_tuple, which tries every combination of n configs. This grows quickly as a function of n, so beware.
    fuzz_with_bisect is recommended, but fuzz_n_tuple can give you peace of mind that a new config will compose with every other config.

    # Example usage:
    import torch._inductor.config as cfg

    fuzzer = ConfigFuzzer(cfg, create_simple_test_model_gpu, seed=2)

    # Test every pair of configs:
    results = fuzzer.fuzz_n_tuple(n, max_combinations=10000000)

    visualize_results(n, results)

    # Test random configs with bisection:
    ret = fuzzer.fuzz_random_with_bisect(num_attempts=10)

    # reproduce a failing config
    fuzzer.reproduce([{"triton.autotune_pointwise": ..., "coordinate_descent_tuning": ...}])

    # Known failures on inductor config:
    cpp_wrapper, triton_debug_sync_graph
    cpp_wrapper, triton_debug_sync_kernel
    cpp_wrapper, disable_cpp_codegen
    combo_kernels, benchmark_combo_kernel, profile_bandwidth, profile_bandwidth_regex

    """

    sample: Callable[[Type[Any], Any], Any]

    def __init__(
        self,
        config_module: ConfigModule,
        test_model_fn_factory: FactoryType,
        seed: int,
        default: Optional[ConfigType] = None,
        sm: SamplingMethod = SamplingMethod.TOGGLE,
        test_timeout: int = 1800,
    ):
        """
        Args:
            config_module: The module containing the configs to fuzz
            test_model_fn_factory: Function that returns a test model, which runs and returns True if successful, or the outputs if they should be compared with eager
            seed: Randomness seed.
            default: Default values for the config. Inductor has preset based on know failures.
            sm: How type value samples are generated, default TOGGLE.
            test_timeout: max time a test can take.
        """
        self.seed = seed
        self.test_timeout = test_timeout
        self.detailed_results: Dict[ComboType, Dict[str, Any]] = {}
        self.config_module = config_module
        self.test_model_fn_factory = test_model_fn_factory
        self.fields: Dict[str, _ConfigEntry] = self.config_module._config
        self.sample = SamplingMethod.dispatch(sm)

        if default is None:
            if self.config_module.__name__ == "torch._inductor.config":
                # Why are some configs disabled by default? Because if we don't the fuzzer produces uninteresting results.
                # It will always hone-in on these failures, even with the most basic model, making it useless for debugging more complex models.
                #
                # More explicit explanations are below:
                # Out of Scope: We can't fuzz, say, the cuda version because that comes from the environment and will produce a failure if not aligned with env.
                # Known Failure: Disabled due to known failure. Hopefully re-enable. Known failures are listed in the docstring of this file.
                # Required: Required for the fuzzer to operate (removing caching, etc.)
                # FSDP: flag meant for FSDP that fails in non FSDP envs. Re-enable these if you're testing FSDP.
                # Typing: disabled because the type annotation of the config isn't constrained enough to produce meaningful fuzz values. These could be improved.
                # Timing: These take too long to compile, feel free to enable.
                self.default = {
                    "force_disable_caches": True,  # Required
                    "cpp.cxx": DEFAULT,  # Out of Scope
                    "TYPE_CHECKING": DEFAULT,  # Not a config
                    "max_autotune_pointwise": DEFAULT,  # Timing
                    "max_autotune_gemm": DEFAULT,  # Timing
                    "max_autotune_gemm_backends": DEFAULT,  # Timing
                    "max_autotune_conv_backends": DEFAULT,  # Timing
                    "max_autotune_gemm_search_space": DEFAULT,  # Timing
                    "max_autotune_subproc_result_timeout_seconds": DEFAULT,  # Timing
                    "max_autotune_subproc_graceful_timeout_seconds": DEFAULT,  # Timing
                    "max_autotune_subproc_terminate_timeout_seconds": DEFAULT,  # Timing
                    "autoheuristic_collect": DEFAULT,  # Typing
                    "autoheuristic_use": DEFAULT,  # Typing
                    "aot_inductor.presets": DEFAULT,  # Typing
                    "cuda.arch": DEFAULT,  # Out of Scope
                    "cuda.version": DEFAULT,  # Out of Scope
                    "cuda.cutlass_dir": DEFAULT,  # Out of Scope
                    "cuda.cuda_cxx": DEFAULT,  # Out of Scope
                    "rocm.arch": DEFAULT,  # Out of Scope
                    "rocm.ck_supported_arch": DEFAULT,  # Out of Scope
                    "rocm.ck_dir": DEFAULT,  # Out of Scope
                    "rocm.rocm_home": DEFAULT,  # Out of Scope
                    "check_stack_no_cycles_TESTING_ONLY": DEFAULT,  # Testing
                    "reorder_for_compute_comm_overlap": DEFAULT,  # FSDP
                    "enabled_metric_tables": DEFAULT,  # Typing
                    "triton.debug_sync_graph": DEFAULT,  # Known Failure
                    "triton.debug_sync_kernel": DEFAULT,  # Known Failure
                    "profile_bandwidth_regex": DEFAULT,  # Known Failure
                    "disable_cpp_codegen": DEFAULT,  # Known Failure
                }
            else:
                raise ValueError("No default passed to ConfigFuzzer.")
        else:
            self.default = default

    def __repr__(self) -> str:
        return (
            f"ConfigFuzzer(config_module={self.config_module}, "
            f"test_model_fn_factor={self.test_model_fn_factory}, seed={self.seed}, default={self.default})"
        )

    def _set_config(self, field_name: str, value: Any) -> None:
        """Set a config value in the module."""
        setattr(self.config_module, field_name, value)

    def _reset_configs(self) -> None:
        """Reset all configs to their default values."""
        for field_name, field_obj in self.fields.items():
            self._set_config(field_name, field_obj.default)

    def _set_status(
        self, results: ResultType, combo: ComboType, status: Status
    ) -> None:
        combo = tuple(sorted(combo))
        results[combo] = status

    def _lookup_status(self, results: ResultType, combo: ComboType) -> Optional[Status]:
        combo = tuple(sorted(combo))
        return results[combo] if combo in results else None

    def new_config(self) -> ConfigType:
        """creates a new config from the default"""
        ret = {
            name: val if val != DEFAULT else self.fields[name].default
            for name, val in self.default.items()
        }
        return ret

    def _combo_run_common(self, results: ResultType, combo: ComboType) -> None:
        print(combo)
        if self._lookup_status(results, combo):
            # we already processed this config
            return

        config = self.new_config()

        skip = False
        for field_name in combo:
            if field_name in config:
                # don't break here because we need to build the config dict
                skip = True
            if field_name.startswith("_"):
                skip = True
            field = self.fields[field_name]
            value = self.sample(field.value_type, field.default)
            config[field_name] = value
        if skip:
            self._set_status(results, combo, Status.SKIPPED)
            return

        self.test_config(results, config)

    def reproduce(self, configs: List[ConfigType]) -> ResultType:
        """entrypoint to reproduce any failure"""
        results: ResultType = {}
        for conf in configs:
            print(f"Starting repro of {conf}")
            new_config = self.new_config()
            new_config.update(conf)
            self.test_config(results, new_config)
        return results

    def fuzz_n_tuple(self, n: int, max_combinations: int = 1000) -> ResultType:
        """
        Test every combination of n configs.

        returns a dict of this shape: {(config-1, config-2... config-n): status}
        """
        results: ResultType = {}
        print(f"Starting {n}-tuple testing with seed {self.seed}")
        random.seed(self.seed)

        for combo in itertools.combinations(self.fields, n):
            self._combo_run_common(results, combo)
            max_combinations -= 1
            if max_combinations <= 0:
                print("Reached maximum combinations limit")
                break

        return results

    def save_state(self, filename: str = "fuzzer_state.pkl") -> None:
        """Save the current fuzzer state to a file"""
        with open(filename, "wb") as f:
            pickle.dump(
                {"results": self.results, "detailed_results": self.detailed_results}, f
            )

    def load_state(self, filename: str = "fuzzer_state.pkl") -> None:
        """Load fuzzer state from a file"""
        with open(filename, "rb") as f:
            state = pickle.load(f)
            self.results = state["results"]
            self.detailed_results = state.get("detailed_results", {})

    def timeout_handler(self, signum, frame):
        raise TimeoutError("Test execution timed out")

    def test_config(self, results: ResultType, config: ConfigType) -> Status:
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.test_timeout)
        """
        Tests a config
        """
        print(f"Testing config {config}")
        config_tuple = tuple(config.keys())
        if ret := self._lookup_status(results, config_tuple):
            return ret
        torch._dynamo.reset()
        test_model_fn = self.test_model_fn_factory()

        def set_config():
            for name, value in config.items():
                self._set_config(name, value)

        def compile_with_options(test_fn):
            if self.config_module.__name__ == "torch._inductor.config":
                return torch.compile(options=config)(test_model_fn)
            self._reset_configs()
            set_config()
            comp = torch.compile()(test_model_fn)

        def run_eager(test_fn):
            if self.config_module.__name__ == "torch._inductor.config":
                # we didn't set config earlier for compile in inductor
                set_config()
            return test_fn()

        def print_config():
            for field, value in config.items():
                print(f"{field} = {value}")

        def handle_return(message, return_status, print_traceback):
            print(f"{message} with config combination:")
            print_config(config)
            if print_traceback:
                traceback.print_exc()
            self._set_status(results, config_tuple, return_status)
            return return_status

        # try compilation
        try:
            comp = compile_with_options(test_model_fn)
        except Exception as exc:  # noqa: E722
            error_info = {
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "config": config.copy(),
            }
            self.detailed_results[config_tuple] = error_info
            return handle_return("Exception compiling", Status.FAILED_COMPILE, True)

        # try running compiled
        try:
            success = comp()
        except Exception:  # noqa: E722
            return handle_return("Exception running", Status.FAILED_RUN_EXCEPTION, True)

        # bool return value means don't compare with eager
        if type(success) is bool:
            if not success:
                return handle_return(
                    "Failure returned bool", Status.FAILED_RUN_RETURN, False
                )
            else:
                ret = Status.PASSED
                self._set_status(results, config_tuple, ret)
                return ret
        # try running in eager
        elif type(success) is tuple:
            try:
                eager_results = test_model_fn()
            except Exception:  # noqa: E722
                return handle_return(
                    "Eager exception", Status.FAILED_RUN_EXCEPTION, True
                )
            for er, cr in zip(eager_results, success):
                if not torch.isclose(er, cr):
                    return handle_return(
                        "Results don't match eager", Status.FAILED_RUN_RETURN, False
                    )
            ret = Status.PASSED
            self._set_status(results, config_tuple, ret)
            return ret
        else:
            raise ValueError(
                f"Unable to process return type of test function: {type(success)}"
            )

    def fuzz_with_bisect(
        self, num_attempts: int = 100, p: float = 0.5
    ) -> List[ConfigType]:
        """
        Test configs and bisect to minimal failing configuration.
        """
        print(f"Starting random testing with bisection, seed {self.seed}, and p {p}")
        random.seed(self.seed)
        self._reset_configs()
        results: ResultType = {}
        ret: List[ConfigType] = []

        for attempt in range(num_attempts):
            print(f"Random attempt {attempt + 1}/{num_attempts}")

            config = self.new_config()

            for field_name, config_entry in self.fields.items():
                if (
                    field_name not in config
                    and not field_name.startswith("_")
                    and random.random() < p
                ):
                    value = self.sample(config_entry.value_type, config_entry.default)
                    config[field_name] = value

            status = self.test_config(results, config)
            if status not in {Status.PASSED, Status.SKIPPED}:
                if minimal_failing_config := self._bisect_failing_config(
                    results, config
                ):
                    print(f"Minimum failing config: {minimal_failing_config}")
                    ret.append(minimal_failing_config)

        return ret

    def _bisect_failing_config(
        self, results: ResultType, failing_config: ConfigType
    ) -> Optional[ConfigType]:
        return self._bisect_failing_config_helper(results, list(failing_config.items()))

    def _bisect_failing_config_helper(
        self, results: ResultType, failing_config: List[Tuple[str, Any]]
    ) -> Optional[ConfigType]:
        """
        Bisect a failing configuration to find minimal set of configs that cause failure.

        Splits it into halves, then fourths, then tries dropping configs one-by-one.
        """
        print(f"bisecting config: {failing_config}")

        if not failing_config:
            return None

        def test(x: List[Tuple[str, Any]]) -> Status:
            d = dict(x)
            result = self.test_config(results, d)
            return result

        if len(failing_config) <= 1:
            return dict(failing_config) if test(failing_config).failing() else None

        # Shuffling helps the worst case
        random.shuffle(failing_config)

        mid = len(failing_config) // 2
        first_half = failing_config[:mid]
        second_half = failing_config[mid:]
        if test(first_half).failing():
            return self._bisect_failing_config_helper(results, first_half)
        if test(second_half).failing():
            return self._bisect_failing_config_helper(results, second_half)

        if len(failing_config) >= 8:
            low = len(failing_config) // 4
            high = mid + low
            quart1 = failing_config[low:]
            if test(quart1).failing():
                return self._bisect_failing_config_helper(results, quart1)
            quart2 = failing_config[:low] + second_half
            if test(quart2).failing():
                return self._bisect_failing_config_helper(results, quart2)
            quart3 = first_half + failing_config[:high]
            if test(quart3).failing():
                return self._bisect_failing_config_helper(results, quart3)
            quart4 = failing_config[high:]
            if test(quart4).failing():
                return self._bisect_failing_config_helper(results, quart4)
        # try dropping one value at a time
        for i in range(len(failing_config)):
            new_list = [x for j, x in enumerate(failing_config) if j != i]
            if test(new_list).failing():
                return self._bisect_failing_config_helper(results, new_list)
        # we have the minimal set
        return dict(failing_config)


def visualize_results(
    n: int, status: ResultType, filename: str = "results.html"
) -> None:
    # TODO Support more dimensions
    assert n == 2
    assert len(status) > 0

    # Create a dictionary for quick lookup of status
    input_set = set({})
    for key in status.keys():
        input_set.add(key[0])
        input_set.add(key[1])
    input_list = sorted(input_set)

    # Start the HTML content
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title> Fuzzer Visualization</title>
        <style>
            table {
                border-collapse: collapse;
                width: 50%;
                margin: 20px auto;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }
            th {
                background-color: #f2f2f2;
            }
            .skipped {
                background-color: yellow;
            }
            .passed {
                background-color: green;
                color: white;
            }
            .failed {
                background-color: red;
                color: white;
            }
        </style>
    </head>
    <body>
        <h2 style="text-align: center;">Fuzzer Visualization</h2>
        <table>
        <thead>
    """

    html_content += "<tr><th>\\</th>"
    for i, col_name in enumerate(input_list):
        col = "<br>".join(col_name)
        html_content += f"<th>{col}</th>"
    html_content += "</tr></thead><tbody>"

    # Add table rows
    for i, row_name in enumerate(input_list):
        html_content += f"<tr><th>{row_name}</th>"
        for j, col_name in enumerate(input_list):
            # Determine the status class for the cell
            status_class = ""
            status_val = ""
            if (row_name, col_name) in status:
                status_enum = status[(row_name, col_name)]
                if status_enum == Status.SKIPPED:
                    status_class = "skipped"
                    status_val = "-"
                elif status_enum == Status.PASSED:
                    status_class = "passed"
                    status_val = "O"
                elif status_enum == Status.FAILED_RUN_EXCEPTION:
                    status_class = "failed"
                    status_val = "E"
                elif status_enum == Status.FAILED_RUN_RETURN:
                    status_class = "failed"
                    status_val = "R"
                elif status_enum == Status.FAILED_COMPILE:
                    status_class = "failed"
                    status_val = "C"

            html_content += f'<td class="{status_class}">{status_val}</td>'
        html_content += "</tr>"

    html_content += """
        </tbody>
        </table>
    </body>
    </html>
    """

    with open(filename, "w") as file:
        file.write(html_content)

    print(f"HTML file '{filename}' has been generated.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Config Fuzzer CLI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num_attempts", type=int, default=100, help="Number of attempts for fuzzing"
    )
    parser.add_argument(
        "--n", type=int, default=2, help="Number of configurations to combine"
    )
    parser.add_argument(
        "--method",
        choices=["n_tuple", "random"],
        default="n_tuple",
        help="Fuzzing method",
    )
    parser.add_argument(
        "--timeout", type=int, default=60, help="Test timeout in seconds"
    )
    parser.add_argument("--save_state", type=str, help="Save state to file")
    parser.add_argument("--load_state", type=str, help="Load state from file")
    parser.add_argument(
        "--gpu", type=bool, default=True, help="Whether to test the GPU or not"
    )

    def create_simple_test_model_cpu() -> FactoryOutputType:
        def test_fn() -> bool:
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
            )

            x = torch.randn(32, 10)
            y = model(x)
            return True

        return test_fn

    def create_simple_test_model_gpu() -> FactoryOutputType:
        batch_size = 32
        seq_length = 50
        hidden_size = 768

        def test_fn() -> bool:
            inp = torch.randn(batch_size, seq_length, hidden_size, device="cuda")
            weight = torch.randn(hidden_size, hidden_size, device="cuda")
            matmul_output = inp @ weight
            final_output = torch.nn.LayerNorm(hidden_size, device="cuda")(matmul_output)
            return True

        return test_fn

    args = parser.parse_args()

    fuzzer = ConfigFuzzer(
        config_module=torch._inductor.config,
        test_model_fn_factory=create_simple_test_model_gpu
        if args.gpu
        else create_simple_test_model_cpu,
        seed=args.seed,
        test_timeout=args.timeout,
    )

    if args.load_state:
        fuzzer.load_state(args.load_state)

    if args.method == "n_tuple":
        results = fuzzer.fuzz_n_tuple(n=args.n, max_combinations=args.num_attempts)
    else:
        results = fuzzer.fuzz_with_bisect(num_attempts=args.num_attempts)

    if args.save_state:
        fuzzer.save_state(args.save_state)
