import importlib
import itertools
import logging
import pickle
import random
import signal
import string
import sys
import traceback
from collections.abc import KeysView, Sequence
from enum import Enum
from functools import partial, wraps
from types import FrameType
from typing import (
    Any,
    Callable,
    get_args,
    get_origin,
    Literal,
    Optional,
    TypeVar,
    Union,
)

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass
from torch._inductor.scheduler import BaseSchedulerNode
from torch.utils._config_module import _ConfigEntry, ConfigModule
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


def is_type(type_hint, comp_type) -> bool:  # type: ignore[no-untyped-def]
    """
    Determines if type_hint is comp_type. There are some type annotations that this doesn't work for.
    I think it's because some Type annotations are Type Objects and some are Special Forms, but not sure.
    There's definite room for improvement to make this more general for someone who deeply understands
    Python types.
    """
    return type_hint is comp_type or get_origin(type_hint) is comp_type


def is_optional_type(type_hint) -> bool:  # type: ignore[no-untyped-def]
    """
    Special case of is_type.
    """
    origin = get_origin(type_hint)

    if origin is Union:
        args = get_args(type_hint)
        return type(None) in args

    return False


def is_callable_type(type_hint) -> bool:  # type: ignore[no-untyped-def]
    """
    Special Case of is_type.
    """
    return type_hint.__name__ == "Callable"


class DummyPass(CustomGraphPass):
    """
    A Dummy pass to be used by ConfigFuzzer
    """

    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        return None

    def uuid(self) -> Optional[Any]:
        return None


T = TypeVar("T")


class TypeExemplars:
    """
    This class returns examples of a Type, given its class name.
    """

    TYPE_EXEMPLARS: dict[str, Any] = {
        CustomGraphPass.__name__: DummyPass(),
        torch.fx.graph.Graph.__name__: torch.fx.graph.Graph(),
        BaseSchedulerNode.__name__: BaseSchedulerNode(None),  # type: ignore[arg-type]
    }

    @staticmethod
    def example(t: type[T]) -> Optional[T]:
        """
        Return an example of a class.
        """
        return TypeExemplars.TYPE_EXEMPLARS.get(t.__name__, None)

    @staticmethod
    def contains(t: type[T]) -> bool:
        return t.__name__ in TypeExemplars.TYPE_EXEMPLARS


def check_halide_import() -> bool:
    """checks if we have halide available"""
    try:
        importlib.import_module("halide")
        return True
    except ModuleNotFoundError:
        return False


if check_halide_import():
    CUDA_BACKEND = ["triton", "halide"]
else:
    CUDA_BACKEND = ["triton"]


class Status(Enum):
    """
    The Status return value enum for Config Fuzzer
    """

    # ConfigFuzzer skipped the test
    SKIPPED = "skipped"
    # ConfigFuzzer compiled and ran the test and function it passed.
    PASSED = "passed"
    # ConfigFuzzer failed to compile the test function
    FAILED_COMPILE = "failed_compile"
    # ConfigFuzzer compiled the test function and running it raised an exception
    FAILED_RUN_COMPILE_EXCEPTION = "failed_run_compile_exception"
    # ConfigFuzzer ran eager and it raised an exception
    FAILED_RUN_EAGER_EXCEPTION = "failed_run_eager_exception"
    # ConfigFuzzer compiled the test function, but the return value indicated that the compiled value didn't match the
    # value from eager (or however else you set up the comparison in the test function)
    FAILED_RUN_RETURN = "failed_run_return"

    def failing(self) -> bool:
        """
        Convenience method to check whether these status represent failure.
        """
        return (
            self == Status.FAILED_COMPILE
            or self == Status.FAILED_RUN_EAGER_EXCEPTION
            or self == Status.FAILED_RUN_COMPILE_EXCEPTION
            or self == Status.FAILED_RUN_RETURN
        )


# Sometime the types of configs aren't expressive enough to be captured by python type system, so the options can be
# manually specified here:
# TODO this needs to be indexed to the module, like inductor or dynamo, for name collisions
TYPE_OVERRIDES: dict[str, list[Any]] = {
    "cuda_backend": CUDA_BACKEND,
    "post_grad_fusion_options": [
        {
            "batch_linear_post_grad": {
                "shape_broadcast_batch_linear": True,
                "fuse_nodes_with_same_users": True,
            },
            "batch_aten_mul": {"fuse_nodes_with_same_parent": False},
            "batch_aten_sigmoid": {"fuse_nodes_with_same_parent": True},
            "batch_aten_add": {"fuse_nodes_with_same_parent": True},
            "normalization_aten_pass": {},
            "unbind_stack_aten_pass": {},
        },
        {
            "batch_aten_add": {},
            "batch_aten_mul": {},
            "batch_aten_sub": {},
            "batch_aten_div": {},
            "group_linear": {"require_fbgemm": True},
        },
    ],
    "autoheuristic_collect": ["pad_mm", "mixed_mm"],
    "autoheuristic_use": ["pad_mm", "mixed_mm"],
    "traceable_tensor_subclasses": [OrderedSet()],
    "nontraceable_tensor_subclasses": [OrderedSet()],
}
SamplingType = Callable[[str, type[Any], Any], Any]


class SamplingMethod(Enum):
    """
    This class handles the process of assigning concrete values to type annotations. So a type annotation of
    ```python
    foo: Optional[int] = None
    ```
    Will be assigned an int if the dispatch function gets TOGGLE, or a 50/50 split between an int and None if it gets
    RANDOM.
    """

    TOGGLE = "TOGGLE"  # toggle to the opposite value
    RANDOM = "RANDOM"  # randomly choose an option

    @staticmethod
    def _generate_value_for_type(
        random_sample: bool, field_name: str, type_hint: type[Any], default: Any
    ) -> Any:
        """
        Generates a value of a type based on the setting.
        """
        # look for name in type overrides
        if field_name in TYPE_OVERRIDES:
            return random.choice(TYPE_OVERRIDES[field_name])

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
                [type(default[0])] if default and len(default) else [type(None)],
            )[0]
            new_default = default[0] if default and len(default) > 0 else None
            return [
                SamplingMethod._generate_value_for_type(
                    random_sample, field_name, elem_type, new_default
                )
                for _ in range(random.randint(1, 3))
            ]
        elif is_type(type_hint, set):  # noqa: set_linter
            indexable = list(default)
            elem_type = getattr(
                type_hint,
                "__args__",
                [type(indexable[0])] if default and len(default) else [type(None)],
            )[0]
            new_default = indexable[0] if default and len(default) > 0 else None
            return {  # noqa: set_linter
                SamplingMethod._generate_value_for_type(
                    random_sample, field_name, elem_type, new_default
                )
                for _ in range(random.randint(1, 3))
            }
        elif is_type(type_hint, OrderedSet):
            indexable = list(default)
            elem_type = getattr(
                type_hint,
                "__args__",
                [type(indexable[0])] if default and len(default) else [type(None)],
            )[0]
            new_default = indexable[0] if default and len(default) > 0 else None
            return OrderedSet(
                [
                    SamplingMethod._generate_value_for_type(
                        random_sample, field_name, elem_type, new_default
                    )
                    for _ in range(random.randint(1, 3))
                ]
            )
        elif is_type(type_hint, dict):
            key_type, value_type = getattr(
                type_hint,
                "__args__",
                map(type, next(iter(default.items())))
                if (default is not None and len(default))
                else (type(None), type(None)),
            )
            if default is not None and len(default.items()) > 0:
                default_key, default_val = next(iter(default.items()))
            else:
                default_key, default_val = None, None
            return {
                SamplingMethod._generate_value_for_type(
                    random_sample, field_name, key_type, default_key
                ): SamplingMethod._generate_value_for_type(
                    random_sample, field_name, value_type, default_val
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
                random_sample, field_name, new_type, new_default
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
                        random_sample, field_name, x[0], x[1]
                    ),
                    zipped,
                )
            )
        elif is_type(type_hint, Literal):
            try:
                if random_sample:
                    return random.choice(type_hint.__args__)
                else:
                    choices = [t for t in type_hint.__args__ if t != default]
                    if choices:
                        return random.choice(choices)
                    else:
                        return default
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
                            random_sample, field_name, elem_type, default
                        ),
                    ]
                )
            else:
                if default is None:
                    return SamplingMethod._generate_value_for_type(
                        random_sample, field_name, elem_type, None
                    )
                else:
                    return None
        elif type_hint is type(None):
            return None
        elif is_callable_type(type_hint):
            try:
                return_type = list(type_hint.__args__)[-1]
            except AttributeError as err:
                raise ValueError("Callable type with no args") from err

            @wraps(lambda *args, **kwargs: None)
            def dummy_function(*args, **kwargs):  # type: ignore[no-untyped-def]
                return SamplingMethod._generate_value_for_type(
                    random_sample, field_name, return_type, None
                )

            return dummy_function
        elif type_hint == torch._ops.OpOverload:
            return torch.ops.aten.add.default
        elif TypeExemplars.contains(type_hint):
            return TypeExemplars.example(type_hint)
        elif type_hint == Any:
            return 1 if not default == 1 else 2
        else:
            raise ValueError(f"Unable to process type {type_hint}. PRs welcome :)")

    @staticmethod
    def dispatch(sm: "SamplingMethod") -> SamplingType:
        """
        Returns a function that will generate values from a type, based on the SamplingMethod passed in.
        """
        if sm == SamplingMethod.RANDOM:
            return partial(SamplingMethod._generate_value_for_type, True)
        elif sm == SamplingMethod.TOGGLE:
            return partial(SamplingMethod._generate_value_for_type, False)
        else:
            raise ValueError(f"malformed sampling method: {sm}")


class Default:
    """
    Singleton default object that will cause the ConfigFuzzer to always use the default value set in the config.
    """


DEFAULT = Default()

# The combination of config settings being set (based on their strings)
ComboType = tuple[str, ...]


class ResultType:
    """
    The mapping of the combo strings to the result status after running the config fuzzer.
    """

    _vals: dict[ComboType, Status]

    def __repr__(self) -> str:
        return f"ResultType[{self._vals}]"

    def __init__(self) -> None:
        self._vals = {}

    def __len__(self) -> int:
        return len(self._vals)

    def num_ran(self) -> int:
        """
        Returns how many combos actually ran (weren't skipped).
        """
        ret = len(self._vals)
        for status in self._vals.values():
            if status == Status.SKIPPED:
                ret -= 1
        return ret

    def set(self, combo: ComboType, status: Status) -> None:
        combo = tuple(sorted(combo))
        self._vals[combo] = status

    def lookup(self, combo: ComboType) -> Optional[Status]:
        combo = tuple(sorted(combo))
        return self._vals.get(combo, None)

    def keys(self) -> KeysView[ComboType]:
        return self._vals.keys()


# Type that maps config strings to their default value
ConfigType = dict[str, Any]
# Callable that returns a bool
FactoryOutputType = Callable[[], bool]
# input function factory
FactoryType = Callable[[], FactoryOutputType]

# Why are some configs disabled by default? Because if we don't the fuzzer produces uninteresting results.
# It will always hone-in on these failures, even with the most basic model, making it useless for
#   debugging more complex models.
#
# More explicit explanations are below:
# Out of Scope: We can't fuzz, say, the cuda version because that comes from the environment and will
#   produce a failure if not aligned with env.
# Known Failure: Disabled due to known failure. Hopefully re-enable. Known failures are listed in the
#   docstring of this file.
# Required: Required for the fuzzer to operate (removing caching, etc.)
# FSDP: Flag meant for FSDP that fails in non FSDP envs. Re-enable these if you're testing FSDP.
# Typing: disabled because the type annotation of the config isn't constrained enough to produce
#   meaningful fuzz values. These could be improved.
# Timing: These take too long to compile, feel free to enable.
MODULE_DEFAULTS: dict[str, ConfigType] = {
    "torch._inductor.config": {
        "force_disable_caches": True,  # Required
        "cpp.cxx": DEFAULT,  # Out of Scope
        "TYPE_CHECKING": DEFAULT,  # Not a config
        "max_autotune_pointwise": DEFAULT,  # Timing
        "max_autotune_gemm": DEFAULT,  # Timing, re-enable when autotune speed improvements merged.
        "max_autotune_gemm_backends": DEFAULT,  # Timing
        "max_autotune_conv_backends": DEFAULT,  # Timing
        "max_autotune_gemm_search_space": DEFAULT,  # Timing
        "max_autotune_subproc_result_timeout_seconds": DEFAULT,  # Timing
        "max_autotune_subproc_graceful_timeout_seconds": DEFAULT,  # Timing
        "max_autotune_subproc_terminate_timeout_seconds": DEFAULT,  # Timing
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
        "sleep_sec_TESTING_ONLY": DEFAULT,  # Testing
        "triton.inject_relu_bug_TESTING_ONLY": DEFAULT,  # Testing
        "reorder_for_compute_comm_overlap": DEFAULT,  # FSDP
        "enabled_metric_tables": DEFAULT,  # Typing
        "triton.debug_sync_graph": DEFAULT,  # Known Failure
        "triton.debug_sync_kernel": DEFAULT,  # Known Failure
        "profile_bandwidth_regex": DEFAULT,  # Known Failure
        "disable_cpp_codegen": DEFAULT,  # Known Failure
        "trace.save_real_tensors": DEFAULT,  # Known Failure
        "pre_grad_fusion_options": DEFAULT,  # Typing
        "external_matmul": DEFAULT,  # Typing, need to add this to type overrides or type exemplars.
        "test_configs.autotune_choice_name_regex": DEFAULT,  # Typing
        "test_configs.autotune_choice_desc_regex": DEFAULT,  # Typing
        "cpp.enable_floating_point_contract_flag": DEFAULT,  # Typing
        "post_grad_custom_pre_pass": DEFAULT,  # Typing
        "post_grad_custom_post_pass": DEFAULT,  # Typing
        "reorder_for_compute_comm_overlap_passes": DEFAULT,  # Typing
        "joint_custom_post_pass": DEFAULT,  # Typing
        "joint_custom_pre_pass": DEFAULT,  # Typing
        "pre_grad_custom_pass": DEFAULT,  # Typing
    },
    "torch._dynamo.config": {
        "traceable_tensor_subclasses": DEFAULT,  # Typing
        "nontraceable_tensor_subclasses": DEFAULT,  # Typing
        "compiled_autograd_kwargs_override": DEFAULT,  # Typing
        "fail_on_recompile_limit_hit": DEFAULT,  # fails in combo with suppress_errors
        "suppress_errors": DEFAULT,
    },
}


class ConfigFuzzer:
    """
    This tool makes it easy to search through config state-space with a minimal reproduction or test, either for
      debugging or just bug hunting.
    It has two entry points:
     - bisect, which randomly flips configs and tries to find the minimal reproduction upon failure.
     - fuzz_n_tuple, which tries every combination of n configs. This grows quickly as a function of n, so beware.
    bisect is recommended, but fuzz_n_tuple can give you peace of mind that a new config will compose with
      every other config.

    The main interface is a function factory that will return Callables to be torch.compiled. This function factory
      should return a test function when it's called. Said test function returns a boolean, which determines whether
      the ConfigFuzzer considers it a successful run or not. Throwing an exception from within the function will be
      considered a failure as well.

    # Example usage:

    ```python
    import torch._inductor.config as cfg


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


    fuzzer = ConfigFuzzer(cfg, create_simple_test_model_gpu, seed=2)

    # Test every pair of configs:
    results = fuzzer.fuzz_n_tuple(n, max_combinations=10000000)

    visualize_results(n, results)

    # Test random configs with bisection:
    ret = fuzzer.bisect(num_attempts=10)

    # reproduce a failing config
    fuzzer.reproduce(
        [{"triton.autotune_pointwise": ..., "coordinate_descent_tuning": ...}]
    )
    ```

    The list of known failures on inductor config are:
    cpp_wrapper, triton_debug_sync_graph
    cpp_wrapper, triton_debug_sync_kernel
    cpp_wrapper, disable_cpp_codegen
    combo_kernels, benchmark_combo_kernel, profile_bandwidth, profile_bandwidth_regex
    trace.enabled, trace.save_real_tensors
    """

    sample: SamplingType
    default: ConfigType

    def __init__(
        self,
        config_module: ConfigModule,
        test_model_fn_factory: FactoryType,
        seed: int,
        default: Optional[ConfigType] = None,
        sm: SamplingMethod = SamplingMethod.TOGGLE,
        test_timeout: int = 3600,
    ):
        """
        Args:
            config_module: The module containing the configs to fuzz
            test_model_fn_factory: Function that returns a test model, which runs and returns True if successful, or
              the outputs if they should be compared with eager
            seed: Randomness seed.
            default: Default values for the config. Inductor has preset based on know failures.
            sm: How type value samples are generated, default TOGGLE.
            test_timeout: max time a test can take.
        """
        if sys.version_info < (3, 10):
            log.error("Only python 3.10 and later supported")
            return
        self.seed = seed
        self.test_timeout = test_timeout
        self.detailed_results: dict[ComboType, dict[str, Any]] = {}
        self.config_module = config_module
        self.test_model_fn_factory = test_model_fn_factory
        self.fields: dict[str, _ConfigEntry] = self.config_module._config
        self.sample = SamplingMethod.dispatch(sm)

        if default is None:
            if self.config_module.__name__ in MODULE_DEFAULTS:
                self.default = MODULE_DEFAULTS[self.config_module.__name__]
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

    def new_config(self) -> ConfigType:
        """creates a new config from the default"""
        ret = {
            name: val if val != DEFAULT else self.fields[name].default
            for name, val in self.default.items()
        }
        return ret

    def reproduce(self, configs: Sequence[ConfigType]) -> ResultType:
        """entrypoint to reproduce any failure"""
        results = ResultType()
        for conf in configs:
            self._reproduce_single_helper(conf, results)
        return results

    def _reproduce_single_helper(self, conf: ConfigType, results: ResultType) -> None:
        print(f"Starting repro of {conf}")
        new_config = self.new_config()
        new_config.update(conf)
        self.test_config(results, new_config)
        print(f"Status of {conf}:\n{results.lookup(tuple(conf.keys()))}")

    def reproduce_single(self, config: ConfigType) -> ResultType:
        results = ResultType()
        self._reproduce_single_helper(config, results)
        return results

    def _fuzz_helper(self, results: ResultType, combo: ComboType) -> Status:
        print(combo)
        if st := results.lookup(combo):
            # we already processed this config
            return st

        config = self.new_config()

        skip = False
        for field_name in combo:
            if field_name in config:
                # don't break here because we need to build the config dict
                skip = True
            if field_name.startswith("_"):
                skip = True
            field = self.fields[field_name]
            value = self.sample(field_name, field.value_type, field.default)
            config[field_name] = value
        if skip:
            results.set(combo, Status.SKIPPED)
            return Status.SKIPPED

        return self.test_config(results, config)

    def fuzz_n_tuple(self, n: int, max_combinations: int = 1000) -> ResultType:
        """
        Test every combination of n configs.

        returns a dict of this shape: {(config-1, config-2... config-n): status}
        """
        results = ResultType()
        print(f"Starting {n}-tuple testing with seed {self.seed}")
        random.seed(self.seed)

        for combo in itertools.combinations(self.fields, n):
            st = self._fuzz_helper(results, combo)
            if st != Status.SKIPPED:
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

    def timeout_handler(self, signum: int, frame: Optional[FrameType]) -> None:
        raise TimeoutError("Test execution timed out")

    def test_config(self, results: ResultType, config: ConfigType) -> Status:
        """
        Tests a config by calling the function produced by the factory function.
        """
        original_handler = signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.test_timeout)
        print(f"Testing config {config}")
        config_tuple = tuple(config.keys())
        if ret := results.lookup(config_tuple):
            signal.signal(signal.SIGALRM, original_handler)
            return ret

        def print_config() -> None:
            for field, value in config.items():
                print(f"{field} = {value}")

        def get_error_info(exc: Exception) -> dict[str, Any]:
            return {
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "config": config.copy(),
            }

        def handle_return(
            message: str,
            return_status: Status,
            print_traceback: bool,
            exc: Optional[Exception],
        ) -> Status:
            signal.signal(signal.SIGALRM, original_handler)
            print(f"{message} with config combination:")
            print_config()
            if exc:
                self.detailed_results[config_tuple] = get_error_info(exc)
            if print_traceback:
                traceback.print_exc()
            results.set(config_tuple, return_status)
            return return_status

        # reset config
        torch._dynamo.reset()
        self._reset_configs()
        for name, value in config.items():
            self._set_config(name, value)

        # try running eager
        test_model_fn = self.test_model_fn_factory()
        try:
            test_model_fn()
        except Exception as exc:  # noqa: E722
            return handle_return(
                "Eager exception", Status.FAILED_RUN_EAGER_EXCEPTION, True, exc
            )

        # try compilation
        try:
            test_model_fn2 = self.test_model_fn_factory()
            comp = torch.compile(test_model_fn2, backend="inductor")
        except Exception as exc:  # noqa: E722
            return handle_return(
                "Exception compiling", Status.FAILED_COMPILE, True, exc
            )

        # try running compiled
        try:
            compile_result = comp()
        except Exception as exc:  # noqa: E722
            return handle_return(
                "Exception running compiled",
                Status.FAILED_RUN_COMPILE_EXCEPTION,
                True,
                exc,
            )

        # bool return value means don't compare with eager
        if not compile_result:
            return handle_return(
                "Function returned False", Status.FAILED_RUN_RETURN, False, None
            )
        else:
            return handle_return("Function succeeded", Status.PASSED, False, None)

    def bisect(self, num_attempts: int = 100, p: float = 0.5) -> list[ConfigType]:
        """
        Test configs and bisect to minimal failing configuration.
        """
        print(f"Starting random testing with bisection, seed {self.seed}, and p {p}")
        random.seed(self.seed)
        self._reset_configs()
        results = ResultType()
        ret: list[ConfigType] = []

        for attempt in range(num_attempts):
            print(f"Random attempt {attempt + 1}/{num_attempts}")

            config = self.new_config()

            for field_name, config_entry in self.fields.items():
                if (
                    field_name not in config
                    and not field_name.startswith("_")
                    and "TESTING_ONLY" not in field_name
                    and random.random() < p
                ):
                    value = self.sample(
                        field_name, config_entry.value_type, config_entry.default
                    )
                    config[field_name] = value

            status = self.test_config(results, config)
            if status not in OrderedSet([Status.PASSED, Status.SKIPPED]):
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
        self, results: ResultType, failing_config: list[tuple[str, Any]]
    ) -> Optional[ConfigType]:
        """
        Bisect a failing configuration to find minimal set of configs that cause failure.

        Splits it into halves, then fourths, then tries dropping configs one-by-one.
        """
        print(f"bisecting config: {failing_config}")

        if not failing_config:
            return None

        def test(x: list[tuple[str, Any]]) -> Status:
            d = dict(x)
            result = self.test_config(results, d)
            return result

        if len(failing_config) <= 1:
            return dict(failing_config) if test(failing_config).failing() else None

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
    n: int, results: ResultType, filename: str = "results.html"
) -> None:
    """
    Creates an HTML document representing the results of running the fuzzer with fuzz_n_tuple, with n = 2.
    """
    # TODO support more dimensions
    assert n == 2
    assert len(results) > 0

    input_set: OrderedSet[str] = OrderedSet({})
    for key in results.keys():
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
    for col_name in input_list:
        col = "<br>".join(col_name)
        html_content += f"<th>{col}</th>"
    html_content += "</tr></thead><tbody>"

    # Add table rows
    for row_name in input_list:
        html_content += f"<tr><th>{row_name}</th>"
        for col_name in input_list:
            # Determine the status class for the cell
            status_enum = results.lookup((row_name, col_name))
            status_class = ""
            status_val = ""
            if status_enum == Status.SKIPPED:
                status_class = "skipped"
                status_val = "-"
            elif status_enum == Status.PASSED:
                status_class = "passed"
                status_val = "O"
            elif status_enum == Status.FAILED_RUN_EAGER_EXCEPTION:
                status_class = "failed"
                status_val = "e"
            elif status_enum == Status.FAILED_RUN_COMPILE_EXCEPTION:
                status_class = "failed"
                status_val = "E"
            elif status_enum == Status.FAILED_RUN_RETURN:
                status_class = "failed"
                status_val = "R"
            elif status_enum == Status.FAILED_COMPILE:
                status_class = "failed"
                status_val = "C"
            else:
                status_class = "skipped"
                status_val = "-"

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
