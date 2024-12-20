# Owner(s): ["module: dynamo"]
import dataclasses
import importlib
import inspect
import math
import types
import unittest
import warnings
from typing import Any, Dict, Set

import torch
import torch._dynamo.config as config
import torch._dynamo.test_case
import torch._functorch.deprecated as deprecated_func
from torch._dynamo.trace_rules import (
    LEGACY_MOD_INLINELIST,
    load_object,
    manual_torch_name_rule_map,
    MOD_INLINELIST,
    torch_c_binding_in_graph_functions,
    torch_non_c_binding_in_graph_functions,
)
from torch._dynamo.utils import hashable, is_safe_constant, istype
from torch._dynamo.variables import TorchInGraphFunctionVariable, UserFunctionVariable
from torch.testing._internal.common_utils import skipIfWindows


try:
    from .utils import create_dummy_module_and_function
except ImportError:
    from utils import create_dummy_module_and_function


ignored_c_binding_in_graph_function_names = {
    # Ignored because they have manual rules defined at `trace_rules.manual_torch_name_rule_map`.
    "torch._nested_tensor_from_mask",
    "torch._nested_from_padded",
    "torch.sparse_compressed_tensor",
    "torch.sparse_bsc_tensor",
    "torch.sparse_bsr_tensor",
    "torch.sparse_coo_tensor",
    "torch.sparse_csc_tensor",
    "torch.sparse_csr_tensor",
    "torch.cuda._get_device_properties",
    # Ignored and go through rules defined at `trace_rules.check`.
    "torch._functionalize_are_all_mutations_under_no_grad_or_inference_mode",
    "torch._cslt_sparse_mm_search",
    "torch._C._abort",
    "torch._C._mps_is_on_macos_or_newer",
    "torch._C._swap_tensor_impl",
    "torch._C._unsafe_reset_storage",
    "torch._dynamo.eval_frame.reset_code",
    "torch._C.autocast_decrement_nesting",
    "torch._C.autocast_increment_nesting",
    "torch._C.clear_autocast_cache",
    "torch._C.set_anomaly_enabled",
    "torch._C.set_autocast_cache_enabled",
    "torch._C.set_autocast_cpu_dtype",
    "torch._C.set_autocast_cpu_enabled",
    "torch._C.set_autocast_enabled",
    "torch._C.set_autocast_gpu_dtype",
    "torch._C.set_autocast_ipu_dtype",
    "torch._C.set_autocast_ipu_enabled",
    "torch._C.set_autocast_xla_dtype",
    "torch._C.set_autocast_xla_enabled",
    "torch.resize_as_",
    "torch.resize_as_sparse_",
    "torch._C._data_address",
    "torch._C._is_cow_tensor",
    "torch._lazy_clone",
    "torch._test_parallel_materialize",
    "torch._C._storage_address",
    "torch._C._pickle_save",
    "torch._validate_sparse_compressed_tensor_args",
    "torch._validate_sparse_csr_tensor_args",
    "torch._validate_sparse_bsr_tensor_args",
    "torch._validate_sparse_csc_tensor_args",
    "torch._validate_sparse_coo_tensor_args",
    "torch._validate_sparse_bsc_tensor_args",
    "torch._validate_compressed_sparse_indices",
}
if torch._C._llvm_enabled():
    ignored_c_binding_in_graph_function_names |= {
        "torch._C._te.set_llvm_aot_workflow",
        "torch._C._te.set_llvm_target_cpu",
        "torch._C._te.set_llvm_target_attrs",
        "torch._C._te.set_llvm_target_triple",
    }


# Helper function to dump the torch name rule map generated based on
# the heuristic defined in gen_allowed_objs_and_ids.
def dump_allowed_torch_name_rule_map() -> None:
    m = gen_allowed_objs_and_ids(record=True, c_binding_only=False).name_rule_map
    for k, v in m.items():
        print(f'"{k}": {v.__name__},')


@dataclasses.dataclass
class AllowedObjects:
    """
    Track the objects, object id - name pairs, and name - dynamo wrapping rule pairs
    from the heuristic defined in `gen_allowed_objs_and_ids`.
    """

    object_ids: Dict[int, str]
    c_binding_in_graph_functions: Set[Any]
    non_c_binding_in_graph_functions: Set[Any]
    name_rule_map: Dict[str, Any]


def gen_allowed_objs_and_ids(record=False, c_binding_only=True) -> AllowedObjects:
    """
    Walk torch.* and get the ids of all the stuff in it
    """

    warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")
    torch_object_ids = {}
    c_binding_in_graph_functions = set()
    non_c_binding_in_graph_functions = set()
    torch_name_rule_map = {}

    # In some platforms, these functions were loaded as classes instead of functions.
    # To mitigate these weired cases, we need this special check.
    def is_special_functions(obj):
        return hashable(obj) and obj in {
            torch._C._cuda_isCurrentStreamCapturing,
            torch._C._graph_pool_handle,
        }

    # Add obj to c_binding_in_graph_functions set or non_c_binding_in_graph_functions set
    # if it's a torch function or method.
    # This is used to generate the in graph function list based on heuristic.
    def heuristic_record_if_in_graph_function(obj, module, name):
        try:
            if hasattr(obj, "__wrapped__"):
                obj = obj.__wrapped__
        except Exception:
            pass
        if isinstance(
            obj,
            (
                types.FunctionType,
                types.BuiltinFunctionType,
                types.MethodDescriptorType,
                types.WrapperDescriptorType,
            ),
        ) or is_special_functions(obj):
            torch_name_rule_map[
                f"{module.__name__}.{name}"
            ] = TorchInGraphFunctionVariable
            if c_binding_only:
                if not hasattr(obj, "__code__"):
                    c_binding_in_graph_functions.add(obj)
            else:
                if hasattr(obj, "__code__"):
                    non_c_binding_in_graph_functions.add(obj)
                else:
                    c_binding_in_graph_functions.add(obj)

    def _is_allowed_module_prefix(obj):
        allowed_modules = ("torch", "math")
        # torch.nn.modules.rnn is disallowed because these modules internally
        # flatten their parameters.  This flattening process will call
        # Tensor.set_ with a Storage, and Storages cannot be traced with
        # AOTAutograd; so we need to graph-break. To ensure this, we inline
        # these functions, rather than keep them opaque-ly in the graph.
        disallowed_modules = [
            "torch.optim.",
            "torch.nn.modules.rnn.",
            "torch._dynamo.",
            "torch._C._dynamo.",
            "torch._inductor.",
            "torch._C.inductor.",
            "torch.fx.",
            "torch._C._autograd",
            "torch._C._cudart",
            "torch._C._distributed_autograd",
            "torch._C._distributed_c10d",
            "torch._C._distributed_rpc",
            "torch._C._functorch",
            "torch._C._monitor",
            "torch._C._nvtx",
            "torch._C._lazy",
            "torch._C._profiler",
            "torch.__config__",
            "torch._custom_op",
            "torch._decomp",
            "torch._dispatch",
            "torch._export",
            "torch._functorch.make_functional",
            "torch._functorch.compile_utils",
            "torch._functorch.partitioners",
            "torch._functorch.aot_autograd",
            "torch._functorch.compilers",
            "torch._functorch.fx_minifier",
            "torch.autograd.profiler_util",
            "torch.autograd.profiler",
            "torch._jit_internal",
            "torch._library",
            "torch._lobpcg",
            "torch._logging",
            "torch._meta_registrations",
            "torch._namedtensor_internals",
            "torch._numpy",
            "torch._sources",
            "torch._subclasses",
            "torch._tensor",
            "torch._tensor_str",
            "torch._utils",
            "torch._utils_internal",
            "torch._vmap_internals",
            "torch.compiler",
            "torch.distributed",
            "torch.export",
            "torch.hub",
            "torch.jit",
            "torch.library",
            "torch.masked.maskedtensor",
            "torch.nn.init",
            "torch.nn.modules.module",
            "torch.nn.parallel",
            "torch.nn.utils",
            "torch.multiprocessing",
            "torch.onnx",
            "torch.overrides",
            "torch.package",
            "torch.profiler",
            "torch.serialization",
            "torch.storage",
            "torch.utils",
            "torch.distributed.",
        ]

        allowed_modules_dot = tuple([x + "." for x in allowed_modules])
        module = inspect.getmodule(obj)
        if module is None:
            return False

        mod_name = module.__name__

        if any(mod_name.startswith(m) for m in disallowed_modules):
            return False

        return mod_name in allowed_modules or mod_name.startswith(allowed_modules_dot)

    def _find_torch_objects(module):
        if any(
            module.__name__.startswith(mod_name)
            for mod_name in config.allowed_functions_module_string_ignorelist
        ):
            return
        torch_object_ids[id(module)] = module.__name__
        for name, obj in list(module.__dict__.items()):
            if id(obj) not in torch_object_ids:
                # Dynamo allows all builtins into the graph and does not attempt
                # to introspect into them. We don't want to allow instances of
                # HigherOrderOperator into the graph all the time (Dynamo needs
                # to introspect the body functions of these HigherOrderOperator
                # first, decide they are safe, and then allow them into the graph).
                # So we exclude HigherOrderOperator from being a builtin.
                import torch._ops

                if isinstance(obj, torch._ops.HigherOrderOperator):
                    continue

                # We want to trace through `grad` and `vmap`
                if obj in (
                    torch.func.grad,
                    deprecated_func.grad,
                    torch.func.vmap,
                    deprecated_func.vmap,
                    torch.nn.functional.triplet_margin_with_distance_loss,
                    torch.cond,
                ):
                    continue

                if isinstance(obj, types.ModuleType):
                    if obj.__name__.startswith("torch.") and _is_allowed_module_prefix(
                        obj
                    ):
                        torch_object_ids[id(obj)] = f"{module.__name__}.{name}"
                        _find_torch_objects(obj)
                elif _is_allowed_module_prefix(obj):
                    if record:
                        heuristic_record_if_in_graph_function(obj, module, name)
                    torch_object_ids[id(obj)] = f"{module.__name__}.{name}"
                elif inspect.getmodule(obj) is None and not is_safe_constant(obj):
                    if record:
                        heuristic_record_if_in_graph_function(obj, module, name)
                    torch_object_ids[id(obj)] = f"{module.__name__}.{name}"

    _find_torch_objects(torch)
    _find_torch_objects(math)

    return AllowedObjects(
        torch_object_ids,
        c_binding_in_graph_functions,
        non_c_binding_in_graph_functions,
        torch_name_rule_map,
    )


class TraceRuleTests(torch._dynamo.test_case.TestCase):
    def _check_set_equality(self, generated, used, rule_map, ignored_set):
        x = generated - used
        y = used - generated
        msg1 = (
            f"New torch objects: {x} "
            f"were not added to `trace_rules.{rule_map}` or `test_trace_rules.{ignored_set}`. "
            "Refer the instruction in `torch/_dynamo/trace_rules.py` for more details."
        )
        msg2 = (
            f"Existing torch objects: {y} were removed. "
            f"Please remove them from `trace_rules.{rule_map}` or `test_trace_rules.{ignored_set}`. "
            "Refer the instruction in `torch/_dynamo/trace_rules.py` for more details."
        )
        self.assertTrue(len(x) == 0, msg1)
        self.assertTrue(len(y) == 0, msg2)

    # We are using python function and module string names for these inlinelist,
    # this unit test is to make sure the functions/modules can be correctly imported
    # or loaded in case there is typo in the strings.
    def test_skipfiles_inlinelist(self):
        for m in LEGACY_MOD_INLINELIST.union(MOD_INLINELIST):
            try:
                mod = importlib.import_module(m)
            except ImportError:
                continue
            else:
                self.assertTrue(
                    isinstance(mod, types.ModuleType),
                    f"{m} from trace_rules.MOD_INLINELIST/LEGACY_MOD_INLINELIST "
                    "is not a python module, please check and correct it.",
                )

    @unittest.skip(
        "This test keeps getting broken and our disable infra is not handling well. see #120627"
    )
    def test_torch_name_rule_map_updated(self):
        # Generate the allowed objects based on heuristic defined in `allowed_functions.py`,
        objs = gen_allowed_objs_and_ids(record=True, c_binding_only=True)
        # Test C binding in graph functions are updated in torch_name_rule_map.
        generated = objs.c_binding_in_graph_functions
        used = set()
        for x in (
            set(torch_c_binding_in_graph_functions.keys())
            | ignored_c_binding_in_graph_function_names
        ):
            obj = load_object(x)
            if obj is not None:
                used.add(obj)
        self._check_set_equality(
            generated,
            used,
            "torch_c_binding_in_graph_functions",
            "ignored_c_binding_in_graph_function_names",
        )
        # For non C binding in graph functions, we only test if they can be loaded successfully.
        for f in torch_non_c_binding_in_graph_functions:
            self.assertTrue(
                isinstance(
                    load_object(f),
                    (
                        types.FunctionType,
                        types.BuiltinFunctionType,
                        types.MethodDescriptorType,
                        types.WrapperDescriptorType,
                    ),
                )
            )

    def test_force_inline_torch_function(self):
        # `torch._dynamo.utils.istype` is skipped by default
        def fn(x):
            if istype(x, torch.Tensor):
                return x + 1
            else:
                return x - 1

        _manual_torch_name_rule_map = manual_torch_name_rule_map.copy()
        # Force inline `torch._dynamo.utils.istype` by setting trace rule.
        _manual_torch_name_rule_map["torch._dynamo.utils.istype"] = UserFunctionVariable

        _torch_name_rule_map = [
            _manual_torch_name_rule_map,
            torch_c_binding_in_graph_functions,
            torch_non_c_binding_in_graph_functions,
        ]

        self.assertTrue(
            "torch._dynamo" not in torch._dynamo.trace_rules.LEGACY_MOD_INLINELIST
        )
        self.assertTrue("torch._dynamo" not in torch._dynamo.trace_rules.MOD_INLINELIST)

        with unittest.mock.patch(
            "torch._dynamo.trace_rules.torch_name_rule_map",
            _torch_name_rule_map,
        ), unittest.mock.patch(
            "torch._dynamo.trace_rules.get_torch_obj_rule_map",
            torch._dynamo.trace_rules.get_torch_obj_rule_map.__wrapped__,  # bypass functools.lru_cache
        ):
            x = torch.rand(3)
            opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)

    def test_force_inline_custom_function(self):
        mod, func = create_dummy_module_and_function()

        def fn(x):
            return func(x)

        _manual_torch_name_rule_map = manual_torch_name_rule_map.copy()
        # Force inline `mod.func` by setting trace rule.
        _manual_torch_name_rule_map[
            f"{mod.__name__}.{func.__name__}"
        ] = UserFunctionVariable

        _torch_name_rule_map = [
            _manual_torch_name_rule_map,
            torch_c_binding_in_graph_functions,
            torch_non_c_binding_in_graph_functions,
        ]

        with unittest.mock.patch(
            "torch._dynamo.trace_rules.torch_name_rule_map",
            _torch_name_rule_map,
        ), unittest.mock.patch(
            "torch._dynamo.trace_rules.get_torch_obj_rule_map",
            torch._dynamo.trace_rules.get_torch_obj_rule_map.__wrapped__,
        ):
            # First adding the module to SKIP_DIRS so that it will be skipped by default.
            torch._dynamo.trace_rules.add(mod.__name__)
            x = torch.rand(3)
            opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)


class TestModuleSurviveSkipFiles(torch._dynamo.test_case.TestCase):
    @unittest.skipIf(
        not torch.distributed.is_available(),
        "need to import MLP module from distributed",
    )
    @skipIfWindows(
        msg="AssertionError: False is not true : MLP did not survive skip files"
    )
    def test_module_survive_skip_files(self):
        from torch.testing._internal.common_fsdp import MLP

        model = MLP(3)
        inp = torch.randn((2, 3))
        frame_count_before = torch._dynamo.convert_frame.FRAME_COUNTER
        model.compile(backend="eager")
        model(inp)
        frame_count_after = torch._dynamo.convert_frame.FRAME_COUNTER
        self.assertTrue(
            frame_count_after > frame_count_before, "MLP did not survive skip files"
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
