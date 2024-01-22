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
from torch._dynamo.skipfiles import (
    FUNC_INLINELIST,
    LEGACY_MOD_INLINELIST,
    MOD_INLINELIST,
)
from torch._dynamo.trace_rules import (
    load_object,
    torch_c_binding_in_graph_functions,
    torch_ctx_manager_classes,
    torch_non_c_binding_in_graph_functions,
)
from torch._dynamo.utils import hashable, is_safe_constant, istype
from torch._dynamo.variables import (
    TorchCtxManagerClassVariable,
    TorchInGraphFunctionVariable,
)

try:
    from .utils import create_dummy_module_and_function
except ImportError:
    from utils import create_dummy_module_and_function


ignored_ctx_manager_class_names = {
    "torch.ExcludeDispatchKeyGuard",
    "torch._C.DisableTorchFunction",
    "torch._C._AutoDispatchBelowAutograd",
    "torch._C._DisableAutocast",
    "torch._C._DisableFuncTorch",
    "torch._C._DisablePythonDispatcher",
    "torch._C._DisableTorchDispatch",
    "torch._C._EnablePreDispatch",
    "torch._C._EnablePythonDispatcher",
    "torch._C._EnableTorchFunction",
    "torch._C._ExcludeDispatchKeyGuard",
    "torch._C._ForceDispatchKeyGuard",
    "torch._C._IncludeDispatchKeyGuard",
    "torch._C._InferenceMode",
    "torch._C._RestorePythonTLSSnapshot",
    "torch._C._SetExcludeDispatchKeyGuard",
    "torch.ao.nn.sparse.quantized.utils.LinearBlockSparsePattern",
    "torch.autograd.anomaly_mode.detect_anomaly",
    "torch.autograd.anomaly_mode.set_detect_anomaly",
    "torch.autograd.forward_ad._set_fwd_grad_enabled",
    "torch.autograd.forward_ad.dual_level",
    "torch.autograd.grad_mode._force_original_view_tracking",
    "torch.autograd.grad_mode._unsafe_preserve_version_counter",
    "torch.autograd.grad_mode.set_multithreading_enabled",
    "torch.autograd.graph._CloneArgBeforeMutateMode",
    "torch.autograd.graph._swap_with_cloned",
    "torch.autograd.graph.save_on_cpu",
    "torch.autograd.graph.saved_tensors_hooks",
    "torch.backends.mkl.verbose",
    "torch.backends.mkldnn.verbose",
    "torch.cpu.StreamContext",
    "torch.cuda.StreamContext",
    "torch.cuda._DeviceGuard",
    "torch.cuda.device",
    "torch.cuda.device_of",
    "torch.cuda.graphs.graph",
    "torch.device",  # as constant folding function
    "torch.sparse.check_sparse_tensor_invariants",
}

ignored_c_binding_in_graph_function_names = {
    # Ignored because they have manual rules defined at `trace_rules.manual_torch_name_rule_map`.
    "torch._nested_tensor_from_mask",
    "torch._nested_from_padded",
    # Ignored and go through rules defined at `skipfiles.check`.
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
    ctx_mamager_classes: Set[Any]
    c_binding_in_graph_functions: Set[Any]
    non_c_binding_in_graph_functions: Set[Any]
    name_rule_map: Dict[str, Any]


def gen_allowed_objs_and_ids(record=False, c_binding_only=True) -> AllowedObjects:
    """
    Walk torch.* and get the ids of all the stuff in it
    """

    warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")
    torch_object_ids = dict()
    ctx_mamager_classes = set()
    c_binding_in_graph_functions = set()
    non_c_binding_in_graph_functions = set()
    torch_name_rule_map = dict()

    # Add obj to ctx_mamager_classes set if it's a torch context manager class.
    # This is used to generate the ctx manager class list based on heuristic.
    def heuristic_record_if_ctx_manager(obj, module, name):
        if (
            issubclass(type(obj), type)
            and hasattr(obj, "__enter__")
            and hasattr(obj, "__exit__")
        ):
            torch_name_rule_map[
                f"{module.__name__}.{name}"
            ] = TorchCtxManagerClassVariable
            ctx_mamager_classes.add(obj)

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
        ]
        if config.trace_distributed:
            disallowed_modules.append("torch.distributed.")

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
                        heuristic_record_if_ctx_manager(obj, module, name)
                        heuristic_record_if_in_graph_function(obj, module, name)
                    torch_object_ids[id(obj)] = f"{module.__name__}.{name}"
                elif inspect.getmodule(obj) is None and not is_safe_constant(obj):
                    if record:
                        heuristic_record_if_ctx_manager(obj, module, name)
                        heuristic_record_if_in_graph_function(obj, module, name)
                    torch_object_ids[id(obj)] = f"{module.__name__}.{name}"

    _find_torch_objects(torch)
    _find_torch_objects(math)

    return AllowedObjects(
        torch_object_ids,
        ctx_mamager_classes,
        c_binding_in_graph_functions,
        non_c_binding_in_graph_functions,
        torch_name_rule_map,
    )


def gen_get_func_inlinelist(dummy_func_inlinelist):
    def get_func_inlinelist():
        inlinelist = set()
        for f in dummy_func_inlinelist:
            module_name, fn_name = f.rsplit(".", 1)
            m = importlib.import_module(module_name)
            fn = getattr(m, fn_name)
            inlinelist.add(fn.__code__)
        return inlinelist

    return get_func_inlinelist


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
            self.assertTrue(
                isinstance(importlib.import_module(m), types.ModuleType),
                f"{m} from skipfiles.MOD_INLINELIST/LEGACY_MOD_INLINELIST is not a python module, please check and correct it.",
            )
        for f in FUNC_INLINELIST:
            module_name, fn_name = f.rsplit(".", 1)
            m = importlib.import_module(module_name)
            self.assertTrue(
                isinstance(getattr(m, fn_name), types.FunctionType),
                f"{f} from skipfiles.FUNC_INLINELIST is not a python function, please check and correct it.",
            )

    def test_torch_name_rule_map_updated(self):
        # Generate the allowed objects based on heuristic defined in `allowed_functions.py`,
        objs = gen_allowed_objs_and_ids(record=True, c_binding_only=True)
        # Test ctx manager classes are updated in torch_name_rule_map.
        generated = objs.ctx_mamager_classes
        used = set()
        for x in (
            set(torch_ctx_manager_classes.keys()) | ignored_ctx_manager_class_names
        ):
            obj = load_object(x)
            if obj is not None:
                used.add(obj)
        self._check_set_equality(
            generated,
            used,
            "torch_ctx_manager_classes",
            "ignored_ctx_manager_class_names",
        )
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
                        types.MethodType,
                        types.BuiltinFunctionType,
                        types.MethodDescriptorType,
                        types.WrapperDescriptorType,
                    ),
                )
            )

    def test_func_inlinelist_torch_function(self):
        def fn(x):
            if istype(x, torch.Tensor):
                return x + 1
            else:
                return x - 1

        func_inlinelist = torch._dynamo.skipfiles.FUNC_INLINELIST.copy()
        func_inlinelist.add("torch._dynamo.utils.istype")

        self.assertTrue(
            "torch._dynamo" not in torch._dynamo.skipfiles.LEGACY_MOD_INLINELIST
        )
        self.assertTrue("torch._dynamo" not in torch._dynamo.skipfiles.MOD_INLINELIST)

        with unittest.mock.patch(
            "torch._dynamo.skipfiles.get_func_inlinelist",
            gen_get_func_inlinelist(func_inlinelist),
        ):
            x = torch.rand(3)
            opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)

    def test_func_inlinelist_third_party_function(self):
        mod, func = create_dummy_module_and_function()

        def fn(x):
            return func(x)

        func_inlinelist = torch._dynamo.skipfiles.FUNC_INLINELIST.copy()
        func_inlinelist.add(f"{mod.__name__}.{func.__name__}")

        with unittest.mock.patch(
            "torch._dynamo.skipfiles.get_func_inlinelist",
            gen_get_func_inlinelist(func_inlinelist),
        ), unittest.mock.patch(
            "torch._dynamo.skipfiles.SKIP_DIRS",
            torch._dynamo.skipfiles.SKIP_DIRS.copy(),
        ):
            # First adding the module to SKIP_DIRS so that it will be skipped.
            torch._dynamo.skipfiles.add(mod.__name__)
            x = torch.rand(3)
            opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
