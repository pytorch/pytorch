# Owner(s): ["module: dynamo"]
import importlib
import types
import unittest

import torch
import torch._dynamo.test_case
from torch._dynamo.allowed_functions import gen_allowed_objs_and_ids
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
from torch._dynamo.utils import istype

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
    "torch._functionalize_are_all_mutations_under_no_grad_or_inference_mode",
    "torch._C._swap_tensor_impl",
    "torch._C._unsafe_reset_storage",
    "torch._dynamo.eval_frame.reset_code",
}
if torch._C._llvm_enabled():
    ignored_c_binding_in_graph_function_names |= {
        "torch._C._te.set_llvm_aot_workflow",
        "torch._C._te.set_llvm_target_cpu",
        "torch._C._te.set_llvm_target_attrs",
        "torch._C._te.set_llvm_target_triple",
    }


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
