# Owner(s): ["module: dynamo"]
import collections
import copy
import importlib
import inspect
import math
import types
import unittest
import warnings

import torch
import torch._dynamo.config as config
import torch._dynamo.test_case
import torch._functorch.deprecated as deprecated_func
from torch._dynamo.external_utils import is_compiling
from torch._dynamo.skipfiles import (
    FUNC_INLINELIST,
    LEGACY_MOD_INLINELIST,
    MOD_INLINELIST,
)
from torch._dynamo.trace_rules import get_torch_obj_rule_map, load_object
from torch._dynamo.utils import is_safe_constant, istype
from torch.fx._symbolic_trace import is_fx_tracing

try:
    from .utils import create_dummy_module_and_function
except ImportError:
    from utils import create_dummy_module_and_function


ignored_torch_name_rule_set = {
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
    "torch._C._profiler._RecordFunctionFast",
    "torch._subclasses.fake_tensor.FakeTensorMode",
    "torch._subclasses.functional_tensor.FunctionalTensorMode",
    "torch.ao.nn.sparse.quantized.utils.LinearBlockSparsePattern",
    "torch.autograd.anomaly_mode.detect_anomaly",
    "torch.autograd.anomaly_mode.set_detect_anomaly",
    "torch.autograd.forward_ad._set_fwd_grad_enabled",
    "torch.autograd.forward_ad.dual_level",
    "torch.autograd.grad_mode._force_original_view_tracking",
    "torch.autograd.grad_mode._unsafe_preserve_version_counter",
    "torch.autograd.grad_mode.set_multithreading_enabled",
    "torch.autograd.graph.saved_tensors_hooks",
    "torch.autograd.profiler.emit_itt",
    "torch.autograd.profiler.emit_nvtx",
    "torch.autograd.profiler_legacy.profile",
    "torch.backends.mkl.verbose",
    "torch.backends.mkldnn.verbose",
    "torch.cpu.StreamContext",
    "torch.cuda.StreamContext",
    "torch.cuda._DeviceGuard",
    "torch.cuda.device",
    "torch.cuda.graphs.graph",
    "torch.device",  # constant folding
    "torch.distributed.autograd.context",
    "torch.hub._Faketqdm",
    "torch.jit._ir_utils._InsertPoint",
    "torch.jit._script.RecursiveScriptClass",
    "torch.jit.strict_fusion",
    "torch.onnx._internal.diagnostics.infra.context.DiagnosticContext",
    "torch.onnx._internal.fx.patcher.ONNXTorchPatcher",
    "torch.overrides.TorchFunctionMode",
    "torch.package.package_exporter.PackageExporter",
    "torch.serialization._opener",
    "torch.sparse.check_sparse_tensor_invariants",
    "torch.utils._config_module.ContextDecorator",
    "torch.utils._contextlib._DecoratorContextManager",
    "torch.utils._device.DeviceContext",
    "torch.utils._python_dispatch.TorchDispatchMode",
    "torch.utils.data.datapipes._decorator.guaranteed_datapipes_determinism",
    "torch.utils.data.datapipes._decorator.runtime_validation_disabled",
    "torch.utils.data.datapipes.dataframe.dataframes.CaptureLikeMock",
    "torch.utils.hooks.RemovableHandle",
}


if torch.distributed.is_available():
    ignored_torch_name_rule_set |= {
        "torch.distributed.rpc.server_process_global_profiler._server_process_global_profile",
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


def _disallowed_function_ids():
    remove = [
        True,
        False,
        None,
        collections.OrderedDict,
        copy.copy,
        copy.deepcopy,
        inspect.signature,
        math.__package__,
        torch.__builtins__,
        torch.autocast_decrement_nesting,
        torch.autocast_increment_nesting,
        torch.autograd.grad,
        torch.clear_autocast_cache,
        torch.cuda.current_device,
        torch.cuda.set_device,
        torch.distributions.constraints.is_dependent,
        torch.distributions.normal.Normal,
        torch.inference_mode,
        torch.jit.isinstance,
        torch.set_anomaly_enabled,
        torch.set_autocast_cache_enabled,
        torch.set_autocast_cpu_dtype,
        torch.set_autocast_cpu_enabled,
        torch.set_autocast_enabled,
        torch.set_autocast_gpu_dtype,
        warnings.warn,
        torch._C._dynamo.eval_frame.unsupported,
        torch.Tensor.__init__,
    ]

    # extract all dtypes from torch
    dtypes = [
        obj for obj in torch.__dict__.values() if isinstance(obj, type(torch.float32))
    ]
    remove += dtypes
    storage = [
        obj
        for obj in torch.__dict__.values()
        if isinstance(obj, type(torch.FloatStorage))
    ]
    remove += storage

    # Distributed APIs don't work well with torch.compile.
    if torch.distributed.is_available():
        remove.extend(
            torch.distributed.distributed_c10d.dynamo_unsupported_distributed_c10d_ops
        )

    return {id(x) for x in remove}


def generate_allow_list():
    """
    Walk torch.* and get the ids of all the stuff in it
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")
    torch_object_ids = dict()
    torch_objects = set()

    def _is_allowed_module_prefix(obj):
        allowed_modules = ("torch", "math")
        # torch.nn.modules.rnn is disallowed because these modules internally
        # flatten their parameters.  This flattening process will call
        # Tensor.set_ with a Storage, and Storages cannot be traced with
        # AOTAutograd; so we need to graph-break. To ensure this, we inline
        # these functions, rather than keep them opaque-ly in the graph.
        disallowed_modules = [
            "torch.optim.",
            "torch.utils._foreach_utils",  # omit the period so we match all the functions in this module
            "torch.utils._pytree",
            "torch.nn.modules.rnn.",
            "torch._dynamo.",
            "torch._C._dynamo.",
            "torch._inductor.",
            "torch._C.inductor.",
            "torch.fx.",
            "torch.distributed.fsdp.",
            "torch.distributed._tensor.",
            # Inline through the ActivationWrapper in
            # torch.distributed.algorithms._checkpoint.checkpoint_wrapper. This
            # nn module calls torch.utils.checkpoint internally. If Dynamo does
            # not trace this, AOT Autograd will try to trace this and can cause
            # issues observed in
            # https://github.com/pytorch/pytorch/issues/108269
            "torch.distributed.algorithms.",
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
                    torch_object_ids[id(obj)] = f"{module.__name__}.{name}"
                    if (
                        issubclass(type(obj), type)
                        and "__enter__" in obj.__dict__
                        and "__exit__" in obj.__dict__
                    ):
                        torch_objects.add(obj)
                elif inspect.getmodule(obj) is None and not is_safe_constant(obj):
                    torch_object_ids[id(obj)] = f"{module.__name__}.{name}"
                    if (
                        issubclass(type(obj), type)
                        and "__enter__" in obj.__dict__
                        and "__exit__" in obj.__dict__
                    ):
                        torch_objects.add(obj)

    _find_torch_objects(torch)
    _find_torch_objects(math)

    if config.trace_distributed:
        from torch.distributed import _functional_collectives_impl as fci

        for f in [
            fci._all_gather_into_tensor,
            fci._all_reduce,
            fci._reduce_scatter_tensor,
            fci._all_reduce_coalesced,
            fci._all_gather_into_tensor_coalesced,
            fci._reduce_scatter_tensor_coalesced,
        ]:
            torch_object_ids[id(f)] = repr(f)

    # torch.Tensor.{fn}
    for name in dir(torch.Tensor):
        method = getattr(torch.Tensor, name)
        if isinstance(
            method, (types.MethodDescriptorType, types.WrapperDescriptorType)
        ):
            torch_object_ids[id(method)] = f"torch.Tensor.{name}"

    for idx in _disallowed_function_ids():
        if idx in torch_object_ids:
            del torch_object_ids[idx]

    for extra in (is_fx_tracing, is_compiling):
        torch_object_ids[id(extra)] = f"{extra.__module__}.{extra.__name__}"

    return torch_objects


class TraceRuleTests(torch._dynamo.test_case.TestCase):
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

    def test_torch_name_rule_map(self):
        generated_torch_name_rule_set = generate_allow_list()
        ignored_torch_obj_rule_set = {
            load_object(x) for x in ignored_torch_name_rule_set
        }
        used_torch_name_rule_set = (
            set(get_torch_obj_rule_map().keys()) | ignored_torch_obj_rule_set
        )
        x = generated_torch_name_rule_set - used_torch_name_rule_set
        y = used_torch_name_rule_set - generated_torch_name_rule_set
        msg1 = (
            f"New torch objects: {x} "
            "were not added to trace_rules.torch_name_rule_map or test_trace_rules.ignored_torch_name_rule_set. "
            "Refer the instruction in `torch/_dynamo/trace_rules.py` for more details."
        )
        msg2 = (
            f"Existing torch objects: {y} were removed. "
            "Please remove them from trace_rules.torch_name_rule_map or test_trace_rules.ignored_torch_name_rule_set. "
            "Refer the instruction in `torch/_dynamo/trace_rules.py` for more details."
        )
        self.assertTrue(len(x) == 0, msg1)
        self.assertTrue(len(y) == 0, msg2)

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
