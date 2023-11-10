import functools
import importlib
import types

import torch

from .allowed_functions import (
    _disallowed_function_ids,
    is_allowed,
    is_user_defined_allowed,
)

from .utils import hashable

from .variables import TorchCtxManagerClassVariable, TorchInGraphFunctionVariable


"""
Map of torch objects to their tracing rules (Dynamo variables).
* TorchVariable: The functions should be put into the FX graph or can be constant folded. E.g.,
  - torch.add: should be put into the FX graph.
  - torch.is_floating_point: constant folded.
* TorchCtxManagerClassVariable: The context manager classes are supported by Dynamo. E.g., torch.no_grad
* SkipFilesVariable: The objects should be skipped from tracing.
* UserFunctionVariable: The functions should be inlined.

We explicitly list torch objects which should be wrapped as TorchCtxManagerClassVariable.
The initial list comes from the heuristic in test/dynamo/test_trace_rules.py:generate_allow_list.

For developers: If you add/remove a torch level API, it may trigger failures from
test/dynamo/test_trace_rules.py:test_torch_name_rule_map. To fix the failures:
If you are adding a new torch level API or Dynamo implementation:
* Add the name with TorchCtxManagerClassVariable to this map
  if you are adding Dynamo implementation for that context manager.
* Remove the object name from test/dynamo/test_trace_rules.ignored_torch_name_rule_set if it's there.

If you are removing an existing torch level API:
* Remove the entry represented the API from this map or test/dynamo/test_trace_rules.ignored_torch_name_rule_set
  depends on where it is.

TODO: Add torch object names mapping to TorchVariable for in graph and constant fold functions.
TODO: We would consolidate the skipfiles.check rules into trace_rules.lookup later.
TODO: We would support explictly list objects treated as skip/inline after the skipfiles.check
and trace_rules.lookup consolidation is done. Then the explicit listing of skip/inline objects have
a higher priority, which can be used to override the skipfiles.check rules in some cases.
"""
manual_torch_name_rule_map = {
    "torch.profiler.profiler.profile": TorchCtxManagerClassVariable,
    "torch.autograd.profiler.profile": TorchCtxManagerClassVariable,
    "torch.autograd.profiler.record_function": TorchCtxManagerClassVariable,
    "torch.default_generator#get_state": TorchInGraphFunctionVariable,
    "torch._C.Generator#get_state": TorchInGraphFunctionVariable,
    "torch.default_generator#set_state": TorchInGraphFunctionVariable,
    "torch._C.Generator#set_state": TorchInGraphFunctionVariable,
    "torch.onnx.is_in_onnx_export": TorchInGraphFunctionVariable,
    "torch.onnx.operators.shape_as_tensor": TorchInGraphFunctionVariable,
    "torch.overrides.is_tensor_like": TorchInGraphFunctionVariable,
    "torch.jit.is_scripting": TorchInGraphFunctionVariable,
    "torch.jit.is_tracing": TorchInGraphFunctionVariable,
    "torch.jit.annotate": TorchInGraphFunctionVariable,
    "torch.distributed.is_available": TorchInGraphFunctionVariable,
    "torch.distributed.is_initialized": TorchInGraphFunctionVariable,
    "torch.distributed.get_rank": TorchInGraphFunctionVariable,
    "torch.distributed.get_world_size": TorchInGraphFunctionVariable,
    "torch.distributed._tensor.DTensor#from_local": TorchInGraphFunctionVariable,
    "torch._utils.is_compiling": TorchInGraphFunctionVariable,
    "torch.overrides.get_default_nowrap_functions": TorchInGraphFunctionVariable,
    "torch.fx._symbolic_trace.is_fx_tracing": TorchInGraphFunctionVariable,
    "torch._dynamo.external_utils.is_compiling": TorchInGraphFunctionVariable,
    "torch.autograd.graph.disable_saved_tensors_hooks": TorchInGraphFunctionVariable,
}


auto_torch_name_rule_map = {
    # Dynamo implemented context managers
    "torch._C.DisableTorchFunctionSubclass": TorchCtxManagerClassVariable,
    "torch.amp.autocast_mode.autocast": TorchCtxManagerClassVariable,
    "torch.autograd.grad_mode.enable_grad": TorchCtxManagerClassVariable,
    "torch.autograd.grad_mode.inference_mode": TorchCtxManagerClassVariable,
    "torch.autograd.grad_mode.no_grad": TorchCtxManagerClassVariable,
    "torch.autograd.grad_mode.set_grad_enabled": TorchCtxManagerClassVariable,
    "torch.cpu.amp.autocast_mode.autocast": TorchCtxManagerClassVariable,
    "torch.cuda.amp.autocast_mode.autocast": TorchCtxManagerClassVariable,
}


torch_name_rule_map = {**manual_torch_name_rule_map, **auto_torch_name_rule_map}


@functools.lru_cache(None)
def get_torch_obj_rule_map():
    d = dict()
    for k, v in torch_name_rule_map.items():
        try:
            obj = load_object(k)
            d[obj] = v
        except (AttributeError, ModuleNotFoundError):
            pass
    return d


def _load_obj_from_str(fully_qualified_name):
    module, obj_name = fully_qualified_name.rsplit(".", maxsplit=1)
    return getattr(importlib.import_module(module), obj_name)


def load_object(name):
    x = name.split("#")
    if len(x) == 2:
        obj = _load_obj_from_str(x[0])
        val = getattr(obj, x[1])
    else:
        assert len(x) == 1
        val = _load_obj_from_str(x[0])
    if hasattr(val, "__wrapped__") and val is not torch.ops:
        val = val.__wrapped__
    return val


@functools.lru_cache(None)
def get_tensor_method():
    s = set()
    for name in dir(torch.Tensor):
        method = getattr(torch.Tensor, name)
        if isinstance(
            method, (types.MethodDescriptorType, types.WrapperDescriptorType)
        ):
            s.add(method)
    return s


def is_in_graph_function(obj):
    if obj in get_tensor_method() or isinstance(
        obj,
        (torch._ops.OpOverloadPacket, torch._ops.OpOverload),
    ):
        return True
    if isinstance(
        obj,
        (
            types.FunctionType,
            types.MethodType,
            types.BuiltinFunctionType,
            types.MethodDescriptorType,
            types.WrapperDescriptorType,
        ),
    ):
        return is_allowed(obj)
    else:
        return False


def lookup(obj):
    if not hashable(obj):
        return None
    if id(obj) in _disallowed_function_ids:
        return None
    if hasattr(obj, "__wrapped__") and obj is not torch.ops:
        obj = obj.__wrapped__
    rule = get_torch_obj_rule_map().get(obj, None)
    if rule is None and (is_in_graph_function(obj) or is_user_defined_allowed(obj)):
        return TorchInGraphFunctionVariable
    else:
        return rule
