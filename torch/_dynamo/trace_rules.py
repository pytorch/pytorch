import enum
import functools
import importlib

from . import variables

"""
Define the Dynamo tracing rules for Torch objects.
* IN_GRAPH_FUNCTION: The functions should be put into the FX graph or can be constant folded. E.g.,
  - torch.add: should be put into the FX graph.
  - torch.is_floating_point: constant folded.
* SUPPORTED_CTX_MANAGER_CLASS: The context manager classes are supported by Dynamo. E.g., torch.no_grad
* SKIP: The objects should be skipped from tracing.
* INLINE: The functions should be inlined.
"""


class TraceRule(enum.Enum):
    IN_GRAPH_FUNCTION = 0
    SUPPORTED_CTX_MANAGER_CLASS = 1
    SKIP = 2
    INLINE = 3


trace_rule_map = {
    TraceRule.IN_GRAPH_FUNCTION: variables.TorchVariable,
    TraceRule.SUPPORTED_CTX_MANAGER_CLASS: variables.TorchCtxManagerClassVariable,
    TraceRule.SKIP: variables.SkipFilesVariable,
    TraceRule.INLINE: variables.UserFunctionVariable,
}

"""
Map of torch object to its trace rule.

We explicitly list torch objects which are treated as IN_GRAPH_FUNCTION and SUPPORTED_CTX_MANAGER_CLASS.
The initial list comes from the heuristic in test/dynamo/test_trace_rules.py:generate_allow_list.

For developers: If you add/remove a torch level API, it may trigger failures from
test/dynamo/test_trace_rules.py:test_torch_name_rule_map_correctness.
To fix them, please follow these steps:
* Add/remove the function name with TraceRule.IN_GRAPH_FUNCTION to this map if it's treated as IN_GRAPH_FUNCTION.
* Add/remove the context manager class name with TraceRule.SUPPORTED_CTX_MANAGER_CLASS to this map
  if you added/removed Dynamo implementation for that context manager.
* Add/remove the object name to test/dynamo/test_trace_rules.ignored_torch_name_rule_set if you think
  it's not IN_GRAPH_FUNCTION or SUPPORTED_CTX_MANAGER_CLASS.

TraceRule.SKIP and TraceRule.INLINE are not used for now. Please check the skip/inline rules at skipfiles.check.
TODO: We would consolidate the skipfiles.check rules into trace_rules.check later.
TODO: We would support explictly list objects treated as SKIP/INLINE after the skipfiles.check
and trace_rules.check consolidation is done. Then the explicit listing of SKIP/INLINE objects have
a higher priority, which can be used to override the skipfiles.check rules in some cases.
"""
torch_name_rule_map = {
    "torch._C.DisableTorchFunctionSubclass": TraceRule.SUPPORTED_CTX_MANAGER_CLASS,
    "torch.amp.autocast_mode.autocast": TraceRule.SUPPORTED_CTX_MANAGER_CLASS,
    "torch.autograd.grad_mode.enable_grad": TraceRule.SUPPORTED_CTX_MANAGER_CLASS,
    "torch.autograd.grad_mode.inference_mode": TraceRule.SUPPORTED_CTX_MANAGER_CLASS,
    "torch.autograd.grad_mode.no_grad": TraceRule.SUPPORTED_CTX_MANAGER_CLASS,
    "torch.autograd.grad_mode.set_grad_enabled": TraceRule.SUPPORTED_CTX_MANAGER_CLASS,
    "torch.autograd.profiler.profile": TraceRule.SUPPORTED_CTX_MANAGER_CLASS,
    "torch.autograd.profiler.record_function": TraceRule.SUPPORTED_CTX_MANAGER_CLASS,
    "torch.cpu.amp.autocast_mode.autocast": TraceRule.SUPPORTED_CTX_MANAGER_CLASS,
    "torch.cuda.amp.autocast_mode.autocast": TraceRule.SUPPORTED_CTX_MANAGER_CLASS,
    "torch.profiler.profiler.profile": TraceRule.SUPPORTED_CTX_MANAGER_CLASS,
}


@functools.lru_cache(None)
def get_torch_obj_rule_map():
    d = dict()
    for k, v in torch_name_rule_map.items():
        obj = load_object(k)
        assert obj not in d
        d[obj] = v
    return d


def load_object(name):
    mod_name, obj_name = name.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, obj_name)
    return obj


def check(obj):
    rule = get_torch_obj_rule_map().get(obj, None)
    return rule
