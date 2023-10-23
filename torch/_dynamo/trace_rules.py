import enum
import functools
import importlib

class TraceRule(enum.Enum):
    IN_GRAPH_FUNCTION = 0
    CONST_FOLD_FUNCTION = 1
    SUPPORTED_CTX_MANAGER_CLASS = 2
    SKIP = 3
    INLINE = 4

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
