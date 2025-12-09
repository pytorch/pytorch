import torch
from torch import nn
from torch.compiler import nested_compile_region
import torch.utils._pytree as pytree
from torch._functorch.aot_autograd import create_functional_call


def get_mod_params_and_buffers(mod):
    named_parameters = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))
    params_and_buffers = {
        **dict(named_parameters),
        **dict(named_buffers),
    }
    params_and_buffers_flat_l, params_spec = pytree.tree_flatten(params_and_buffers)
    params_and_buffers_flat = tuple(params_and_buffers_flat_l)
    return params_and_buffers_flat


def functionalize(_mod):
    class MyMod(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            out = self.module(*args, **kwargs)
            flat_out, self.spec = pytree.tree_flatten(out)
            return flat_out

    mod = MyMod(_mod)
    named_parameters = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))
    params_and_buffers = {
        **dict(named_parameters),
        **dict(named_buffers),
    }
    params_and_buffers_flat_l, params_spec = pytree.tree_flatten(params_and_buffers)
    params_and_buffers_flat = tuple(params_and_buffers_flat_l)
    params_len = len(params_and_buffers_flat)
    buffer_len = sum([not isinstance(x, nn.Parameter) for x in params_and_buffers_flat])
    functional_call = create_functional_call(
        mod, params_spec, params_len, store_orig_mod=True
    )
    return functional_call, params_and_buffers_flat  # , mod.spec


def _hash_module(module):
    return type(module)


def functionalize2(module):
    return lambda parameter_and_buffer_dicts, args, kwargs: torch.func.functional_call(
        module, parameter_and_buffer_dicts, args, kwargs
    )


def get_parameters_and_buffer_dicts(module):
    named_parameters = dict(module.named_parameters(remove_duplicate=False))
    named_buffers = dict(module.named_buffers(remove_duplicate=False))
    params_and_buffers = {
        **dict(named_parameters),
        **dict(named_buffers),
    }
    return params_and_buffers


class Sequential(nn.Module):
    """
    A variant of nn.Sequential that uses functionalization to compile the submodules
    only once per submodule that shares the same hash
    """

    def __init__(self, *modules):
        super().__init__()
        self._mod = nn.ModuleList(modules)
        self._functional_calls = {}
        self.func_type = "v2"
        use_nested_compile_region = True
        for module in self._mod:
            module_hash = _hash_module(module)
            if module_hash not in self._functional_calls:
                if self.func_type == "v1":
                    fn = functionalize(module)[0]
                else:
                    fn = functionalize2(module)
                if use_nested_compile_region:
                    fn = nested_compile_region(fn, is_pure=True)
                    # fn = nested_compile_region(fn)
                self._functional_calls[module_hash] = fn

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            x = (x,)
        x = tuple(x)
        for module in self._mod:
            module_hash = _hash_module(module)
            f = self._functional_calls[module_hash]
            if self.func_type == "v1":
                args = get_mod_params_and_buffers(module)
                new_args = tuple(args) + tuple(x)
                x = f(*new_args, **{})
            else:
                x = f(get_parameters_and_buffer_dicts(module), x, {})[0]
        return tuple(x)


class MyBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, out_dim)
        self.l2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return (self.l2(self.l1(x)), )

class MyBlock2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, out_dim)
        self.l2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return (self.l1(x) + self.l2(x), )

m = Sequential(MyBlock(10, 10), MyBlock(10, 10), MyBlock(10, 10), MyBlock2(10, 10), MyBlock(10, 10), MyBlock2(10, 10))
# m = Sequential(MyBlock(10, 10), MyBlock(10, 10), MyBlock(10, 10))
# , nn.Linear(10, 1))
x = torch.randn(1, 10, requires_grad=True)
o = m(x)
opt_mod = torch.compile(m, fullgraph=True, backend="aot_eager")
oo = opt_mod(x)
