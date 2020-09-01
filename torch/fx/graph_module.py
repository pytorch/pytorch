import torch
import torch.overrides
import linecache
from typing import Type, Dict, List, Any
from .graph import Graph

# normal exec loses the source code, however we can patch
# the linecache module to still recover it.
# using exec_with_source will add it to our local cache
# and then tools like TorchScript will be able to get source info.
_next_id = 0
def exec_with_source(src: str, globals: Dict[str, Any]):
    global _next_id
    key = f'<eval_with_key_{_next_id}>'
    _next_id += 1
    _eval_cache[key] = [line + '\n' for line in src.splitlines()]
    exec(compile(src, key, 'exec'), globals)

# patch linecache so that any code we exec using exec_with_source
# works with inspect
_eval_cache : Dict[str, List[str]] = {}
_orig_getlines = linecache.getlines
def patched_getline(*args, **kwargs):
    if args[0] in _eval_cache:
        return _eval_cache[args[0]]
    return _orig_getlines(*args, **kwargs)
linecache.getlines = patched_getline


class GraphModule(torch.nn.Module):
    def __new__(cls: 'Type[GraphModule]', *args, **kwargs):
        # each instance of a graph module needs its own forward method
        # so create a new singleton class for each instance.
        # it is a subclass of the user-defined class, the only difference
        # is an extra layer to install the forward method

        class GraphModuleImpl(cls):  # type: ignore
            pass
        return super().__new__(GraphModuleImpl)

    def __init__(self, root: torch.nn.Module, graph: Graph):
        super().__init__()
        self.root = root
        self.training = self.root.training
        self.graph = graph
        self._generate_forward()

    def _generate_forward(self) -> None:
        body, result, free_variables = self.graph.python_code(root_module='self')
        body = '\n'.join('    ' + line for line in body.split('\n')) + '\n'
        self.code = f"""\
def forward(self, {', '.join(free_variables)}):
    self = self.root
{body}
    return {result}
"""
        # print(self.code)
        # install forward into the classes dictionary, this is what normally happens in the
        # 'class' statement
        # __new__ ensured that each instance has its own class
        gbls: Dict[str, Any] = {
            'torch': torch
        }
        exec_with_source(self.code, gbls)
        cls = type(self)
        for k, v in gbls.items():
            setattr(cls, k, v)

# workarounds for issues in __torch_function__

# WAR for __torch_function__ not handling tensor lists,
# fix is in https://github.com/pytorch/pytorch/pull/34725
# orig_cat = torch.cat
# def patched_cat(*args, **kwargs):
#     tensors = args[0]
#     for t in tensors:
#         if isinstance(t, Proxy):
#             return t.__torch_function__(patched_cat, (), args, kwargs)
#     return orig_cat(*args, **kwargs)
# patched_cat.__module__ = 'torch'
# patched_cat.__name__ = 'cat'
# torch.cat = patched_cat
