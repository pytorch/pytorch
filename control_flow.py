import torch
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
import torch._C as _C
from torch._C import DispatchKey, DispatchKeySet, ExcludeDispatchKeyGuard
from torch.overrides import handle_torch_function, has_torch_function

"""
Structured control flow operators prototype.

This is a prototype of the cond operator. Its API is the following:
  cond(pred, true_fn, false_fn, *operands)

Note that cond has some special features that makes it different from
an ATen operator:
- It accepts two functions as arguments (true_fn, false_fn)
- It accepts varargs (*operands)

We don't quite know how to shoehorn these types into the PyTorch dispatcher
(I'm sure it's possible though), so the proposal is: handle all of the
weirdness in Python.

The approach we take is:
- We set up a "Python version of the PyTorch Dispatcher" (call this PyDispatcher).
  This is responsible for performing dispatch on operations in python.
- We have a notion of a "pyoperator" (not to be confused with Anjali's Python Op
  Registration API). A "pyoperator" is an Operator that was defined in Python
  and handled by the "Python version of the PyTorch Dispatcher"
  (Anjali's Python Op Registration API creates operators in Python that are handled
  by the PyTorch C++ Dispatcher).
- A "pyoperator":
  - Does not require a schema
  - Can accept functions as arguments
  - Can accept varargs as arguments

Given a PyOperator, we can define "rules" for it for each dispatch key.
"""


SUPPORTED_KEYS = {
    DispatchKey.CPU,
    DispatchKey.BackendSelect,
    DispatchKey.ADInplaceOrView,
    DispatchKey.AutogradCPU,
    DispatchKey.Python,
    DispatchKey.PythonTLSSnapshot,
}

"""
This is a dispatcher (in Python)
- You can define new operations (in Python) without schemas
- It interfaces with the PyTorch dispatcher
"""

class PyDispatcher:
    def __init__(self):
        self.current_dispatching_op = None
        self.already_dispatched_keys = None

    def call(self, operator, args, kwargs):
        try:
            key = compute_dispatch_key(operator, args, kwargs)
            self.record_dispatch(key, operator)
            print(f"PyDispatcher.call {key}")
            return dispatch(key, operator, args, kwargs)
        finally:
            self.reset_dispatch_record()

    def redispatch(self, operator, args, kwargs):
        # Redispatch doesn't go to the top
        assert operator == self.currently_dispatching_op
        key = compute_dispatch_key(operator, args, kwargs, self.already_dispatched_keys)
        self.record_dispatch(key, operator)
        print(f"PyDispatcher.redispatch {key}")
        return dispatch(key, operator, args, kwargs)

    def reset_dispatch_record(self):
        self.current_dispatching_op = None
        self.already_dispatched_keys = None

    def record_dispatch(self, dispatch_key, operator):
        self.currently_dispatching_op = operator
        if self.already_dispatched_keys is None:
            self.already_dispatched_keys = DispatchKeySet(dispatch_key)
        else:
            self.already_dispatched_keys = self.already_dispatched_keys | DispatchKeySet(dispatch_key)


dispatcher_singleton = PyDispatcher()

class PyOperator:
    def __init__(self, name):
        self.name = name
        self.table = {}

        # TODO: torch_dispatch expects PyOperator to be an instance of a torch.ops.aten op.
        self.overloadpacket = self

        # Hack for FX tracing
        self.__name__ = f'torch.{name}'

    def impl(self, dispatch_key, fn):
        assert dispatch_key not in self.table
        self.table[dispatch_key] = fn

    def fallthrough(self, dispatch_key):
        assert dispatch_key not in self.table
        self.table[dispatch_key] = fallthrough_fn(self, dispatch_key)

    def __call__(self, *args, **kwargs):
        flat_args = to_flat_tuple(args, kwargs)
        if has_torch_function(flat_args):
            return handle_torch_function(self, flat_args, *args, **kwargs)

        return dispatcher_singleton.call(self, args, kwargs)


def compute_dispatch_key(PyOperator, args, kwargs, additional_exclude=None):
    tensors = get_tensors(args, kwargs)
    dispatch_key = key_extractor(tensors, additional_exclude)
    return dispatch_key


def dispatch(dispatch_key, operator, args, kwargs):
    print("Dispatching:", dispatch_key, operator.__name__)
    if dispatch_key not in SUPPORTED_KEYS:
        raise RuntimeError(f'NYI: {dispatch_key}')
    assert dispatch_key in operator.table
    kernel = operator.table[dispatch_key]
    return kernel(*args, **kwargs)


def key_extractor(tensors, additional_exclude=None):
    key_set = _C._dispatch_tls_local_include_set()
    for tensor in tensors:
        key_set = key_set | _C._dispatch_keys(tensor)
    key_set = key_set - _C._dispatch_tls_local_exclude_set()
    if additional_exclude is not None:
        key_set = key_set - additional_exclude
    return key_set.highestPriorityTypeId()


def to_flat_tuple(args, kwargs):
    flat_args, _ = tree_flatten(args)
    flat_kwargs, _ = tree_flatten(kwargs)
    flat_all = flat_args + flat_kwargs
    return flat_all

def get_tensors(args, kwargs):
    flat_all = to_flat_tuple(args, kwargs)
    tensor_args = [t for t in flat_all if isinstance(t, torch.Tensor)]
    return tuple(tensor_args)

"""
We're going to define a `cond` operation.
In order to do this, we need implementations for each of the dispatch keys.
"""

def cond_dense(pred, true_fn, false_fn, *operands):
    print("Running cond dense", pred)
    # print("Pred?", pred.code)
    # if pred(tuple()):
    if pred.elem:
        print("True!")
        x = true_fn(*operands)
        return x
    else:
        print("False!")
        x = false_fn(*operands)
        return x


def cond_autograd(pred, true_fn, false_fn, *operands):
    # TODO: support autograd
    flat_operands, _ = tree_flatten((true_fn, false_fn) + operands)
    assert all([not f.requires_grad for f in flat_operands
                if isinstance(f, torch.Tensor)])

    guard = ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.AutogradCPU))
    return cond(pred, true_fn, false_fn, *operands)


def cond_adinplaceorview(*args, **kwargs):
    guard = ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.ADInplaceOrView))
    return cond(*args, **kwargs)


def fallthrough_fn(operator, dispatch_key):
    def inner(*args, **kwargs):
        return dispatcher_singleton.redispatch(operator, args, kwargs)
    return inner


def python_fallback(op):
    def inner(*args, **kwargs):
        print("Input:", args)
        # Get all tensors. For each tensor, try their torch_dispatch
        # until one returns something other than NotImplemented
        def extract():
            tensors = get_tensors(args, kwargs)
            for tensor in tensors:
                # print("T:", tensor)
                ret = tensor.__torch_dispatch__(op, None, args, kwargs)
                if ret is NotImplemented:
                    continue
                return ret
            return NotImplemented

        mode = torch._C._get_torch_dispatch_mode()
        if mode is not None:
            with mode.restore():
                return extract()
        else:
            return cond_dense(*args)

    return inner


cond = PyOperator('cond')
cond.impl(DispatchKey.CPU, cond_dense)
cond.impl(DispatchKey.AutogradCPU, cond_autograd)
cond.fallthrough(DispatchKey.ADInplaceOrView)
cond.fallthrough(DispatchKey.BackendSelect)

cond.impl(DispatchKey.Python, python_fallback(cond))
cond.fallthrough(DispatchKey.PythonTLSSnapshot)


"""
Test case #1: basic
"""

def true_fn(x):
    return x.sin()

def false_fn(x):
    return x.cos()

x = torch.randn(4)
# result = cond(False, true_fn, false_fn, x)
# assert torch.allclose(result, torch.cos(x))

"""
Test case #2: tracing

NB: We need some additional way to add a new "lowering rule" for
lowering the cond call to an FX node. In particular,
cond accepts a true_fn/false_fn and these need to be traced out.

I've hardcoded the logic into ProxyTensor.
"""
print("EXAMPLE 2")
from torch.fx.experimental.proxy_tensor import make_fx

def f(x, y):
    return cond(y, true_fn, false_fn, x)

graph = make_fx(f)(x, torch.tensor(True))
print("graph.code:")
print(graph.code)
graph.graph.print_tabular()
print("Invoking:")
result_false = graph.forward(x, torch.tensor(True))
print("False:", result_false)
result_true = graph(x, torch.tensor(False))
print("True:", result_true)
# result_true()
# print(graph.forward())

exit(0)

"""
def forward(self, x_1, pred_1):
    _tensor_constant0 = self._tensor_constant0
    conditional = __main___torch_cond(False, wrapped(), wrapped(), _tensor_constant0);  _tensor_constant0 = None
    return conditional
    
opcode         name               target                                          args                                              kwargs
-------------  -----------------  ----------------------------------------------  ------------------------------------------------  --------
placeholder    x_1                x_1                                             ()                                                {}
placeholder    pred_1             pred_1                                          ()                                                {}
get_attr       _tensor_constant0  _tensor_constant0                               ()                                                {}
call_function  conditional        <__main__.PyOperator object at 0x7ff4b4480100>  (False, wrapped(), wrapped(), _tensor_constant0)  {}
output         output             output                                          (conditional,)                                    {}

"""

"""
Test case #3: tracing complex

NB: We need some additional way to add a new "lowering rule" for
lowering the cond call to an FX node. In particular,
cond accepts a true_fn/false_fn and these need to be traced out.

I've hardcoded the logic into ProxyTensor.
"""
from torch.fx.experimental.proxy_tensor import make_fx

def true_fn(x, pred2):
    def true_nested(y):
        return y * y 

    def false_nested(y):
        return y + y
        
    return cond(pred2, true_nested, false_nested, x.sin())

def false_fn(x, _):
    return x.cos()

def f(x, pred, pred2):
    return cond(pred, true_fn, false_fn, (x, pred2))

graph = make_fx(f)(x, False, True)
print("graph.code:")
print(graph.code)
graph.graph.print_tabular()

"""
def forward(self, x_1, pred_1, pred2_1):
    _tensor_constant0 = self._tensor_constant0
    conditional = __main___torch_cond(False, wrapped(), wrapped(), (_tensor_constant0, True));  _tensor_constant0 = None
    return conditional
    
opcode         name               target                                          args                                                      kwargs
-------------  -----------------  ----------------------------------------------  --------------------------------------------------------  --------
placeholder    x_1                x_1                                             ()                                                        {}
placeholder    pred_1             pred_1                                          ()                                                        {}
placeholder    pred2_1            pred2_1                                         ()                                                        {}
get_attr       _tensor_constant0  _tensor_constant0                               ()                                                        {}
call_function  conditional        <__main__.PyOperator object at 0x7ff4b4480100>  (False, wrapped(), wrapped(), (_tensor_constant0, True))  {}
output         output             output                                          (conditional,)                                            {}
"""
"""
More test cases (coming soon)

3. Autograd
4. functorch transforms!
"""
