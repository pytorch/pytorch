import torch
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
import torch._C as _C
from torch._C import DispatchKey, DispatchKeySet, ExcludeDispatchKeyGuard
from torch.overrides import handle_torch_function, has_torch_function
from torch.fx.experimental.proxy_tensor import make_fx

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
        self.current_key = None

    def call(self, operator, args, kwargs):
        try:
            key = compute_dispatch_key(operator, args, kwargs, self.current_key)
            self.record_dispatch(key, operator)
            print(f"PyDispatcher.call {key}")
            return dispatch(key, operator, args, kwargs)
        finally:
            self.reset_dispatch_record()

    def redispatch(self, operator, args, kwargs):
        # Redispatch doesn't go to the top
        assert operator == self.currently_dispatching_op
        key = compute_dispatch_key(operator, args, kwargs, self.current_key, self.already_dispatched_keys)
        self.record_dispatch(key, operator)
        print(f"PyDispatcher.redispatch {key}")
        return dispatch(key, operator, args, kwargs)

    def reset_dispatch_record(self):
        self.current_dispatching_op = None
        self.already_dispatched_keys = None

    def record_dispatch(self, dispatch_key, operator):
        self.currently_dispatching_op = operator
        self.current_key = dispatch_key
        if self.already_dispatched_keys is None:
            self.already_dispatched_keys = DispatchKeySet(dispatch_key)
        else:
            self.already_dispatched_keys = self.already_dispatched_keys | DispatchKeySet(dispatch_key)


dispatcher_singleton = PyDispatcher()

class PyOperator:
    def __init__(self, name):
        self.name = name
        self.table = {}
        self.entrance_rules = {}

        # TODO: torch_dispatch expects PyOperator to be an instance of a torch.ops.aten op.
        self.overloadpacket = self

        # Hack for FX tracing
        self.__name__ = f'torch.{name}'

    def impl(self, dispatch_key, fn, reentrant=False):
        assert dispatch_key not in self.table
        self.table[dispatch_key] = fn
        self.entrance_rules[dispatch_key] = reentrant

    def fallthrough(self, dispatch_key):
        assert dispatch_key not in self.table
        self.table[dispatch_key] = fallthrough_fn(self, dispatch_key)
        self.entrance_rules[dispatch_key] = False

    def __call__(self, *args, **kwargs):
        flat_args = to_flat_tuple(args, kwargs)
        if has_torch_function(flat_args):
            return handle_torch_function(self, flat_args, *args, **kwargs)

        return dispatcher_singleton.call(self, args, kwargs)


def compute_dispatch_key(operator, args, kwargs, current_key, additional_exclude=None):
    if current_key is not None and operator.entrance_rules[current_key]:
        return current_key

    tensors = get_tensors(args, kwargs)
    dispatch_key = key_extractor(tensors, additional_exclude)
    return dispatch_key


def dispatch(dispatch_key, operator, args, kwargs):
    if dispatch_key not in SUPPORTED_KEYS:
        raise RuntimeError(f'NYI: {dispatch_key} {SUPPORTED_KEYS}')
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
from contextlib import contextmanager

@contextmanager
def suspend_mode(mode):
    torch._C._set_torch_dispatch_mode(None)
    yield
    torch._C._set_torch_dispatch_mode(mode)


def cond_dense(pred, true_fn, false_fn, operands):
    mode = torch._C._get_torch_dispatch_mode()
    if mode:
        with suspend_mode(mode):
            args = (pred, true_fn, false_fn, operands)
            return mode.__torch_dispatch__(cond, None, args, {})
    try:
        if pred:
            return true_fn(operands)
        else:
            return false_fn(operands)
    except Exception as e:
        # Do something proper here, someday
        print("Exception", e)



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
        mode = torch._C._get_torch_dispatch_mode()

        if mode:
            with suspend_mode(mode):
                return mode.__torch_dispatch__(op, None, args, kwargs)
        else:
            return cond_dense(*args)

    return inner


cond = PyOperator('cond')
cond.impl(DispatchKey.CPU, cond_dense, True)
cond.impl(DispatchKey.AutogradCPU, cond_autograd)
cond.fallthrough(DispatchKey.ADInplaceOrView)
cond.fallthrough(DispatchKey.BackendSelect)

cond.impl(DispatchKey.Python, python_fallback(cond))
cond.fallthrough(DispatchKey.PythonTLSSnapshot)


"""
Test case #1: basic
# """
print("EXAMPLE 1")

def true_fn(x):
    return x.sin()

def false_fn(x):
    return x.cos()

x = torch.randn(4)
result = cond(False, true_fn, false_fn, x)
assert torch.allclose(result, torch.cos(x))

"""
Test case #2: tracing
"""
print("EXAMPLE 2")

def f(x, y):
    return cond(y, true_fn, false_fn, x)

graph = make_fx(f)(x, torch.tensor(False))
result_true = graph.forward(x, torch.tensor(True))
result_false = graph.forward(x, torch.tensor(False))
assert not torch.allclose(result_true, result_false)
assert torch.allclose(result_true, torch.sin(x))
assert torch.allclose(result_false, torch.cos(x))

"""
def forward(self, x_1, y_1):
    true_graph = self.true_graph
    false_graph = self.false_graph
    conditional = __main___torch_cond(y_1, true_graph, false_graph, x_1);  y_1 = true_graph = false_graph = x_1 = None
    return conditional

opcode         name         target                                          args                                 kwargs
-------------  -----------  ----------------------------------------------  -----------------------------------  --------
placeholder    x_1          x_1                                             ()                                   {}
placeholder    y_1          y_1                                             ()                                   {}
get_attr       true_graph   true_graph                                      ()                                   {}
get_attr       false_graph  false_graph                                     ()                                   {}
call_function  conditional  <__main__.PyOperator object at 0x7f2b61df5100>  (y_1, true_graph, false_graph, x_1)  {}
output         output       output                                          (conditional,)                       {}

"""

"""
Test case #3: tracing complex/nested

I've hardcoded the logic into ProxyTensor.
"""
print("EXAMPLE 3")

def true_nested(y):
    return y * y

def false_nested(y):
    return y + y

def true_fn(x, pred2):
    return cond(pred2, true_nested, false_nested, x)

def false_fn(x, _):
    return x.cos()

def f(x, pred, pred2):
    return cond(pred, true_fn, false_fn, (x, pred2))

graph = make_fx(f)(x, torch.tensor(False), torch.tensor(False))

result_true_true = graph.forward(x, torch.tensor(True), torch.tensor(True)) # True + True -> x * x
result_true_false = graph.forward(x, torch.tensor(True), torch.tensor(False)) # True + True -> x + x
result_false_true = graph.forward(x, torch.tensor(False), torch.tensor(True)) #  False + either -> cos
result_false_false = graph.forward(x, torch.tensor(False), torch.tensor(False)) #  False + either -> cos


assert not torch.allclose(result_true_true, result_true_false)
assert not torch.allclose(result_false_true, result_true_true)

assert torch.allclose(result_false_true, result_false_false)

assert torch.allclose(result_true_true, x * x)
assert torch.allclose(result_true_false, x + x)

assert torch.allclose(result_false_true, torch.cos(x))

print("Done, all tests passed")

"""
def forward(self, x_1, pred_1, pred2_1):
    true_graph = self.true_graph
    false_graph = self.false_graph
    conditional = __main___torch_cond(False, true_graph, false_graph, [(x_1, True)]);  true_graph = false_graph = x_1 = None
    return conditional

opcode         name         target                                          args                                             kwargs
-------------  -----------  ----------------------------------------------  -----------------------------------------------  --------
placeholder    x_1          x_1                                             ()                                               {}
placeholder    pred_1       pred_1                                          ()                                               {}
placeholder    pred2_1      pred2_1                                         ()                                               {}
get_attr       true_graph   true_graph                                      ()                                               {}
get_attr       false_graph  false_graph                                     ()                                               {}
call_function  conditional  <__main__.PyOperator object at 0x7f717b617100>  (False, true_graph, false_graph, [(x_1, True)])  {}
output         output       output                                          (conditional,)                                   {}                                         {}
"""
"""
More test cases (coming soon)

3. Autograd
4. functorch transforms!
"""
