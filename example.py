import torch
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from torch.utils._python_dispatch import enable_python_mode
import functools
from functools import partial
from typing import Optional
import contextlib

supported_ops = {
    torch.ops.aten.zeros,
    torch.ops.aten.mul,
    torch.ops.aten.select,
    torch.ops.aten.randn,
    torch.ops.aten.sum,
    torch.ops.aten.sin,
    torch.ops.aten.cos,
    torch.ops.aten.add,
    torch.ops.aten.stack,
    torch.ops.aten.to,
    torch.ops.aten.detach,
    torch.ops.aten.expand,
    torch.ops.aten.eye,
    torch.ops.aten.neg,
    torch.ops.aten.exp,
    torch.ops.aten.permute,
    torch.ops.aten.detach_,
}

# Part 1: The mode stack.
# The mode stack lives at a PythonMode dispatch key that is at the very beginning of
# all dispatch keys! It is separate from the Python dispatch key, which lives after
# autograd.
#
# The mode stack holds (subclass type object, mode instance) objects.
# The subclass is the subclass of the current transform (each transform dynamically
# allocates a subclass) while the mode instance holds metadata associated with
# the subclass.
#
# Whenever something is in the mode stack, we unconditionally dispatch to the
# subclass's __torch_dispatch__.
#
# TODO: Could we just save the metadata on the subclass type object?
# TODO: Does dispatchkey::PythonMode need to be separte from dispatchkey::Python
def mode_stack_size():
    return torch._C._autograd._mode_stack_size()

def pop_mode_stack():
    assert mode_stack_size() > 0
    return torch._C._autograd._exit_python_mode()

def push_mode_stack(subclass, mode):
    return torch._C._autograd._enter_python_mode(subclass, mode)

def mode_stack_top():
    assert mode_stack_size() > 0
    result = pop_mode_stack()
    push_mode_stack(*result)
    return result

@contextlib.contextmanager
def temporarily_pop_mode_stack():
    assert mode_stack_size() > 0
    subclass, mode = pop_mode_stack()
    try:
        yield subclass, mode
    finally:
        push_mode_stack(subclass, mode)

@contextlib.contextmanager
def noop():
    yield None, None

def batched_fallback(func, subclass, args, kwargs):
    if not func in supported_ops:
        raise RuntimeError(f"not supported: {func.__name__}")

    # TODO: assumes there are no Tensors in kwargs
    flat_args, _ = tree_flatten(args)
    if not any([isinstance(e, subclass) for e in flat_args]):
        return func(*args)

    # Naive batching rule: for-loop + stack
    bdim_size = None
    for e in flat_args:
        if isinstance(e, subclass):
            bdim_size = e.elem.size(e.bdim)

    def get_slice(idx, e):
        return e.elem.select(e.bdim, idx) if isinstance(e, subclass) else e

    results = []
    for i in range(bdim_size):
        sliced_args = tree_map(partial(get_slice, i), args)
        res = func(*sliced_args, **kwargs)
        assert isinstance(res, torch.Tensor)
        results.append(res)

    result = torch.stack(results)
    return subclass(result, 0)

def gen_subclass(cls, parent=None):
    class Generated(cls):
        __torch_base__ = cls
        __torch_parent__ = parent
        pass
    return Generated

def is_descendant(parent, descendant):
    assert issubclass(parent, torch.Tensor)
    assert issubclass(descendant, torch.Tensor)
    if type(parent) == torch.Tensor:
        return True
    while getattr(descendant, '__torch_parent__', None) is not None:
        if parent == descendant.__torch_parent__:
            return True
        descendant = descendant.__torch_parent__
    return False

def supports_torch_parent_priority(fn):
    @functools.wraps(fn)
    def wrapped(cls, func, types, args, kwargs):
        other_types = [t for t in types if t != cls and issubclass(t, torch.Tensor)]
        if len(other_types) > 0 and not all(is_descendant(t, cls) for t in other_types):
            return NotImplemented
        return fn(cls, func, types, args, kwargs)
    return wrapped

# Part 2: the BatchedTensor object
# There is a base BatchedTensor object that all vmap transforms dynamically subclass.
# It has a torch_dispatch which basically calls a "batched fallback" in lieu of
# having batching rules.
#
# There is also special behavior for randomness: when someone calls torch.randn,
# we check the mode instance for some metadata (the batch_size and the randomness
# behavior)
#
class BatchedTensor(torch.Tensor):
    elem: torch.Tensor
    __torch_function__ = torch._C._disabled_torch_function_impl

    __slots__ = ['elem', 'bdim']

    @staticmethod
    def __new__(cls, elem, bdim, *args, **kwargs):
        r = torch.Tensor._make_subclass(cls, elem.to('cpu')[0], elem.requires_grad)
        assert is_descendant(type(elem), cls)  # the thing you hold must be a __torch_parent__
        r.elem = elem
        r.bdim = bdim
        r.current_subclass = BatchedTensor
        return r

    def __repr__(self):
        return f"BatchedTensor({self.elem}, {self.bdim})"

    def unwrap_batched(self, out_dim=0):
        return self.elem.movedim(self.bdim, 0)

    @classmethod
    @supports_torch_parent_priority
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        ctx = temporarily_pop_mode_stack if mode_stack_size() > 0 and cls == mode_stack_top()[0] else noop
        with ctx() as (mode_subclass, mode):
            assert mode_subclass is None or cls == mode_subclass

            if func == torch.ops.aten.randn and mode is not None:
                if mode.randomness == 'error':
                    raise RuntimeError("No randomness allowed")
                if mode.randomness == 'same':
                    return func(*args, **kwargs)
                if mode.randomness == 'different':
                    args = list(args)
                    args[0] = [mode.batch_size] + args[0]
                    return cls(func(*args, **kwargs), 0)

            return batched_fallback(func, cls, args, kwargs)

# Part 3: VmapMode and functional vmap
#
# VmapMode creates a new BatchedTensor subclass for use and stores some
# metadata (batch_size, randomness setting).
#
# On entry it pushes something onto the mode stack; on exit it removes something.
#
# The vmap API is a wrapper around VmapMode.

class VmapMode():
    def __init__(self, batch_size, randomness, parent=None):
        self.batch_size = batch_size
        self.randomness = randomness
        if parent is None:
            parent = mode_stack_top()[0] if mode_stack_size() > 0 else torch.Tensor
        self.parent = parent

    def __enter__(self):
        subclass = gen_subclass(BatchedTensor, self.parent)
        push_mode_stack(subclass, self)
        return subclass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pop_mode_stack()

# randomness = {"error", "same", "different"}
def vmap(f, in_dims=(0,), randomness="error", parent=None):
    def wrapped(*args):
        batch_sizes = [arg.size(in_dim) for in_dim, arg in zip(in_dims, args) if in_dim is not None]
        batch_size = batch_sizes[0]

        with VmapMode(batch_size, randomness, parent) as GenBatchedTensor:
            def wrap(e, in_dim):
                assert in_dim is None or in_dim == 0
                if in_dim is None:
                    return e
                return GenBatchedTensor(e, in_dim)

            batched_args = tuple(wrap(arg, in_dim) for arg, in_dim in zip(args, in_dims))
            batched_out = f(*batched_args)
            assert isinstance(batched_out, torch.Tensor)

        if isinstance(batched_out, GenBatchedTensor):
            return batched_out.elem
        else:
            return batched_out.expand(batch_size, *batched_out.shape)
    return wrapped

# Part 4: JVP transform
#
# We define a JVPTensor, a JVPMode (which holds no metadata), and
# JVP formulas.

def opt_apply(lamb, optional):
    if optional is None:
        return optional
    return lamb(optional)

def opt_mul(x, y):
    if x is None or y is None:
        return None
    return torch.mul(x, y)

def opt_add(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return torch.add(x, y)

def sin_jvp_rule(unpacked_jvp_tensor):
    x_primal, x_tangent = unpacked_jvp_tensor
    return (x_primal.sin(), opt_mul(x_primal.cos(), x_tangent)),

def cos_jvp_rule(unpacked_jvp_tensor):
    x_primal, x_tangent = unpacked_jvp_tensor
    return (x_primal.cos(), opt_mul(-x_primal.sin(), x_tangent)),

# Part 4.5
# NB: You can call transforms INSIDE rules !!!! because of how the mode stack works.
#
# This means that decomposition/functionalization could be transforms that
# are called inside of e.g. batching rules
def exp_jvp_rule(unpacked_jvp_tensor):
    x_primal, x_tangent = unpacked_jvp_tensor
    assert x_primal.dim() > 0  # To demonstrate vmap inside jvp
    result = vmap(torch.exp)(x_primal)
    return (result, opt_mul(x_primal.exp(), x_tangent)),

def neg_jvp_rule(unpacked_jvp_tensor):
    x_primal, x_tangent = unpacked_jvp_tensor
    return (-x_primal, opt_apply(lambda x: -x, x_tangent)),

def sum_jvp_rule(unpacked_jvp_tensor):
    x_primal, x_tangent = unpacked_jvp_tensor
    return (x_primal.sum(), opt_apply(torch.sum, x_tangent)),

def to_jvp_rule(unpacked_jvp_tensor, *args, **kwargs):
    x_primal, x_tangent = unpacked_jvp_tensor
    return (torch.ops.aten.to(x_primal, *args, **kwargs),
            opt_apply(lambda x: torch.ops.aten.to(x, *args, **kwargs), x_tangent)),

def detach_jvp_rule(unpacked_jvp_tensor):
    x_primal, x_tangent = unpacked_jvp_tensor
    return (opt_apply(torch.Tensor.detach, x_primal), opt_apply(torch.Tensor.detach, x_tangent)),

def mul_jvp_rule(unpacked_x, unpacked_y):
    x_primal, x_tangent = unpacked_x
    y_primal, y_tangent = unpacked_y

    return (x_primal * y_primal, opt_add(opt_mul(x_primal, y_tangent), opt_mul(y_primal, x_tangent))),

def add_jvp_rule(unpacked_x, unpacked_y):
    x_primal, x_tangent = unpacked_x
    y_primal, y_tangent = unpacked_y

    return (x_primal + y_primal, opt_add(x_tangent, y_tangent)),

def stack_jvp_rule(unpacked, dim=0):
    primals, tangents = zip(*unpacked)
    return (torch.stack(primals, dim), torch.stack(tangents, dim)),

def select_jvp_rule(unpacked, dim, index):
    primal, tangent = unpacked
    return (primal.select(dim, index), tangent.select(dim, index)),

jvp_rules = {
    torch.ops.aten.sin: sin_jvp_rule,
    torch.ops.aten.cos: cos_jvp_rule,
    torch.ops.aten.mul: mul_jvp_rule,
    torch.ops.aten.add: add_jvp_rule,
    torch.ops.aten.stack: stack_jvp_rule,
    torch.ops.aten.select: select_jvp_rule,
    torch.ops.aten.neg: neg_jvp_rule,
    torch.ops.aten.detach: detach_jvp_rule,
    torch.ops.aten.to: to_jvp_rule,
    torch.ops.aten.sum: sum_jvp_rule,
    torch.ops.aten.exp: exp_jvp_rule,
}

class JVPTensor(torch.Tensor):
    primal: torch.Tensor
    tangent: Optional[torch.Tensor]
    __torch_function__ = torch._C._disabled_torch_function_impl

    __slots__ = ['primal', 'tangent']

    @staticmethod
    def __new__(cls, primal, tangent, *args, **kwargs):
        r = torch.Tensor._make_subclass(cls, primal, False)
        assert is_descendant(type(primal), cls)  # the thing you hold must be a __torch_parent__
        assert is_descendant(type(tangent), cls)  # the thing you hold must be a __torch_parent__
        r.primal = primal
        r.tangent = tangent
        return r

    def __repr__(self):
        return f"JVPTensor({self.primal}, {self.tangent})"

    @classmethod
    @supports_torch_parent_priority
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        ctx = temporarily_pop_mode_stack if mode_stack_size() > 0 and cls == mode_stack_top()[0] else noop
        with ctx() as (mode_subclass, mode):
            assert mode_subclass is None or cls == mode_subclass

            flat_args, _ = tree_flatten(args)
            flat_kwargs, _ = tree_flatten(kwargs)
            if not any([isinstance(e, cls) for e in flat_args]) and \
                not any([isinstance(e, cls) for e in flat_kwargs]):
                return func(*args)

            def unpack(x):
                if isinstance(x, cls):
                    return (x.primal, x.tangent)
                if isinstance(x, torch.Tensor):
                    return (x, None)
                return x

            unpacked_args = tree_map(unpack, args)
            unpacked_kwargs = tree_map(unpack, kwargs)

            if func not in jvp_rules:
                raise RuntimeError(f"jvp not supported: {func.__name__}")

            # NB: we only support functions that return Tensors...
            rule = jvp_rules[func]
            output = rule(*unpacked_args, **unpacked_kwargs)
            result = tuple(cls(primal, tangent) for primal, tangent in output)
            # hack
            if len(result) == 1:
                return result[0]
            return result

# Nothing interesting here....
class JVPMode():
    def __init__(self, parent=None):
        if parent is None:
            parent = mode_stack_top()[0] if mode_stack_size() > 0 else torch.Tensor
        self.parent = parent

    def __enter__(self):
        subclass = gen_subclass(JVPTensor, self.parent)
        push_mode_stack(subclass, self)
        return subclass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pop_mode_stack()

# NB: Yes, jvp is NOT a higher-order function.
def jvp(f, primals, tangents, parent=None):
    with JVPMode(parent) as GenJVPTensor:
        with temporarily_pop_mode_stack():
            jvp_tensors = tuple(GenJVPTensor(primal, tangent) for primal, tangent in zip(primals, tangents))
        results = f(*jvp_tensors)

    def unwrap(x):
        if isinstance(x, GenJVPTensor):
            return x.primal, x.tangent
        return x

    # Assumes f is a flat tuple (not a pytree)
    results, spec = tree_flatten(results)
    results = tree_map(unwrap, results)
    primals_out, tangents_out = zip(*results)
    return tree_unflatten(primals_out, spec), tree_unflatten(tangents_out, spec)

# ============================================================================
#
#                   Part 5: Transform API tests
#
# ============================================================================


# basic vmap test
x = torch.randn(3, 2, 5, 7)
y = vmap(vmap(vmap(torch.sum)))(x)
assert torch.allclose(y, x.sum([-1]))

# complicated vmap test
x = torch.arange(3)
y = torch.arange(4)
z = vmap(vmap(torch.mul, (0, None)), (None, 0))(x, y)
assert torch.allclose(z, y.unsqueeze(-1) * x)

# vmap mode test
def foo(x):
    return torch.randn(1)
y = vmap(foo, randomness='same')(torch.ones(3))
assert torch.allclose(y - y[0], torch.zeros_like(y))
z = vmap(foo, randomness='different')(torch.ones(3))
assert not torch.allclose(z - z[0], torch.zeros_like(z))

# basic JVP test
x = torch.randn([])
t = torch.ones([])
y, y_t = jvp(torch.sin, (x,), (t,))
assert torch.allclose(y, x.sin())
assert torch.allclose(y_t, x.cos())

# nested JVP test
def derivative(f):
    def wrapped(x):
        primal_out, tangent_out = jvp(f, (x,), (torch.ones([]),))
        return tangent_out
    return wrapped

x = torch.tensor(0.4)
y = derivative(torch.sin)(x)
assert torch.allclose(y, x.cos())
z = derivative(derivative(torch.sin))(x)
assert torch.allclose(z, -x.sin())

# compose vmap and jvp
def jacfwd(f):
    # TODO: This should take more than just a single primal...
    def wrapper_fn(primal):
        assert primal.dim() == 1
        basis = torch.eye(primal.numel(), dtype=primal.dtype, device=primal.device)

        def push_jvp(basis):
            _, jvp_out = jvp(f, (primal,), (basis,))
            return jvp_out

        result = vmap(push_jvp)(basis)
        # result = result.view(*primal.shape, *primal.shape)
        return result
    return wrapper_fn

x = torch.randn(3)
res = jacfwd(torch.sin)(x)
assert torch.allclose(res, torch.diagflat(torch.cos(x)))

# more composition
def foo(x):
    return x.sin().sum()

res = jacfwd(jacfwd(foo))(x)
assert torch.allclose(res, torch.diagflat(-torch.sin(x)))

# test for calling vmap inside a jvp rule (see exp_jvp_rule)
x = torch.randn(1)
t = torch.ones(1)
p_out, t_out = jvp(torch.exp, (x,), (t,))
assert torch.allclose(t_out, x.exp())


# ============================================================================
#
#                   Part 6: Imperative API Tests
#
# ============================================================================

def make_batched_tensor_class(parent=None):
    if parent is None:
        if mode_stack_size() > 0:
            parent = mode_stack_top()[0]
        else:
            parent = torch.Tensor
    return gen_subclass(BatchedTensor, parent)

def make_jvp_tensor_class(parent=None):
    if parent is None:
        if mode_stack_size() > 0:
            parent = mode_stack_top()[0]
        else:
            parent = torch.Tensor
    return gen_subclass(JVPTensor, parent)

# Tensor subclasses are only allowed to hold instances of Tensor subclasses that
# are its __torch_parent__ ancestors.
BatchedTensor0 = make_batched_tensor_class()
BatchedTensor1 = make_batched_tensor_class(parent=BatchedTensor0)

a, b = torch.arange(5), torch.arange(3)
x = BatchedTensor0(a, 0)
y = BatchedTensor1(b, 0)
z = x * y
output = z.unwrap_batched().unwrap_batched()
assert torch.allclose(output, a.unsqueeze(-1) * b)

# Jacobian computation
BatchedTensor0 = make_batched_tensor_class()
JVPTensor1 = make_jvp_tensor_class(parent=BatchedTensor0)
x = torch.randn(5)
v = torch.eye(5)
dual = JVPTensor1(x, BatchedTensor0(v, 0))
out_dual = dual.sin()
result = out_dual.tangent.unwrap_batched()
assert torch.allclose(result, torch.diag(x.cos()))

# ============================================================================
#
#              Part 7: Mixing imperative and transform APIs
#
# ============================================================================

# After writing the following cases, my gut is telling me we might want to
# restrict mixing of imperative and transform APIs.
#
# Here's what we might want to allow:
# - passing Tensor subclasses into a transformed function. The transformed
# function should not create and manipulate Tensor subclasses.
# Concretely, the following should be OK:
# >>> vmap(foo)(MaskedTensor(...))
#
# Here's what we might not want to allow:
# - transforming a function that creates and manipulates Tensor subclasses.
# Concretely, this would mean that we're not going to let someone make a
# MaskedTensor where the value or mask is a BatchedTensor created from inside
# vmap. If they want to do something like this, then they need to compose
# vmap and mask transforms or compose BatchedTensor and MaskedTensor (from the
# imperative APIs).
#
# This might suggest that transforms should always go first and the imperative
# APIs should run after transforms. But where do modes fit into this?
#
# TODO: Is there a difference between a "mode" and a "transform"? The gut says:
# you probably want to be able to LoggingTensorMode at different levels, so
# "modes" are a part of the imperative API. Does that mean we want separate
# transform and mode stacks?

# Jacobian computation.
# NB: I don't like how the `jvp` transform needs to be passed a parent arg.
# We can get over this restriction by saying that transforms always run before
# the imperative APIs.
BatchedTensor0 = make_batched_tensor_class()
x = torch.randn(5)
v = torch.eye(5)
primal, tangent = jvp(torch.sin, (x,), (BatchedTensor0(v, 0),), parent=BatchedTensor0)
assert torch.allclose(tangent.unwrap_batched(), torch.diag(x.cos()))


# Creating a JVPTensor inside of vmap!
# Per-sample-grad example
# NB: The problem I have with this example is that if someone returns the JVPTensor
# instead of unwrapping it inside the function, then a BatchedTensor will escape.
# The user didn't create that BatchedTensor (vmap did!) so they're going to be
# confused when they see it.
def grad_of_sin(x):
    JVPTensor0 = make_jvp_tensor_class()
    assert x.shape == ()
    v = torch.tensor(1.0)
    dual = JVPTensor0(x, v)
    out_dual = dual.sin()
    return out_dual.tangent

x = torch.randn(5)
grads = vmap(grad_of_sin)(x)
assert torch.allclose(grads, x.cos())
