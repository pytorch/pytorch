import torch
import torch.utils._pytree as pytree
from torch.testing._internal.common_methods_invocations import wrapper_set_seed
from functorch.compile import compiled_function, min_cut_rematerialization_partition, nop
import re
import contextlib

@contextlib.contextmanager
def assert_raises_regex(exception_cls, regex):
    try:
        yield
    except exception_cls as e:
        msg = str(e)
        if not re.search(regex, msg):
            raise AssertionError(
                f"Expected exception to match regex. regex: {regex}, exception: {msg}")
    except Exception as e:
        raise AssertionError(
            f"Expected {exception_cls} to be raised, instead got exception {type(e)}")
    finally:
        pass
    raise AssertionError("Expected exception to be raised but none was")

def aot_autograd_check(
        func, args, kwargs, dynamic,
        assert_raises_regex_fn=assert_raises_regex,
        assert_equals_fn=torch.testing._comparison.assert_close):
    flat_args, args_spec = pytree.tree_flatten((args, kwargs))
    for arg in flat_args:
        if isinstance(arg, int) and arg == -42:
            raise TestFrameworkError("We've reserved -42 as an arg")

    sentinel_val = -42
    is_tensor_spec = [sentinel_val if isinstance(arg, torch.Tensor) else arg for arg in flat_args]
    args = [arg for arg in flat_args if isinstance(arg, torch.Tensor)]

    def f(args):
        cur_flat_args = list(is_tensor_spec)
        args = iter(args)
        for idx, v in enumerate(cur_flat_args):
            if v == sentinel_val:
                cur_flat_args[idx] = next(args)
        c_args, c_kwargs = pytree.tree_unflatten(cur_flat_args, args_spec)
        return func(*c_args, **c_kwargs)

    compiled_f = compiled_function(f, nop, nop, dynamic=dynamic, partition_fn=min_cut_rematerialization_partition)
    _test_aot_autograd_forwards_backwards_helper(
        f, compiled_f, args, assert_raises_regex_fn, assert_equals_fn)


def _test_aot_autograd_forwards_backwards_helper(
        f, compiled_f, args, assert_raises_regex_fn, assert_equals_fn):
    # Verify grads are equal between compiled and non-compiled versions of f.

    def call_forwards_backwards(f):
        out = wrapper_set_seed(f, args)
        if not isinstance(out, torch.Tensor):
            flat_out, _ = pytree.tree_flatten(out)
            sm = 0
            for i in flat_out:
                sm += i.sum().abs()
            sm.backward()
        else:
            out.sum().abs().backward()

    def reset_grads():
        def f(x):
            x.grad = None
        pytree.tree_map(f, args)

    def get_grads(args):
        return pytree.tree_map(lambda x: x.grad, args)

    reset_grads()
    call_forwards_backwards(f)
    orig_grad = get_grads(args)

    reset_grads()
    # See https://github.com/pytorch/pytorch/pull/98960#issuecomment-1505962215
    if all(x is None for x in orig_grad):
        with assert_raises_regex_fn(RuntimeError, 'does not require grad and does not have a grad_fn'):
            call_forwards_backwards(compiled_f)
    else:
        call_forwards_backwards(compiled_f)
        compiled_grad = get_grads(args)
        assert_equals_fn(orig_grad, compiled_grad)

    def create_new_arg(x):
        if isinstance(x, torch.Tensor) and x.dtype == torch.float32:
            return x.detach().uniform_(0, 1).requires_grad_(x.requires_grad)
        return x

    args = pytree.tree_map(create_new_arg, args)

    reset_grads()
    call_forwards_backwards(f)
    orig_grad = get_grads(args)

    reset_grads()
    # See https://github.com/pytorch/pytorch/pull/98960#issuecomment-1505962215
    if all(x is None for x in orig_grad):
        with assert_raises_regex_fn(RuntimeError, 'does not require grad and does not have a grad_fn'):
            call_forwards_backwards(compiled_f)
    else:
        call_forwards_backwards(compiled_f)
        compiled_grad = get_grads(args)
        assert_equals_fn(orig_grad, compiled_grad)
