import torch
import torch.utils._pytree as pytree
from torch.testing._internal.common_methods_invocations import wrapper_set_seed
from functorch.compile import compiled_function, min_cut_rematerialization_partition, nop
from .make_fx import randomize
import re


class assert_raises_regex:
    def __init__(self, exception_cls, regex):
        self.exception_cls = exception_cls
        self.regex = regex

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, traceback):
        if exc_type == self.exception_cls:
            msg = str(exc_val)
            if not re.search(self.regex, msg):
                raise AssertionError(
                    f"Expected exception to match regex. regex: {self.regex}, exception: {msg}")
            return True  # Squashes the exception
        if exc_type is not None:
            raise AssertionError(
                f"Expected {self.exception_cls} to be raised, instead got exception {exc_type}")
        raise AssertionError("Expected exception to be raised but none was")


def aot_autograd_check(
        func,
        args,
        kwargs,
        dynamic,
        assert_raises_regex_fn=assert_raises_regex,
        assert_equals_fn=torch.testing._comparison.assert_close,
        check_gradients=True,
        try_check_data_specialization=False):
    """Compares func(*args, **kwargs) in eager-mode to under AOTAutograd.

    Compares outputs and (if check_gradients=True) gradients produced by
    AOTAutograd against eager-mode PyTorch.

    We assume that func(*args, **kwargs) succeeds in eager-mode PyTorch.

    """
    flat_args, args_spec = pytree.tree_flatten((args, kwargs))
    args_is_tensor = [isinstance(arg, torch.Tensor) for arg in flat_args]
    args = [arg for arg in flat_args if isinstance(arg, torch.Tensor)]

    # We construct a new function that only accepts Tensors as inputs
    def func_no_tensors(args):
        reconstructed_flat_args = []
        args = iter(args)
        for idx, v in enumerate(flat_args):
            if isinstance(v, torch.Tensor):
                reconstructed_flat_args.append(next(args))
            else:
                reconstructed_flat_args.append(v)

        c_args, c_kwargs = pytree.tree_unflatten(reconstructed_flat_args, args_spec)
        return func(*c_args, **c_kwargs)

    compiled_f = compiled_function(
        func_no_tensors, nop, nop, dynamic=dynamic, partition_fn=min_cut_rematerialization_partition)

    out = wrapper_set_seed(func_no_tensors, args)
    if check_gradients == "auto":
        any_tensor_requires_grad = pytree.tree_any_only(torch.Tensor, lambda x: x.requires_grad, args)
        any_output_requires_grad = pytree.tree_any_only(torch.Tensor, lambda x: x.requires_grad, out)
        check_gradients = any_tensor_requires_grad and any_output_requires_grad
    if not check_gradients:
        compiled_out = wrapper_set_seed(compiled_f, args)
        assert_equals_fn(compiled_out, out, msg=outputs_msg)
        return
    _test_aot_autograd_forwards_backwards_helper(
        func_no_tensors, compiled_f, args, assert_raises_regex_fn, assert_equals_fn,
        try_check_data_specialization)

outputs_msg = (
    "Outputs of the operator are different in eager-mode PyTorch vs "
    "AOTAutograd. This means the operator will have incorrect output "
    "underneath torch.compile. This could be because the operator's "
    "implementation not traceable or that there is a bug in AOTAutograd."
)


def _test_aot_autograd_forwards_backwards_helper(
        f, compiled_f, args, assert_raises_regex_fn, assert_equals_fn,
        try_check_data_specialization):
    # Verify grads are equal between compiled and non-compiled versions of f.

    def call_forwards_backwards(f, args):
        flat_args, _ = pytree.tree_flatten(args)
        diff_args = [arg for arg in flat_args if isinstance(arg, torch.Tensor) and
                     arg.requires_grad]
        out = wrapper_set_seed(f, args)
        flat_out, _ = pytree.tree_flatten(out)

        sm = 0
        for i in flat_out:
            if isinstance(i, torch.Tensor):
                # We need to call .abs() because it is possible that the output of the
                # operator is a complex Tensor and autograd will yell at autograd.grad
                # on a complex Tensor unless we manually provide the grad_output flag.
                sm += i.sum().abs()
        assert isinstance(sm, torch.Tensor)
        return out, torch.autograd.grad(sm, diff_args, allow_unused=True)

    def check(args, ignore_failure=False):
        try:
            orig_out, orig_grad = call_forwards_backwards(f, args)
        except Exception:
            if ignore_failure:
                return
            raise

        # See https://github.com/pytorch/pytorch/pull/98960#issuecomment-1505962215
        if all(x is None for x in orig_grad):
            with assert_raises_regex_fn(RuntimeError, 'does not require grad and does not have a grad_fn'):
                call_forwards_backwards(compiled_f, args)
            return

        msg = (
            "Gradients of the operator are different in eager-mode PyTorch vs "
            "AOTAutograd. This means the operator will have incorrect gradients "
            "underneath torch.compile. This could be because the operator's "
            "backward is incorrectly registered or not traceable or that there "
            "is a bug in AOTAutograd."
        )

        compiled_out, compiled_grad = call_forwards_backwards(compiled_f, args)
        assert_equals_fn(compiled_out, orig_out, msg=outputs_msg)
        assert_equals_fn(compiled_grad, orig_grad, msg=msg)

    check(args, ignore_failure=False)

    # Randomize the data and run the traced graph with it, to catch bugs
    # where we may have baked in Tensor data into the trace.
    # This is not guaranteed to succeed, because `f` might have preconditions
    # on the values of the inputs, so we just ignore if this test fails.
    if try_check_data_specialization:
        args = randomize(args)
        check(args, ignore_failure=True)
