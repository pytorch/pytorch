# mypy: ignore-errors

import torch
import torch.utils._pytree as pytree
from torch.testing._utils import wrapper_set_seed
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
        assert_equals_fn=torch.testing.assert_close,
        check_gradients=True,
        try_check_data_specialization=False,
        skip_correctness_check=False):
    """Compares func(*args, **kwargs) in eager-mode to under AOTAutograd.

    Compares outputs and (if check_gradients=True) gradients produced by
    AOTAutograd against eager-mode PyTorch.

    We assume that func(*args, **kwargs) succeeds in eager-mode PyTorch.

    """
    flat_args, args_spec = pytree.tree_flatten((args, kwargs))
    args = [arg for arg in flat_args if isinstance(arg, torch.Tensor)]

    # We construct a new function that only accepts Tensors as inputs
    def func_no_tensors(args):
        reconstructed_flat_args = []
        args = iter(args)
        for v in flat_args:
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
        if not skip_correctness_check:
            assert_equals_fn(compiled_out, out, msg=outputs_msg)
        return
    _test_aot_autograd_forwards_backwards_helper(
        func_no_tensors, compiled_f, args, assert_raises_regex_fn, assert_equals_fn,
        try_check_data_specialization, skip_correctness_check)

outputs_msg = (
    "Outputs of the operator are different in eager-mode PyTorch vs "
    "AOTDispatcher tracing. This means the operator will have incorrect output "
    "underneath torch.compile. This could be because the operator's "
    "implementation not traceable."
)


def _test_aot_autograd_forwards_backwards_helper(
        f, compiled_f, args, assert_raises_regex_fn, assert_equals_fn,
        try_check_data_specialization, skip_correctness_check=False):
    # Verify grads are equal between compiled and non-compiled versions of f.

    def call_forwards_backwards(f, args):
        flat_args = pytree.arg_tree_leaves(*args)
        diff_args = [arg for arg in flat_args if isinstance(arg, torch.Tensor) and
                     arg.requires_grad]
        out = wrapper_set_seed(f, args)
        flat_out = pytree.tree_leaves(out)

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
        tensor_args = [x for x in pytree.tree_flatten(args)[0] if isinstance(x, torch.Tensor)]
        any_non_leaves = any(x.grad_fn is not None for x in tensor_args)
        if all(x is None for x in orig_grad) and any_non_leaves:
            with assert_raises_regex_fn(RuntimeError, 'does not require grad and does not have a grad_fn'):
                call_forwards_backwards(compiled_f, args)
            return

        msg = (
            "Gradients of the operator are different in eager-mode PyTorch vs "
            "AOTDispatcher. This means the operator will have incorrect gradients "
            "underneath torch.compile. This could be because the operator's "
            "backward is incorrectly registered or not traceable."
        )

        compiled_out, compiled_grad = call_forwards_backwards(compiled_f, args)
        if not skip_correctness_check:
            try:
                assert_equals_fn(compiled_out, orig_out)
            except Exception as e:
                raise type(e)(outputs_msg) from e
            try:
                assert_equals_fn(compiled_grad, orig_grad)
            except Exception as e:
                raise type(e)(msg) from e

    check(args, ignore_failure=False)

    # Randomize the data and run the traced graph with it, to catch bugs
    # where we may have baked in Tensor data into the trace.
    # This is not guaranteed to succeed, because `f` might have preconditions
    # on the values of the inputs, so we just ignore if this test fails.
    if try_check_data_specialization:
        args = randomize(args)
        check(args, ignore_failure=True)
