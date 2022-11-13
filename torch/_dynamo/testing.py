import contextlib
import dis
import functools
import logging
import os.path
import types
import unittest
from unittest.mock import patch

import torch
from torch import fx

from . import config, eval_frame, optimize_assert, reset
from .bytecode_transformation import (
    create_instruction,
    debug_checks,
    is_generator,
    transform_code_object,
)
from .guards import CheckFunctionManager, GuardedCode
from .utils import same

unsupported = eval_frame.unsupported
three = 3

log = logging.getLogger(__name__)


def clone_me(x):
    if x is None:
        return None
    return x.detach().clone().requires_grad_(x.requires_grad)


def named_parameters_for_optimized_module(mod):
    assert isinstance(mod, eval_frame.OptimizedModule)
    return mod._orig_mod.named_parameters


def remove_optimized_module_prefix(name):
    prefix = "_orig_mod."
    assert name.startswith(prefix)
    name = name[len(prefix) :]
    return torch.distributed.fsdp._common_utils.clean_tensor_name(name)


def collect_results(model, prediction, loss, example_inputs):
    results = []
    results.append(prediction)
    results.append(loss)
    if isinstance(loss, torch.Tensor) and loss.item() > 1:
        log.warning(
            f"High loss value alert - {loss:.2f}. Can result in unstable gradients."
        )

    grads = dict()
    params = dict()
    for name, param in model.named_parameters():
        if isinstance(model, eval_frame.OptimizedModule):
            name = remove_optimized_module_prefix(name)
        param_copy = param
        grad = param.grad
        # Treat None and zero grad as same
        if param.grad is None:
            grad = torch.zeros_like(param)
        grads[name + ".grad"] = grad
        params[name] = param_copy
    results.append(grads)
    results.append(params)
    for example in example_inputs:
        if isinstance(example, (tuple, list)):
            for inp in example:
                if isinstance(inp, torch.Tensor):
                    results.append(inp.grad)
        else:
            if isinstance(example, torch.Tensor):
                results.append(example.grad)
    return results


def requires_bwd_pass(out):
    if isinstance(out, torch.Tensor):
        return out.requires_grad
    elif isinstance(out, (list, tuple)):
        return any([requires_bwd_pass(x) for x in out])
    elif out is None:
        return False
    raise NotImplementedError("Don't know how to reduce", type(out))


def reduce_to_scalar_loss(out):
    """Reduce the output of a model to get scalar loss"""
    if isinstance(out, torch.Tensor):
        # Mean does not work on integer tensors
        return out.sum() / out.numel()
    elif isinstance(out, (list, tuple)):
        return sum([reduce_to_scalar_loss(x) for x in out]) / len(out)
    elif type(out).__name__ in (
        "MaskedLMOutput",
        "Seq2SeqLMOutput",
        "CausalLMOutputWithCrossAttentions",
    ):
        return reduce_to_scalar_loss(out.logits)
    elif type(out).__name__ == "SquashedNormal":
        return out.mean.sum()
    elif isinstance(out, dict):
        return sum([reduce_to_scalar_loss(value) for value in out.values()]) / len(
            out.keys()
        )
    raise NotImplementedError("Don't know how to reduce", type(out))


def debug_dir():
    path = os.path.join(os.path.dirname(__file__), "../debug")
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def debug_dump(name, code: types.CodeType, extra=""):
    with open(os.path.join(debug_dir(), name), "w") as fd:
        fd.write(
            f"{dis.Bytecode(code).info()}\n\n{dis.Bytecode(code).dis()}\n\n{extra}\n"
        )


def debug_insert_nops(frame, cache_size):
    """used to debug jump updates"""

    def insert_nops(instructions, code_options):
        instructions.insert(0, create_instruction("NOP"))
        instructions.insert(0, create_instruction("NOP"))

    if is_generator(frame.f_code):
        return None

    debug_checks(frame.f_code)
    code = transform_code_object(frame.f_code, insert_nops)

    return GuardedCode(code, CheckFunctionManager().check_fn)


class CompileCounter:
    def __init__(self):
        self.frame_count = 0
        self.op_count = 0

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        self.frame_count += 1
        for node in gm.graph.nodes:
            if "call" in node.op:
                self.op_count += 1
        return gm.forward

    def clear(self):
        self.frame_count = 0
        self.op_count = 0


class CompileCounterWithBackend:
    def __init__(self, backend):
        self.frame_count = 0
        self.op_count = 0
        self.backend = backend

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        from torch._dynamo.eval_frame import lookup_backend

        self.frame_count += 1
        for node in gm.graph.nodes:
            if "call" in node.op:
                self.op_count += 1
        return lookup_backend(self.backend)(gm, example_inputs)


def standard_test(self, fn, nargs, expected_ops=None, expected_ops_dynamic=None):
    if config.dynamic_shapes and expected_ops_dynamic is not None:
        expected_ops = expected_ops_dynamic

    actual = CompileCounter()
    if expected_ops is None:
        expected = CompileCounter()
        try:
            gm = torch.fx.symbolic_trace(fn)
            expected(gm)
            print("\nfx.symbolic_trace graph:")
            gm.graph.print_tabular()
            expected_ops = expected.op_count
        except Exception:
            pass  # Silently ignore FX errors (not our issue)

    args1 = [torch.randn(10, 10) for _ in range(nargs)]
    args2 = [torch.randn(10, 10) for _ in range(nargs)]
    correct1 = fn(*args1)
    correct2 = fn(*args2)
    reset()
    opt_fn = optimize_assert(actual)(fn)
    val1a = opt_fn(*args1)
    val2a = opt_fn(*args2)
    val1b = opt_fn(*args1)
    val2b = opt_fn(*args2)
    reset()
    self.assertTrue(same(val1a, correct1))
    self.assertTrue(same(val1b, correct1))
    self.assertTrue(same(val2a, correct2))
    self.assertTrue(same(val2b, correct2))
    self.assertEqual(actual.frame_count, 1)
    if expected_ops is not None:
        self.assertEqual(actual.op_count, expected_ops)


def dummy_fx_compile(gm: fx.GraphModule, example_inputs):
    return gm.forward


def format_speedup(speedup, pvalue, is_correct=True, pvalue_threshold=0.1):
    if not is_correct:
        return "ERROR"
    if pvalue > pvalue_threshold:
        return f"{speedup:.3f}x SAME"
    return f"{speedup:.3f}x p={pvalue:.2f}"


def requires_static_shapes(fn):
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        if config.dynamic_shapes:
            raise unittest.SkipTest("requires static shapes")
        return fn(*args, **kwargs)

    return _fn


def rand_strided(size, stride, dtype=torch.float32, device="cpu"):
    needed_size = sum((shape - 1) * stride for shape, stride in zip(size, stride)) + 1
    if dtype.is_floating_point:
        buffer = torch.randn(needed_size, dtype=dtype, device=device)
    else:
        buffer = torch.ones(size=[needed_size], dtype=dtype, device=device)
    return torch.as_strided(buffer, size, stride)


def _make_fn_with_patches(fn, *patches):
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        with contextlib.ExitStack() as stack:
            for attr, val in patches:
                stack.enter_context(patch.object(config, attr, val))

            return fn(*args, **kwargs)

    return _fn


def make_test_cls_with_patches(cls, cls_prefix, fn_suffix, *patches):
    class DummyTestClass(cls):
        pass

    DummyTestClass.__name__ = f"{cls_prefix}{cls.__name__}"

    for name in dir(cls):
        if name.startswith("test_"):
            fn = getattr(cls, name)
            if not callable(fn):
                continue
            new_name = f"{name}{fn_suffix}"
            fn = _make_fn_with_patches(fn, *patches)
            fn.__name__ = new_name
            setattr(DummyTestClass, name, None)
            setattr(DummyTestClass, new_name, fn)

    return DummyTestClass
