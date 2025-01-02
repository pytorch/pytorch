import contextlib
import dis
import functools
import logging
import os.path
import random
import re
import sys
import types
import unittest
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from unittest.mock import patch

import torch
from torch import fx
from torch._dynamo.backends.debugging import aot_eager
from torch._dynamo.output_graph import OutputGraph

from . import config, eval_frame, optimize_assert, reset
from .bytecode_transformation import (
    create_instruction,
    debug_checks,
    is_generator,
    transform_code_object,
)
from .guards import CheckFunctionManager, CompileId, GuardedCode
from .types import DynamoFrameType
from .utils import same


np: Optional[types.ModuleType] = None
try:
    import numpy as np
except ModuleNotFoundError:
    np = None


unsupported = eval_frame.unsupported
three = 3

log = logging.getLogger(__name__)


def clone_me(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.detach().clone().requires_grad_(x.requires_grad)


def remove_optimized_module_prefix(name: str) -> str:
    return re.sub(r"^_orig_mod[.]", "", name)


def extract_graph_and_tracker(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
    from torch._dynamo.symbolic_convert import InstructionTranslator

    gm = None
    region_tracker = None

    def extract_graph_backend(_gm, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal gm
        nonlocal region_tracker
        gm = _gm
        region_tracker = InstructionTranslator.current_tx().output.region_tracker
        return _gm

    torch.compile(backend=extract_graph_backend, fullgraph=True)(fn)(*args, **kwargs)
    return gm.graph, region_tracker  # type: ignore[union-attr]


def collect_results(
    model: torch.nn.Module, prediction: Any, loss: Any, example_inputs: Any
) -> List[Any]:
    results = []
    results.append(prediction)
    results.append(loss)
    # if isinstance(loss, torch.Tensor) and loss.item() > 1:
    #     log.warning(
    #         f"High loss value alert - {loss:.2f}. Can result in unstable gradients."
    #     )

    grads = {}
    params = {}
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
    buffers = {}
    for name, buffer in model.named_buffers():
        if isinstance(model, eval_frame.OptimizedModule):
            name = remove_optimized_module_prefix(name)
        buffers[name] = buffer
    results.append(buffers)
    for example in example_inputs:
        if isinstance(example, (tuple, list)):
            results.extend(inp.grad for inp in example if isinstance(inp, torch.Tensor))
        else:
            if isinstance(example, torch.Tensor):
                results.append(example.grad)
    return results


def requires_bwd_pass(out: Any) -> bool:
    if isinstance(out, torch.Tensor):
        return out.requires_grad
    elif isinstance(out, (list, tuple)):
        return any(requires_bwd_pass(x) for x in out)
    elif out is None:
        return False
    elif isinstance(out, int):
        return False
    raise NotImplementedError("Don't know how to reduce", type(out))


@overload
def reduce_to_scalar_loss(out: torch.Tensor) -> torch.Tensor:
    ...


@overload
def reduce_to_scalar_loss(
    out: Union[List[Any], Tuple[Any, ...], Dict[Any, Any]]
) -> float:
    ...


def reduce_to_scalar_loss(out: Any) -> Union[torch.Tensor, float]:
    """Reduce the output of a model to get scalar loss"""
    if isinstance(out, torch.Tensor):
        # Mean does not work on integer tensors
        return out.sum() / out.numel()
    elif isinstance(out, (list, tuple)):
        return sum(reduce_to_scalar_loss(x) for x in out) / len(out)
    elif type(out).__name__ in (
        "MaskedLMOutput",
        "Seq2SeqLMOutput",
        "CausalLMOutputWithCrossAttentions",
    ):
        return reduce_to_scalar_loss(out.logits)
    elif type(out).__name__ == "SquashedNormal":
        return out.mean.sum()
    elif isinstance(out, dict):
        return sum(reduce_to_scalar_loss(value) for value in out.values()) / len(
            out.keys()
        )
    raise NotImplementedError("Don't know how to reduce", type(out))


def debug_dir() -> str:
    path = os.path.join(os.path.dirname(__file__), "../debug")
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def debug_dump(name: str, code: types.CodeType, extra: str = "") -> None:
    with open(os.path.join(debug_dir(), name), "w") as fd:
        fd.write(
            f"{dis.Bytecode(code).info()}\n\n{dis.Bytecode(code).dis()}\n\n{extra}\n"
        )


def debug_insert_nops(
    frame: DynamoFrameType, cache_size: int, hooks: Any, _: Any, *, skip: int = 0
) -> Optional[GuardedCode]:
    """used to debug jump updates"""

    def insert_nops(instructions: List[Any], code_options: Any) -> None:
        instructions.insert(0, create_instruction("NOP"))
        instructions.insert(0, create_instruction("NOP"))

    if is_generator(frame.f_code):
        return None

    debug_checks(frame.f_code)
    code = transform_code_object(frame.f_code, insert_nops)
    graph = OutputGraph(
        code_options={},
        compiler_fn=None,
        root_tx=None,
        export=False,
        export_constraints=None,
        frame_state={"_id": 0},
        # TODO: shouldn't this be f_locals/f_globals from frame?
        local_scope=locals(),
        global_scope=globals(),
        f_code=frame.f_code,
        torch_function_mode_stack=[],
    )

    return GuardedCode(code, CheckFunctionManager(frame.f_code, graph).guard_manager, CompileId(0, 0))  # type: ignore[arg-type]


class CompileCounter:
    def __init__(self) -> None:
        self.frame_count = 0
        self.op_count = 0

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
    ) -> Callable[..., Any]:
        self.frame_count += 1
        for node in gm.graph.nodes:
            if "call" in node.op:
                self.op_count += 1
        return gm.forward

    def clear(self) -> None:
        self.frame_count = 0
        self.op_count = 0


class CompileCounterWithBackend:
    def __init__(self, backend: str) -> None:
        self.frame_count = 0
        self.op_count = 0
        self.backend = backend
        self.graphs: List[torch.fx.GraphModule] = []

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
    ) -> Callable[..., Any]:
        from .backends.registry import lookup_backend

        self.frame_count += 1
        for node in gm.graph.nodes:
            if "call" in node.op:
                self.op_count += 1
        self.graphs.append(gm)
        return lookup_backend(self.backend)(gm, example_inputs)


# Equivalent to backend="eager", but also records graphs that
# we can assert on
class EagerAndRecordGraphs:
    def __init__(self) -> None:
        self.graphs: List[torch.fx.GraphModule] = []

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
    ) -> Callable[..., Any]:
        self.graphs.append(gm)
        return gm.forward


class AotEagerAndRecordGraphs:
    def __init__(self) -> None:
        self.graphs: List[torch.fx.GraphModule] = []
        self.fw_graphs: List[torch.fx.GraphModule] = []
        self.bw_graphs: List[torch.fx.GraphModule] = []

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
    ) -> Callable[..., Any]:
        self.graphs.append(gm)

        def fw_compiler(
            gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
        ) -> Callable[..., Any]:
            self.fw_graphs.append(gm)
            return gm.forward

        def bw_compiler(
            gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
        ) -> Callable[..., Any]:
            self.bw_graphs.append(gm)
            return gm.forward

        return aot_eager(
            gm,
            example_inputs,
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
        )


def strip_comment(code: str) -> str:
    return re.sub(r"(?m)^ *#.*\n?", "", code)


def remove_trailing_space(code: str) -> str:
    return "\n".join([line.rstrip() for line in code.split("\n")])


def normalize_gm(gm_str: str) -> str:
    # strip comments as comments have path to files which may differ from
    # system to system.
    return remove_trailing_space(strip_comment(gm_str))


def empty_line_normalizer(code: str) -> str:
    """
    Normalize code: remove empty lines.
    """
    normal_code = re.sub(r"[\r\n]+", "\n", code)
    return normal_code


def standard_test(
    self: Any,
    fn: Callable[..., Any],
    nargs: int,
    expected_ops: Optional[int] = None,
    expected_ops_dynamic: Optional[int] = None,
    expected_frame_count: int = 1,
) -> None:
    if not config.assume_static_by_default and expected_ops_dynamic is not None:
        expected_ops = expected_ops_dynamic

    actual = CompileCounter()

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
    self.assertEqual(actual.frame_count, expected_frame_count)
    if expected_ops is not None:
        self.assertEqual(actual.op_count, expected_ops)


def dummy_fx_compile(
    gm: fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Callable[..., Any]:
    return gm.forward


def format_speedup(
    speedup: float,
    pvalue: float,
    is_correct: bool = True,
    pvalue_threshold: float = 0.1,
) -> str:
    if not is_correct:
        return "ERROR"
    if pvalue > pvalue_threshold:
        return f"{speedup:.3f}x SAME"
    return f"{speedup:.3f}x p={pvalue:.2f}"


def rand_strided(
    size: Sequence[int],
    stride: Sequence[int],
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "cpu",
    extra_size: int = 0,
) -> torch.Tensor:
    needed_size = (
        sum((shape - 1) * stride for shape, stride in zip(size, stride))
        + 1
        + extra_size
    )
    if dtype.is_floating_point:
        if dtype.itemsize == 1:
            """
            normal distribution kernel is not implemented for fp8..
            Workaround that by creating a fp16 tensor and then cast.
            """
            buffer = torch.randn(needed_size, dtype=torch.float16, device=device).to(
                dtype=dtype
            )
        else:
            buffer = torch.randn(needed_size, dtype=dtype, device=device)
    else:
        buffer = torch.zeros(size=[needed_size], dtype=dtype, device=device)
    return torch.as_strided(buffer, size, stride)


_T = TypeVar("_T")


def check_dynamic_shape_capture() -> bool:
    # This also mirrors config from `test/dynamo/test_dynamic_shapes.py:make_dynamic_cls`
    return not config.assume_static_by_default


def _make_fn_with_patches(fn: Callable[..., _T], *patches: Any) -> Callable[..., _T]:
    @functools.wraps(fn)
    def _fn(*args: Any, **kwargs: Any) -> _T:
        with contextlib.ExitStack() as stack:
            for module, attr, val in patches:
                stack.enter_context(patch.object(module, attr, val))

            return fn(*args, **kwargs)

    return _fn


def make_test_cls_with_patches(
    cls: type,
    cls_prefix: str,
    fn_suffix: str,
    *patches: Any,
    xfail_prop: Optional[str] = None,
    decorator: Callable[[Callable[..., Any]], Callable[..., Any]] = lambda x: x,
) -> type:
    DummyTestClass = type(f"{cls_prefix}{cls.__name__}", cls.__bases__, {})
    DummyTestClass.__qualname__ = DummyTestClass.__name__

    for name in dir(cls):
        if name.startswith("test_"):
            fn = getattr(cls, name)
            if not callable(fn):
                setattr(DummyTestClass, name, getattr(cls, name))
                continue
            new_name = f"{name}{fn_suffix}"
            new_fn = _make_fn_with_patches(fn, *patches)
            new_fn.__name__ = new_name
            if xfail_prop is not None and hasattr(fn, xfail_prop):
                new_fn = unittest.expectedFailure(new_fn)
            setattr(DummyTestClass, new_name, decorator(new_fn))
        # NB: Doesn't handle slots correctly, but whatever
        elif not hasattr(DummyTestClass, name):
            setattr(DummyTestClass, name, getattr(cls, name))

    return DummyTestClass


# test Python 3.11+ specific features
def skipIfNotPy311(fn: Callable[..., Any]) -> Callable[..., Any]:
    if sys.version_info >= (3, 11):
        return fn
    return unittest.skip(fn)


def skipIfNotPy312(fn: Callable[..., Any]) -> Callable[..., Any]:
    if sys.version_info >= (3, 12):
        return fn
    return unittest.skip("Requires Python 3.12+")(fn)


def xfailIfPy312(fn: Callable[..., Any]) -> Callable[..., Any]:
    if sys.version_info >= (3, 12):
        return unittest.expectedFailure(fn)
    return fn


def skipIfPy312(fn: Callable[..., Any]) -> Callable[..., Any]:
    if sys.version_info >= (3, 12):
        return unittest.skip("Not supported in Python 3.12+")(fn)
    return fn


def requiresPy310(fn: Callable[..., Any]) -> Callable[..., Any]:
    if sys.version_info >= (3, 10):
        return fn
    else:
        return unittest.skip("Requires Python 3.10+")(fn)


# Controls tests generated in test/inductor/test_torchinductor_dynamic_shapes.py
# and test/dynamo/test_dynamic_shapes.py
def expectedFailureDynamic(fn: Callable[..., Any]) -> Callable[..., Any]:
    fn._expected_failure_dynamic = True  # type: ignore[attr-defined]
    return fn


# Controls tests generated in test/inductor/test_torchinductor_codegen_dynamic_shapes.py
def expectedFailureCodegenDynamic(fn: Callable[..., Any]) -> Callable[..., Any]:
    fn._expected_failure_codegen_dynamic = True  # type: ignore[attr-defined]
    return fn


# Controls test generated in test/inductor/test_cpp_wrapper.py
def expectedFailureDynamicWrapper(fn: Callable[..., Any]) -> Callable[..., Any]:
    fn._expected_failure_dynamic_wrapper = True  # type: ignore[attr-defined]
    return fn


def reset_rng_state(use_xla: bool = False) -> None:
    torch.manual_seed(1337)
    random.seed(1337)
    if np:
        np.random.seed(1337)
    if use_xla:
        import torch_xla.core.xla_model as xm

        xm.set_rng_state(1337, str(xm.xla_device()))
