import contextlib
import copy
import functools
import inspect
import logging
import os
import sys
import threading
import traceback
import types
import warnings
from importlib import import_module
from unittest.mock import patch

import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.parallel.distributed import DistributedDataParallel

from . import config, convert_frame, skipfiles, utils
from .exc import ResetRequired
from .mutation_guard import install_generation_tagging_init
from .optimizations.distributed import DDPOptimizer
from .utils import checkpoint_params, clone_inputs, compile_times, same

log = logging.getLogger(__name__)

try:
    from torch.fx.experimental import proxy_tensor
except ImportError:
    proxy_tensor = None

_eval_frame = torch._C._dynamo.eval_frame
set_eval_frame = _eval_frame.set_eval_frame
reset_code = _eval_frame.reset_code
unsupported = _eval_frame.unsupported
skip_code = _eval_frame.skip_code
set_guard_fail_hook = _eval_frame.set_guard_fail_hook
set_guard_error_hook = _eval_frame.set_guard_error_hook
always_optimize_code_objects = utils.ExactWeakKeyDictionary()
null_context = contextlib.nullcontext
unset = object()
compile_lock = threading.RLock()
most_recent_backend = None


def remove_from_cache(f):
    """
    Make sure f.__code__ is not cached to force a recompile
    """
    if isinstance(f, types.CodeType):
        reset_code(f)
    elif hasattr(f, "__code__"):
        reset_code(f.__code__)
    elif hasattr(getattr(f, "forward", None), "__code__"):
        reset_code(f.forward.__code__)
    else:
        from . import reset

        reset()
        log.warning("could not determine __code__ for %s", f)


def nothing():
    pass


def innermost_fn(fn):
    """
    In case of nesting of _TorchDynamoContext calls, find the innermost
    function. TorchDynamo caches on fn.__code__ object, so its necessary to find
    the innermost function to pass on the optimize, run, disable etc.
    """
    unaltered_fn = fn
    while hasattr(unaltered_fn, "_torchdynamo_orig_callable"):
        unaltered_fn = unaltered_fn._torchdynamo_orig_callable
        assert callable(unaltered_fn)
    return unaltered_fn


class _TorchDynamoContext:
    def __init__(
        self,
        callback,
        on_enter=nothing,
        backend_ctx_ctor=null_context,
        patch_fn=nothing,
        first_ctx=False,
    ):
        super().__init__()
        assert callable(callback) or callback is False or callback is None
        self.callback = callback
        self.prior = unset
        self.on_enter = on_enter
        self.extra_ctx_ctor = backend_ctx_ctor
        self.first_ctx = first_ctx
        patch_fn()

    def __enter__(self):
        if config.raise_on_ctx_manager_usage:
            raise RuntimeError(
                "torch._dynamo.optimize(...) is used with a context manager. "
                "Please refer to https://github.com/pytorch/torchdynamo#usage-example "
                "to use torch._dynamo.optimize(...) as an annotation/decorator. "
            )
        self.on_enter()
        self.prior = set_eval_frame(self.callback)
        self.backend_ctx = self.extra_ctx_ctor()
        self.backend_ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_eval_frame(self.prior)
        self.prior = unset
        self.backend_ctx.__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, fn):
        fn = innermost_fn(fn)
        # Optimize the forward method of torch.nn.Module object
        if isinstance(fn, torch.nn.Module):
            mod = fn
            optimized_forward = self(mod.forward)

            class TorchDynamoNNModuleWrapper:
                """
                A wrapper that redirects the forward call to the optimized
                forward, while for rest it redirects the calls to the original
                module.
                """

                def __getattr__(self, name):
                    return getattr(mod, name)

                def forward(self, *args, **kwargs):
                    return optimized_forward(*args, **kwargs)

                def __call__(self, *args, **kwargs):
                    return self.forward(*args, **kwargs)

            new_mod = TorchDynamoNNModuleWrapper()
            # Save the function pointer to find the original callable while nesting
            # of decorators.
            new_mod._torchdynamo_orig_callable = mod
            return new_mod

        assert callable(fn)
        callback = self.callback
        on_enter = self.on_enter
        backend_ctx_ctor = self.extra_ctx_ctor

        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            if (
                not isinstance(self, DisableContext)
                and torch.fx._symbolic_trace.is_fx_tracing()
            ):
                if config.error_on_nested_fx_trace:
                    raise RuntimeError(
                        "Detected that you are using FX to symbolically trace "
                        "a dynamo-optimized function. This is not supported at the moment."
                    )
                else:
                    return fn(*args, **kwargs)

            on_enter()
            prior = set_eval_frame(callback)
            backend_ctx = backend_ctx_ctor()
            backend_ctx.__enter__()
            try:
                return fn(*args, **kwargs)
            finally:
                set_eval_frame(prior)
                backend_ctx.__exit__(None, None, None)

        # hooks to properly handle inlining
        if isinstance(self, DisableContext):
            _fn._torchdynamo_disable = True
        else:
            _fn._torchdynamo_inline = fn

        # Save the function pointer to find the original callable while nesting
        # of decorators.
        _fn._torchdynamo_orig_callable = fn

        # If the function is called using torch._dynamo.optimize decorator, we
        # should prevent any type of skipping.
        if callback not in (None, False):
            always_optimize_code_objects[fn.__code__] = True

        return _fn


class OptimizeContext(_TorchDynamoContext):
    def __init__(self, callback, backend_ctx_ctor, first_ctx=False):
        def on_enter():
            global most_recent_backend
            if (
                most_recent_backend is not None
                and most_recent_backend is not compiler_fn
            ):
                raise ResetRequired()
            most_recent_backend = compiler_fn
            install_generation_tagging_init()

        compiler_fn = innermost_fn(callback)
        super().__init__(
            callback=callback,
            on_enter=on_enter,
            backend_ctx_ctor=backend_ctx_ctor,
            patch_fn=TorchPatcher.patch,
            first_ctx=first_ctx,
        )


class RunOnlyContext(_TorchDynamoContext):
    def __init__(self):
        super().__init__(callback=False)


class DisableContext(_TorchDynamoContext):
    def __init__(self):
        super().__init__(callback=None)


def catch_errors_wrapper(callback):
    @functools.wraps(callback)
    def catch_errors(frame, cache_size):
        if frame.f_lasti >= 0 or skipfiles.check(frame.f_code.co_filename):
            log.debug(f"skipping {frame.f_code.co_name} {frame.f_code.co_filename}")
            return None
        if frame.f_code.co_filename == "<string>" and frame.f_code.co_name == "__new__":
            # nametuple constructor
            return None
        if config.optimize_ddp:
            ddp_module = DistributedDataParallel._get_active_ddp_module()
            if ddp_module:
                with compile_lock:
                    ddp_optimizer = DDPOptimizer(
                        bucket_bytes_cap=ddp_module.bucket_bytes_cap,
                        backend_compile_fn=callback._torchdynamo_orig_callable,
                    )
                    hijacked_callback = convert_frame.convert_frame(
                        ddp_optimizer.compile_fn, guard_export_fn=None
                    )
                    return hijacked_callback(frame, cache_size)

        with compile_lock:
            return callback(frame, cache_size)

    catch_errors._torchdynamo_orig_callable = callback
    return catch_errors


def _optimize_catch_errors(compile_fn, backend_ctx_ctor=null_context):
    return OptimizeContext(
        catch_errors_wrapper(compile_fn),
        backend_ctx_ctor=backend_ctx_ctor,
        first_ctx=True,
    )


class WrapperBackend:
    def __init__(self, backend=None):
        self.backend = backend

    @property
    def example_inputs(self):
        return clone_inputs(self.original_example_inputs)

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):

        self.restore = checkpoint_params(gm)
        self.original_example_inputs = clone_inputs(example_inputs)
        self.gm = gm
        copy_gm = copy.deepcopy(self.gm)
        self.candidate = self.backend(copy_gm, self.original_example_inputs)

        if self.candidate is None or self.candidate is self.gm.forward:
            return self.gm.forward

        if not config.verify_correctness:
            return self.candidate

        # if verify_correctness=True
        try:
            correct = self.gm.forward(*self.example_inputs)
            result = self.candidate(*self.example_inputs)

            # TODO: replace `same` function with the one in testing
            if same(correct, result):
                return self.candidate

            raise RuntimeError(f"incorrect results of backend {self}")
            return self.gm.forward

        except Exception:
            log.exception("error in verify_correctness")
            raise
        finally:
            self.restore()


def get_compiler_fn(compiler_fn):
    from .debug_utils import wrap_backend_debug

    compiler_str = compiler_fn if isinstance(compiler_fn, str) else None
    compiler_fn = lookup_backend(compiler_fn)
    return wrap_backend_debug(compiler_fn, compiler_str)


@functools.lru_cache(1)
def lookup_backend(compiler_fn):
    """Expand backend strings to functions"""
    if compiler_fn == "inductor":
        compiler_fn = import_module(f"{config.inductor_import}.compile_fx").compile_fx
    elif isinstance(compiler_fn, str):
        from .optimizations import BACKENDS

        compiler_fn = BACKENDS[compiler_fn]
    return compiler_fn


class _NullDecorator(contextlib.nullcontext):
    def __call__(self, fn):
        assert callable(fn)
        return fn


def optimize(
    backend="inductor", *, nopython=False, guard_export_fn=None, disable=False
):
    """
    The main entrypoint of TorchDynamo.  Do graph capture and call
    backend() to optimize extracted graphs.

    Args:
        backend: One of the two things:
            - Either, a function/callable taking a torch.fx.GraphModule and
            example_inputs and returning a python callable that runs the
            graph faster.
            One can also provide additional context for the backend, like
            torch.jit.fuser("fuser2"), by setting the backend_ctx_ctor attribute.
            See AOTAutogradMemoryEfficientFusionWithContext for the usage.
            - Or, a string backend name in `torch._dynamo.list_backends()`
        nopython: If True, graph breaks will be errors and there will
            be a single whole-program graph.
        disable: If True, turn this decorator into a no-op

    Example Usage:

        @torch._dynamo.optimize()
        def toy_example(a, b):
            ...
    """
    torch._C._log_api_usage_once("torch._dynamo.optimize")
    if disable or os.environ.get("TORCHDYNAMO_DISABLE", "") == "1":
        return _NullDecorator()
    if sys.platform == "win32":
        warnings.warn(
            "Windows is not currently supported, "
            + f"{config.dynamo_import}.optimize() will do nothing"
        )
        return _NullDecorator()
    if sys.version_info >= (3, 11):
        warnings.warn(
            "Python 3.11+ not yet supported, "
            f"{config.dynamo_import}.optimize() will do nothing"
        )
        return _NullDecorator()

    backend = get_compiler_fn(backend)

    # Find if backend has any extra context manager
    backend_ctx_ctor = getattr(backend, "backend_ctx_ctor", null_context)

    if nopython:
        return optimize_assert(backend, guard_export_fn=guard_export_fn)
    return _optimize_catch_errors(
        convert_frame.convert_frame(backend, guard_export_fn=guard_export_fn),
        backend_ctx_ctor,
    )


@patch("torch._dynamo.symbolic_convert.explain", True)
def explain(f, *args, **kwargs):
    # TODO(voz): Do we want a decorator for this?
    from . import reset

    reset()

    out_guards = []
    graphs = []
    ops_per_graph = []
    op_count = 0
    break_reasons = []

    def dynamo_graph_accumulating_compiler(gm: torch.fx.GraphModule, example_inputs):
        nonlocal graphs
        nonlocal op_count
        nonlocal ops_per_graph

        graphs.append(gm)
        ops = []
        for node in gm.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)

        op_count += len(ops)
        ops_per_graph.append(ops)
        if gm.compile_subgraph_reason is not None:
            break_reasons.append(gm.compile_subgraph_reason)
        return gm.forward

    def guard_export_print(guards):
        nonlocal out_guards
        out_guards.append(guards)

    with patch(f"{__name__}.most_recent_backend", None):
        opt_f = optimize(
            dynamo_graph_accumulating_compiler,
            nopython=False,
            guard_export_fn=guard_export_print,
        )(f)
        # TODO(voz): We may have instances of `f` that mutate inputs, we should track sideffects and reject.
        opt_f(*args, **kwargs)

    graph_count = len(graphs)

    # For the explanation summary, dedupe reasons by the innermost stack frame and dedupe by it.
    deduped_reasons = {}
    for reason in break_reasons:
        innermost_frame = reason.user_stack[-1]
        # __repr__ uniquely identifies a FrameSummary so we can use it for deduping
        deduped_reasons[repr(innermost_frame)] = reason

    formatted_list = ""
    for idx, break_reason in enumerate(deduped_reasons.values()):
        formatted_stack = "".join(traceback.format_list(break_reason.user_stack))
        msg = f"{break_reason.reason}\n{formatted_stack}"
        formatted_list += f"{idx + 1}. {msg} \n"

    explanation = f"Dynamo produced {graph_count} graphs"
    explanation += f"with {graph_count - 1} graph break and {op_count} ops"
    explanation += f"\n Break reasons: \n\n{formatted_list}"

    explanation += compile_times()

    # TODO(voz): Do we want a decorator for this?
    reset()
    return explanation, out_guards, graphs, ops_per_graph, break_reasons


def export(
    f, *args, aten_graph=False, decomposition_table=None, tracing_mode="real", **kwargs
):
    torch._C._log_api_usage_once("torch._dynamo.export")
    if decomposition_table is not None or tracing_mode != "real":
        assert (
            aten_graph
        ), "Specifying a decomposition_table table or tracing mode is illegal without setting aten_graph=True"
    f = innermost_fn(f)

    graph = None
    out_guards = None
    graph_captured_input = None
    graph_captured_result = None

    def produce_matching(source_args, candidate_args):
        matched_elements_positions = []
        dict_of_source_args = dict()
        for i in range(0, len(source_args)):
            element_id = id(source_args[i])
            dict_of_source_args[element_id] = i

        for i in range(0, len(candidate_args)):
            arg = candidate_args[i]
            # 1-element tensor arg can be unspec int/float
            if isinstance(arg, torch.Tensor) and torch.numel(arg) == 1:
                if id(arg) in dict_of_source_args:
                    matched_elements_positions.append(dict_of_source_args[id(arg)])
                elif id(arg.item()) in dict_of_source_args:
                    matched_elements_positions.append(
                        dict_of_source_args[id(arg.item())]
                    )
                else:
                    raise AssertionError(
                        "Dynamo input/output is not consistent with traced input/output"
                    )
            else:
                assert (
                    id(arg) in dict_of_source_args
                ), "Dynamo input and output is a strict subset of traced input/output"
                matched_elements_positions.append(dict_of_source_args[id(arg)])

        return matched_elements_positions

    def guard_export_print(guards):
        nonlocal out_guards
        assert out_guards is None, "whole graph export entails exactly one guard export"
        out_guards = guards

    def dynamo_normalization_capturing_compiler(
        gm: torch.fx.GraphModule, example_inputs
    ):
        nonlocal graph

        assert graph is None, "whole graph export entails exactly one graph"
        graph = gm

        def result_capturing_wrapper(*graph_inputs):
            nonlocal graph_captured_result
            nonlocal graph_captured_input

            graph_captured_input = graph_inputs
            graph_captured_result = graph(*graph_inputs)
            return graph_captured_result

        return result_capturing_wrapper

    # TODO(voz): Handle kwargs properly?
    flat_args, in_spec = pytree.tree_flatten(args)

    remove_from_cache(f)
    with patch(f"{__name__}.most_recent_backend", None):
        opt_f = optimize_assert(
            dynamo_normalization_capturing_compiler,
            guard_export_fn=guard_export_print,
            export=True,
        )(f)
        # TODO(voz): We may have instances of `f` that mutate inputs, we should track sideffects and reject.
        result_traced = opt_f(*args, **kwargs)
    remove_from_cache(f)

    assert graph is not None, "whole graph export entails exactly one call"
    assert out_guards is not None, "whole graph export entails exactly one guard export"

    matched_input_elements_positions = produce_matching(flat_args, graph_captured_input)

    flat_results_traced, out_spec_traced = pytree.tree_flatten(result_traced)

    flat_both = list(graph_captured_result) + flat_args
    matched_output_elements_positions = produce_matching(flat_both, flat_results_traced)

    class ChangeInputOutputSignature(torch.fx.interpreter.Transformer):
        def __init__(
            self,
            m,
        ):
            super().__init__(m)
            arg_len = len(flat_args)
            self.new_args = [
                super(ChangeInputOutputSignature, self).placeholder(f"arg{i}", (), {})
                for i in range(0, arg_len)
            ]
            self.old_args_gen = (
                self.new_args[i] for i in matched_input_elements_positions
            )

        def placeholder(self, target, args, kwargs):
            arg = next(self.old_args_gen)
            if "val" in self.current_node.meta:
                arg.node.meta["val"] = self.current_node.meta["val"]
            return arg

        def output(self, target, args, kwargs):
            dynamo_result_flat = args[0]
            lookup = [*dynamo_result_flat, *self.new_args]
            new_result_flat = [lookup[i] for i in matched_output_elements_positions]
            new_result = pytree.tree_unflatten(new_result_flat, out_spec_traced)

            return super().output(target, (new_result,), {})

        def run_node(self, n):
            self.current_node = n
            return super().run_node(n)

    if aten_graph:
        # Running graph with interpreter is needed for propagating the stack_trace
        def graph_with_interpreter(*args):
            with torch.fx.traceback.override_stack_trace():
                return torch.fx.Interpreter(graph).run(*args)

        graph = make_fx(
            graph_with_interpreter,
            decomposition_table=decomposition_table,
            tracing_mode=tracing_mode,
        )(*graph_captured_input)

    new_graph = ChangeInputOutputSignature(
        graph,
    ).transform()

    return (new_graph, out_guards)


def assume_constant_result(fn):
    fn._dynamo_marked_constant = True
    return fn


def optimize_assert(backend, *, guard_export_fn=None, export=False):
    """
    The same as `torch._dynamo.optimize(backend, nopython=True)`
    """
    backend = get_compiler_fn(backend)

    # Find if backend has any extra context manager
    backend_ctx_ctor = getattr(backend, "backend_ctx_ctor", null_context)

    return _optimize_catch_errors(
        convert_frame.convert_frame_assert(backend, guard_export_fn, export=export),
        backend_ctx_ctor,
    )


def run(fn=None):
    """Don't do any dynamic compiles, just use prior optimizations"""
    if fn is not None:
        fn = innermost_fn(fn)
        assert callable(fn)
        return RunOnlyContext()(fn)
    return RunOnlyContext()


def disable(fn=None):
    """Decorator and context manager to disable TorchDynamo"""
    if fn is not None:
        fn = innermost_fn(fn)
        assert callable(fn)
        return DisableContext()(fn)
    return DisableContext()


def skip(fn=None):
    """
    Skip frames associated with the function code, but still process recursively
    invoked frames
    """
    if fn is None:
        return skip
    fn = innermost_fn(fn)
    assert callable(fn)
    skip_code(fn.__code__)
    fn._torchdynamo_disable = True
    return fn


class TorchPatcher:
    @staticmethod
    @functools.lru_cache(None)
    def patch():
        # Disable TorchDynamo on some torch.* compilers generated frames
        torch.jit.trace = disable(torch.jit.trace)
        torch.jit.trace_module = disable(torch.jit.trace_module)
        torch.jit._get_trace_graph = disable(torch.jit._get_trace_graph)

        # symbolic_trace creates new frames. We disable Dynamo on such frames
        torch.fx._symbolic_trace.Tracer.trace = disable(
            torch.fx._symbolic_trace.Tracer.trace
        )

        torch.onnx.export_to_pretty_string = disable(torch.onnx.export_to_pretty_string)
        torch.distributions.Distribution.set_default_validate_args(False)

        if proxy_tensor is not None:
            proxy_tensor.dispatch_trace = disable(proxy_tensor.dispatch_trace)

        optimizers = [
            opt
            for opt in torch.optim.__dict__.values()
            if inspect.isclass(opt) and issubclass(opt, torch.optim.Optimizer)
        ]

        # disable dynamo for the wrapper that helps give dynamo hints about entering DDP
        if hasattr(DistributedDataParallel, "_inside_ddp_forward"):
            DistributedDataParallel._inside_ddp_forward = skip(
                DistributedDataParallel._inside_ddp_forward
            )

        # disable profile hook
        for opt in optimizers:
            opt._cuda_graph_capture_health_check = disable(
                opt._cuda_graph_capture_health_check
            )
            # disable any currently set hooks
            # Note: we only want to disable the profiling hook
            # which is the *last* hook applied, we want to keep the no_grad hook
            hooked = getattr(opt.step, "hooked", False)
            if hooked:
                unwrapped_step = getattr(opt.step, "__wrapped__", None)
                if unwrapped_step:
                    opt.step = unwrapped_step

            # disable future hooking
            opt.step.hooked = True

    @staticmethod
    def suppress_torch_distributed_warnings(fn):
        def inner_fn(*args, **kwargs):
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="torch.distributed"
            )
            return fn(*args, **kwargs)

        return inner_fn
