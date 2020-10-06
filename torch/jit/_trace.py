"""Tracing

This module contains functionality to support the JIT's tracing frontend, notably:
    * torch.jit.trace
    * torch.jit.trace_module

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""
import torch

import copy
import os
import contextlib
import functools
import warnings
import inspect
import re
from typing import Any, Dict, List, Optional, Set

from torch.jit._state import _python_cu, _enabled
from torch.jit._script import ScriptModule, _CachedForward, script
from torch._jit_internal import _qualified_name
from torch.autograd import function
from torch.nn import Module

_flatten = torch._C._jit_flatten
_unflatten = torch._C._jit_unflatten


def _create_interpreter_name_lookup_fn(frames_up=1):
    def _get_interpreter_name_for_var(var):
        frame = inspect.currentframe()
        if not frame:
            raise RuntimeError("failed to inspect frame")

        i = 0
        while i < frames_up + 1:
            frame = frame.f_back
            if not frame:
                raise RuntimeError("failed to get frame")
            i += 1

        f_locals = frame.f_locals
        f_globals = frame.f_globals

        for k, v in f_locals.items():
            if isinstance(v, torch.Tensor) and var is v:
                return k if k != "self" else ""
        return ""

    return _get_interpreter_name_for_var


def _unique_state_dict(module, keep_vars=False):
    # since Parameter.detach() always creates a new torch.Tensor instance,
    # id(v) doesn't work with it. So we always get the Parameter or Buffer
    # as values, and deduplicate the params using Parameters and Buffers
    state_dict = module.state_dict(keep_vars=True)
    filtered_dict = type(state_dict)()
    seen_ids: Set[int] = set()
    for k, v in state_dict.items():
        if id(v) in seen_ids:
            continue
        seen_ids.add(id(v))
        if keep_vars:
            filtered_dict[k] = v
        else:
            filtered_dict[k] = v.detach()
    return filtered_dict


class ONNXTracedModule(torch.nn.Module):
    def __init__(
        self,
        inner,
        strict=True,
        force_outplace=False,
        return_inputs=False,
        return_inputs_states=False,
    ):
        super(ONNXTracedModule, self).__init__()
        # inner may be a Module, or it may be an arbitrary callable
        # If it's a Module, we get its parameters automatically, which lets
        # us avoid a special casing functions versus modules.
        self.inner = inner
        self.strict = strict
        self._force_outplace = force_outplace
        self._return_inputs = return_inputs
        self._return_inputs_states = return_inputs_states

    def forward(self, *args: torch.Tensor):
        in_vars, in_desc = _flatten(args)
        # NOTE: use full state, because we need it for BatchNorm export
        # This differs from the compiler path, which doesn't support it at the moment.
        module_state = list(_unique_state_dict(self, keep_vars=True).values())

        ret_inputs = []
        inputs_states = []
        outs = []

        def wrapper(*args):
            in_args: List[torch.Tensor] = []
            for i in range(len(in_vars)):
                if not isinstance(args[i], torch.Tensor):
                    raise RuntimeError('Expected Tensor argument')
                in_args.append(args[i])

            trace_inputs = _unflatten(in_args, in_desc)

            ret_inputs.append(
                tuple(x.clone(memory_format=torch.preserve_format) for x in args)
            )
            if self._return_inputs_states:
                inputs_states.append(_unflatten(in_args, in_desc))
            outs.append(self.inner(*trace_inputs))
            if self._return_inputs_states:
                inputs_states[0] = (inputs_states[0], trace_inputs)
            out_vars, _ = _flatten(outs)
            if len(out_vars) == 1:
                return out_vars[0]
            else:
                return tuple(out_vars)

        graph, out = torch._C._create_graph_by_tracing(
            wrapper,
            in_vars + module_state,
            _create_interpreter_name_lookup_fn(),
            self.strict,
            self._force_outplace,
        )

        if self._return_inputs:
            return graph, outs[0], ret_inputs[0]
        if self._return_inputs_states:
            return graph, outs[0], inputs_states[0]
        else:
            return graph, outs[0]


def _clone_inputs(args):
    def clone_input(a):
        if a is None:
            return None
        elif isinstance(a, torch.Tensor):
            # TODO: figure out one liner to .clone() and set requires_grad
            v = (
                a.detach()
                .clone(memory_format=torch.preserve_format)
                .requires_grad_(a.requires_grad)
            )
            if a.grad is not None:
                v.grad = clone_input(v.grad)
            return v
        else:
            return a.clone(memory_format=torch.preserve_format)

    return function._nested_map(
        lambda x: isinstance(x, torch.Tensor), clone_input, condition_msg="tensors"
    )(args)


# This is purely for developer debugging.  We are not going to advertise it.
_JIT_TIME = os.environ.get("PYTORCH_JIT_TIME", False)  # CUDA-only timing
_JIT_DISABLE = os.environ.get("PYTORCH_JIT_DISABLE", False)
_JIT_STATS = os.environ.get("PYTORCH_JIT_STATS", False)


@contextlib.contextmanager
def _time(trace_name, name, time=True):
    if (not _JIT_TIME and not time) or not torch.cuda.is_available():
        yield
        return
    stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stream.record_event(start)
    try:
        yield
    finally:
        stream.record_event(end)
        end.synchronize()
        print("{} {} time: {} ms".format(trace_name, name, start.elapsed_time(end)))


def verify(model, args, loss_fn=torch.sum, devices=None):
    """
    Verify that a JIT compiled model has the same behavior as its uncompiled
    version along with its backwards pass.  If your model returns multiple
    outputs, you must also specify a `loss_fn` to produce a loss for which
    the backwards will be computed.

    This function has side-effects (e.g., it executes your model / saves and loads
    parameters), so don't expect the model to come out exactly the same as what
    you passed in.

    Arguments:
        model (compiled torch.nn.Module or function): the module/function to be
            verified.  The module/function definition MUST have been decorated with
            `@torch.jit.compile`.
        args (tuple or Tensor): the positional arguments to pass to the
            compiled function/module to be verified.  A non-tuple is assumed to
            be a single positional argument to be passed to the model.
        loss_fn (function, optional): the loss function to be applied to
            the output of the model, before backwards is invoked.  By default,
            we assume that a model returns a single result, and we :func:`torch.sum`
            before calling backwards; if this is inappropriate, you can pass your
            own loss function.  Note that if a model returns a tuple of results,
            these are passed as separate positional arguments to `loss_fn`.
        devices (iterable of device IDs, optional): the GPU devices which the
            compiled module will be run on.  This determines the RNG state we
            must save when running both compiled and uncompiled versions of the model.
    """
    # TODO: In principle, we track device information in our trace, so it
    # should be possible to check if our execution actually obeyed the 'devices'
    # the user provided.

    # TODO: Consider adding a utility function to torch.jit to test
    # for this case
    if not isinstance(model, torch._C.CompiledFunction):  # type: ignore
        raise TypeError(
            "Cannot verify an uncompiled module.  Add @torch.jit.compile to compile it"
        )
    is_module = isinstance(model, Module)

    if not isinstance(args, tuple):
        args = (args,)

    saved_args = _clone_inputs(args)
    if is_module:
        saved_state = copy.deepcopy(model.state_dict())

    def run_fwd_bwd(args, force_trace=False, assert_compiled=False):
        params = list(model.parameters()) if is_module else []
        in_vars, _ = _flatten((args, params))
        # We use a special API to reset the trace and compile it from scratch.
        compiled_fn = model
        if force_trace:
            compiled_fn.clear_cache()
        if assert_compiled:
            hits = compiled_fn.hits
        out = model(*args)
        if assert_compiled and compiled_fn.hits == hits:
            raise RuntimeError("failed to use the compiled function")
        if not isinstance(out, tuple):
            out = (out,)
        if loss_fn == torch.sum and len(out) != 1:
            raise ValueError(
                (
                    "Model returns {} outputs, but default loss function "
                    "(torch.sum) can only handle a single output"
                ).format(len(out))
            )
        out_vars, _ = _flatten(out)
        saved_outs = [
            v.detach().clone(memory_format=torch.preserve_format) for v in out_vars
        ]
        loss = loss_fn(*out)
        grads = torch.autograd.grad([loss], in_vars)
        # TODO: I'm not sure if the clone here is necessary but it is safer
        saved_grads = [
            v.detach().clone(memory_format=torch.preserve_format) for v in grads
        ]
        return (saved_outs, saved_grads)

    with torch.random.fork_rng(devices, _caller="torch.jit.verify"):
        uncompiled_outs, uncompiled_grads = run_fwd_bwd(args, force_trace=True)
        assert model.has_trace_for(*args)

    if is_module:
        model.load_state_dict(saved_state)
    compiled_outs, compiled_grads = run_fwd_bwd(args, assert_compiled=True)

    _verify_equal(uncompiled_outs, compiled_outs)
    _verify_equal(uncompiled_grads, compiled_grads)


def _verify_equal(xs, ys):
    for x, y in zip(xs, ys):
        if x.sub(y).abs().max() > 1e-6:
            raise RuntimeError("JIT and real computation mismatch")


def indent(s):
    return "\n".join(["\t" + line for line in s.splitlines()])


class TracingCheckError(Exception):
    def __init__(self, graph_diff_error, tensor_compare_error, extra_msg=None):
        self.message = "Tracing failed sanity checks!\n"
        if extra_msg is not None:
            self.message += extra_msg + "\n"
        if graph_diff_error is not None:
            self.message += "ERROR: Graphs differed across invocations!\n"
            self.message += indent(graph_diff_error) + "\n"
        if tensor_compare_error is not None:
            self.message += (
                "ERROR: Tensor-valued Constant nodes differed in value "
                "across invocations. This often indicates that the tracer has"
                " encountered untraceable code.\n"
            )
            self.message += indent(tensor_compare_error) + "\n"
        super(TracingCheckError, self).__init__(self.message)


# Check the traced module against a set of user-provided validation inputs
@torch.no_grad()
def _check_trace(
    check_inputs,
    func,
    traced_func,
    check_tolerance,
    strict,
    force_outplace,
    is_trace_module,
    _module_class,
):
    # Note: tracing is independent of optimizations, which consume the trace
    for inputs in check_inputs:

        if isinstance(inputs, torch.Tensor):
            inputs = (inputs,)

        if is_trace_module:
            copied_dict = {}
            for name, data in inputs.items():
                copied_dict[name] = _clone_inputs(data)
            check_mod = torch.jit.trace_module(
                func.__self__ if hasattr(func, "__self__") else func,
                copied_dict,
                check_trace=False,
                strict=strict,
                _force_outplace=force_outplace,
                _module_class=_module_class,
                _compilation_unit=torch._C.CompilationUnit(),
            )
            check_mod_func = check_mod._c._get_method(traced_func.name)
            inputs = inputs[traced_func.name]
            if isinstance(inputs, (torch.Tensor, dict)):
                inputs = (inputs,)
        else:
            check_mod = torch.jit.trace(
                func,
                _clone_inputs(inputs),
                check_trace=False,
                strict=strict,
                _force_outplace=force_outplace,
                _module_class=_module_class,
            )
            check_mod_func = check_mod

        def graph_diagnostic_info():
            mod_canonicalized = torch._C._jit_pass_canonicalize(traced_func.graph)
            torch._C._jit_pass_inline(mod_canonicalized)
            torch._C._jit_pass_erase_shape_information(mod_canonicalized)
            mod_str = str(mod_canonicalized)
            mod_str = re.sub(r"___torch_mangle_[0-9]+\.", "", mod_str)
            check_canonicalized = torch._C._jit_pass_canonicalize(check_mod_func.graph)
            torch._C._jit_pass_inline(check_canonicalized)
            torch._C._jit_pass_erase_shape_information(check_canonicalized)
            check_str = str(check_canonicalized)
            check_str = re.sub(r"___torch_mangle_[0-9]+\.", "", check_str)

            graph_diff_errors = None
            if mod_str != check_str:
                import difflib

                graph_diff = difflib.ndiff(
                    mod_str.splitlines(True), check_str.splitlines(True)
                )
                graph_diff_errors = "Graph diff:\n" + indent("".join(graph_diff)) + "\n"

                for n_mod, n_check in zip(
                    mod_canonicalized.nodes(), check_canonicalized.nodes()
                ):
                    if str(n_mod) != str(n_check):
                        graph_diff_errors += "First diverging operator:\n"
                        node_diff = difflib.ndiff(
                            str(n_mod).splitlines(True), str(n_check).splitlines(True)
                        )
                        source_printout = (
                            "Node diff:\n" + indent("".join(node_diff)) + "\n"
                        )
                        mod_stack = n_mod.sourceRange()
                        if mod_stack:
                            source_printout += (
                                "Trace source location:\n" + indent(mod_stack) + "\n"
                            )
                        check_stack = n_check.sourceRange()
                        if check_stack:
                            source_printout += (
                                "Check source location:\n" + indent(check_stack) + "\n"
                            )
                        graph_diff_errors += source_printout

                        break  # For now, only print out the first pair of nodes that diverges

            tensor_compare_errors = None
            # Check Tensor-valued constant nodes
            for n_mod, n_check in zip(
                mod_canonicalized.nodes(), check_canonicalized.nodes()
            ):
                if n_mod.kind() != n_check.kind():
                    break  # Graphs have already diverged

                if n_mod.kind() == "prim::Constant" and not (
                    n_mod.mustBeNone() or n_check.mustBeNone()
                ):
                    if not n_mod.hasAttribute("value"):
                        continue
                    if n_mod.kindOf("value") != "t" or n_check.kindOf("value") != "t":
                        continue

                    mod_tensor_val = n_mod.t("value")
                    check_tensor_val = n_check.t("value")

                    try:
                        torch.testing.assert_allclose(mod_tensor_val, check_tensor_val)
                    except (RuntimeError, AssertionError) as e:
                        if tensor_compare_errors is None:
                            tensor_compare_errors = ""
                        tensor_compare_errors += "Node:\n" + indent(str(n_mod)) + "\n"
                        compare_stack = n_mod.sourceRange()
                        if compare_stack:
                            tensor_compare_errors += (
                                "Source Location:\n" + indent(compare_stack) + "\n"
                            )
                        tensor_compare_errors += "Comparison exception: " + indent(
                            str(e)
                        )

                        break  # For now, only print the first diverging pair

            return graph_diff_errors, tensor_compare_errors

        def wrap_retval(x):
            return x if isinstance(x, tuple) else (x,)

        def run_mod_and_filter_tensor_outputs(mod, inputs, running_what):
            try:
                outs = wrap_retval(mod(*_clone_inputs(inputs)))
                outs = [out for out in outs if isinstance(out, torch.Tensor)]
                return outs
            except Exception as e:
                graph_diff_errors, tensor_compare_errors = graph_diagnostic_info()
                msg = f"encountered an exception while running the {running_what} with test inputs.\nException:\n{indent(str(e))}"
                raise TracingCheckError(
                    graph_diff_errors,
                    tensor_compare_errors,
                    extra_msg=msg,
                ) from e

        has_warned = [False]

        def maybe_warn_nondeterministic():
            if has_warned[0]:
                return
            has_warned[0] = True
            nondeterm_ops = [
                op for op in traced_func.graph.nodes() if op.isNondeterministic()
            ]
            if len(nondeterm_ops) > 0:
                nondeterministic_ops_warning = "Trace had nondeterministic nodes. "
                nondeterministic_ops_warning += (
                    "Did you forget call .eval() on your model? Nodes:\n"
                )
                nondeterministic_ops_warning += "\n".join(
                    [indent(str(op)) for op in nondeterm_ops][:20]
                )
                nondeterministic_ops_warning += (
                    "\nThis may cause errors in trace checking. To disable trace checking,"
                    " pass check_trace=False to torch.jit.trace()"
                )
                warnings.warn(
                    nondeterministic_ops_warning, category=TracerWarning, stacklevel=5
                )

        def compare_outputs(original, reference, match_what):
            all_ok = True
            for i, (orig, ref) in enumerate(zip(original, reference)):
                try:
                    if orig.is_quantized:
                        orig = orig.dequantize()
                    if ref.is_quantized:
                        ref = ref.dequantize()
                    torch.testing.assert_allclose(
                        orig.double(),
                        ref.double(),
                        rtol=check_tolerance,
                        atol=torch.testing._get_default_tolerance(orig, ref)[1],
                    )
                except AssertionError as e:
                    maybe_warn_nondeterministic()
                    warnings.warn(
                        "Output nr "
                        + str(i + 1)
                        + ". of the traced function does not match "
                        "the corresponding output of the "
                        + match_what
                        + ". Detailed error:\n"
                        + str(e),
                        category=TracerWarning,
                        stacklevel=4,
                    )
                    all_ok = False

            return all_ok

        traced_outs = run_mod_and_filter_tensor_outputs(traced_func, inputs, "trace")
        fn_outs = run_mod_and_filter_tensor_outputs(func, inputs, "Python function")
        if compare_outputs(traced_outs, fn_outs, "Python function"):
            check_outs = run_mod_and_filter_tensor_outputs(
                check_mod_func, inputs, "repeated trace"
            )
            compare_outputs(traced_outs, check_outs, "repeated trace")

        diag_info = graph_diagnostic_info()
        if any(info is not None for info in diag_info):
            raise TracingCheckError(*diag_info)


class TracerWarning(Warning):
    @staticmethod
    def ignore_lib_warnings():
        # We ignore warnings from all submodules excluding the JIT, because we need them e.g. for _check_trace
        warnings.filterwarnings(
            "ignore", category=TracerWarning, module="torch.(?!jit)"
        )


# We ignore the tracer warnings coming form inside the library, because all our shape
# checks in nn will trigger them.
TracerWarning.ignore_lib_warnings()
torch._C._tracer_warn_use_python()


def make_tuple(example_inputs):
    if isinstance(example_inputs, (torch.Tensor, dict)):
        return (example_inputs,)
    # done primarily so that weird iterables fail here and not pybind11 code
    if not isinstance(example_inputs, tuple):
        return tuple(example_inputs)
    return example_inputs


def make_module(mod, _module_class, _compilation_unit):
    if isinstance(mod, ScriptModule):
        return mod
    elif torch._jit_internal.module_has_exports(mod):

        infer_methods_stubs_fn = torch.jit._recursive.make_stubs_from_exported_methods
        return torch.jit._recursive.create_script_module(
            mod,
            infer_methods_stubs_fn,
            share_types=False
        )
    else:
        if _module_class is None:
            _module_class = TopLevelTracedModule
        return _module_class(mod, _compilation_unit=_compilation_unit)


def wrap_check_inputs(check_inputs):
    if check_inputs is None:
        return None

    return [{"forward": c} for c in check_inputs]


def trace(
    func,
    example_inputs,
    optimize=None,
    check_trace=True,
    check_inputs=None,
    check_tolerance=1e-5,
    strict=True,
    _force_outplace=False,
    _module_class=None,
    _compilation_unit=_python_cu,
):
    """
    Trace a function and return an executable  or :class:`ScriptFunction`
    that will be optimized using just-in-time compilation. Tracing is ideal for
    code that operates only on ``Tensor``\\s and lists, dictionaries, and
    tuples of ``Tensor``\\s.

    Using `torch.jit.trace` and `torch.jit.trace_module`, you can turn an
    existing module or Python function into a TorchScript
    :class:`ScriptFunction` or :class:`ScriptModule`. You must provide example
    inputs, and we run the function, recording the operations performed on all
    the tensors.

    * The resulting recording of a standalone function produces `ScriptFunction`.
    * The resulting recording of `nn.Module.forward` or `nn.Module` produces
      `ScriptModule`.

    This module also contains any parameters that the original
    module had as well.

    Warning:
        Tracing only correctly records functions and modules which are not data
        dependent (e.g., do not have conditionals on data in tensors) and do not have
        any untracked external dependencies (e.g., perform input/output or
        access global variables). Tracing only records operations done when the given
        function is run on the given tensors. Therefore, the returned
        `ScriptModule` will always run the same traced graph on any input. This
        has some important implications when your module is expected to run
        different sets of operations, depending on the input and/or the module
        state. For example,

        * Tracing will not record any control-flow like if-statements or loops.
          When this control-flow is constant across your module, this is fine
          and it often inlines the control-flow decisions. But sometimes the
          control-flow is actually part of the model itself. For instance, a
          recurrent network is a loop over the (possibly dynamic) length of an
          input sequence.
        * In the returned :class:`ScriptModule`, operations that have different
          behaviors in ``training`` and ``eval`` modes will always behave as if
          it is in the mode it was in during tracing, no matter which mode the
          `ScriptModule` is in.

        In cases like these, tracing would not be appropriate and
        :func:`scripting <torch.jit.script>` is a better choice. If you trace
        such models, you may silently get incorrect results on subsequent
        invocations of the model. The tracer will try to emit warnings when
        doing something that may cause an incorrect trace to be produced.

    Arguments:
        func (callable or torch.nn.Module):  A Python function or `torch.nn.Module`
            that will be run with `example_inputs`. `func` arguments and return
            values  must be tensors or (possibly nested) tuples that contain
            tensors. When a module is passed `torch.jit.trace`, only the
            ``forward`` method is run and traced (see :func:`torch.jit.trace
            <torch.jit.trace_module>` for details).
        example_inputs (tuple or torch.Tensor):  A tuple of example inputs that
            will be passed to the function while tracing. The resulting trace
            can be run with inputs of different types and shapes assuming the
            traced operations support those types and shapes. `example_inputs`
            may also be a single Tensor in which case it is automatically
            wrapped in a tuple.

    Keyword arguments:
        check_trace (``bool``, optional): Check if the same inputs run through
            traced code produce the same outputs. Default: ``True``. You might want
            to disable this if, for example, your network contains non-
            deterministic ops or if you are sure that the network is correct despite
            a checker failure.

        check_inputs (list of tuples, optional): A list of tuples of input
            arguments that should be used to check the trace against what is
            expected. Each tuple is equivalent to a set of input arguments that
            would be specified in ``example_inputs``. For best results, pass in
            a set of checking inputs representative of the space of shapes and
            types of inputs you expect the network to see.  If not specified,
            the original ``example_inputs`` are used for checking
        check_tolerance (float, optional): Floating-point comparison tolerance
            to use in the checker procedure.  This can be used to relax the
            checker strictness in the event that results diverge numerically
            for a known reason, such as operator fusion.
        strict (``bool``, optional): run the tracer in a strict mode or not
            (default: ``True``). Only turn this off when you want the tracer to
            record your mutable container types (currently ``list``/``dict``)
            and you are sure that the container you are using in your
            problem is a ``constant`` structure and does not get used as
            control flow (if, for) conditions.

    Returns:
        If `func` is `nn.Module` or ``forward`` of `nn.Module`, `trace` returns
        a :class:`ScriptModule` object with a single ``forward`` method
        containing the traced code.  The returned `ScriptModule` will
        have the same set of sub-modules and parameters as the original
        ``nn.Module``.  If ``func`` is a standalone function, ``trace``
        returns `ScriptFunction`.

    Example (tracing a function):

    .. testcode::

        import torch

        def foo(x, y):
            return 2 * x + y

        # Run `foo` with the provided inputs and record the tensor operations
        traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))

        # `traced_foo` can now be run with the TorchScript interpreter or saved
        # and loaded in a Python-free environment

    Example (tracing an existing module)::

        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(self, x):
                return self.conv(x)

        n = Net()
        example_weight = torch.rand(1, 1, 3, 3)
        example_forward_input = torch.rand(1, 1, 3, 3)

        # Trace a specific method and construct `ScriptModule` with
        # a single `forward` method
        module = torch.jit.trace(n.forward, example_forward_input)

        # Trace a module (implicitly traces `forward`) and construct a
        # `ScriptModule` with a single `forward` method
        module = torch.jit.trace(n, example_forward_input)

    """
    if not _enabled:
        return func
    if optimize is not None:
        warnings.warn(
            "`optimize` is deprecated and has no effect. Use `with torch.jit.optimized_execution() instead"
        )

    if isinstance(func, torch.jit.ScriptModule):
        # it is hard to trace it because the forward method on ScriptModule is already defined, so it
        # would result in an error.
        warnings.warn(
            "The input to trace is already a ScriptModule, tracing it is a no-op. Returning the object as is."
        )
        return func

    if isinstance(func, torch.nn.Module):
        return trace_module(
            func,
            {"forward": example_inputs},
            None,
            check_trace,
            wrap_check_inputs(check_inputs),
            check_tolerance,
            strict,
            _force_outplace,
            _module_class,
        )

    if (
        hasattr(func, "__self__")
        and isinstance(func.__self__, torch.nn.Module)
        and func.__name__ == "forward"
    ):
        return trace_module(
            func.__self__,
            {"forward": example_inputs},
            None,
            check_trace,
            wrap_check_inputs(check_inputs),
            check_tolerance,
            strict,
            _force_outplace,
            _module_class,
        )

    # Special case for common case of passing a single Tensor
    if isinstance(example_inputs, (torch.Tensor, dict)):
        example_inputs = (example_inputs,)
    # done primarily so that weird iterables fail here and not pybind11 code
    elif not isinstance(example_inputs, tuple):
        example_inputs = tuple(example_inputs)

    var_lookup_fn = _create_interpreter_name_lookup_fn(0)

    if hasattr(func, "__self__") and isinstance(func.__self__, torch.nn.Module):
        raise AttributeError(
            "trace doesn't support compiling individual module's functions.\n"
            "Please use trace_module"
        )

    name = _qualified_name(func)
    traced = torch._C._create_function_from_trace(
        name, func, example_inputs, var_lookup_fn, strict, _force_outplace
    )

    # Check the trace against new traces created from user-specified inputs
    if check_trace:
        if check_inputs is not None:
            _check_trace(
                check_inputs,
                func,
                traced,
                check_tolerance,
                strict,
                _force_outplace,
                False,
                _module_class,
            )
        else:
            _check_trace(
                [example_inputs],
                func,
                traced,
                check_tolerance,
                strict,
                _force_outplace,
                False,
                _module_class,
            )

    return traced


_trace_module_map: Optional[Dict[Any, Any]] = None


def trace_module(
    mod,
    inputs,
    optimize=None,
    check_trace=True,
    check_inputs=None,
    check_tolerance=1e-5,
    strict=True,
    _force_outplace=False,
    _module_class=None,
    _compilation_unit=_python_cu,
):
    """
    Trace a module and return an executable :class:`ScriptModule` that will be optimized
    using just-in-time compilation. When a module is passed to :func:`torch.jit.trace <torch.jit.trace>`, only
    the ``forward`` method is run and traced. With ``trace_module``, you can specify a dictionary of
    method names to example inputs to trace (see the ``inputs``) argument below.

    See :func:`torch.jit.trace <torch.jit.trace>` for more information on tracing.

    Arguments:
        mod (torch.nn.Module):  A ``torch.nn.Module`` containing methods whose names are
                                specified in ``inputs``. The given methods will be compiled
                                as a part of a single `ScriptModule`.
        inputs (dict):  A dict containing sample inputs indexed by method names in ``mod``.
                                The inputs will be passed to methods whose names correspond to inputs'
                                keys while tracing.
                                ``{ 'forward' : example_forward_input, 'method2': example_method2_input}``
    Keyword arguments:
        check_trace (``bool``, optional): Check if the same inputs run through
                                      traced code produce the same outputs. Default: ``True``. You might want
                                      to disable this if, for example, your network contains non-
                                      deterministic ops or if you are sure that the network is correct despite
                                      a checker failure.

        check_inputs (list of dicts, optional): A list of dicts of input arguments that should be used
                                                 to check the trace against what is expected. Each tuple
                                                 is equivalent to a set of input arguments that would
                                                 be specified in ``inputs``. For best results, pass in a
                                                 set of checking inputs representative of the space of
                                                 shapes and types of inputs you expect the network to see.
                                                 If not specified, the original ``inputs`` are used for checking
        check_tolerance (float, optional): Floating-point comparison tolerance to use in the checker procedure.
                                           This can be used to relax the checker strictness in the event that
                                           results diverge numerically for a known reason, such as operator fusion.

    Returns:
        A :class:`ScriptModule` object with a single ``forward`` method containing the traced code.
        When ``func`` is a ``torch.nn.Module``, the returned :class:`ScriptModule` will have the same set of
        sub-modules and parameters as ``func``.

    Example (tracing a module with multiple methods)::

        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(self, x):
                return self.conv(x)

            def weighted_kernel_sum(self, weight):
                return weight * self.conv.weight


        n = Net()
        example_weight = torch.rand(1, 1, 3, 3)
        example_forward_input = torch.rand(1, 1, 3, 3)

        # Trace a specific method and construct `ScriptModule` with
        # a single `forward` method
        module = torch.jit.trace(n.forward, example_forward_input)

        # Trace a module (implicitly traces `forward`) and construct a
        # `ScriptModule` with a single `forward` method
        module = torch.jit.trace(n, example_forward_input)

        # Trace specific methods on a module (specified in `inputs`), constructs
        # a `ScriptModule` with `forward` and `weighted_kernel_sum` methods
        inputs = {'forward' : example_forward_input, 'weighted_kernel_sum' : example_weight}
        module = torch.jit.trace_module(n, inputs)

    """
    if not _enabled:
        return mod
    if optimize is not None:
        warnings.warn(
            "`optimize` is deprecated and has no effect. Use `with torch.jit.optimized_execution() instead"
        )

    var_lookup_fn = _create_interpreter_name_lookup_fn(0)

    if not isinstance(mod, torch.nn.Module):
        raise AttributeError("expected torch.nn.Module as the first argument")

    if not isinstance(inputs, dict):
        raise AttributeError("expected a dictionary of (method_name, input) pairs")

    old_module_map = torch.jit._trace._trace_module_map
    try:
        trace_module_map: Dict[Any, Any] = {}

        def register_submods(mod, prefix):
            for name, child in mod.named_children():
                submod_qualname = prefix + "." + name
                trace_module_map[child] = submod_qualname
                register_submods(child, submod_qualname)

        trace_module_map["__module"] = mod
        torch.jit._trace._trace_module_map = trace_module_map
        register_submods(mod, "__module")

        module = make_module(mod, _module_class, _compilation_unit)

        for method_name, example_inputs in inputs.items():
            # this is needed since Module.__call__ sets up some extra tracing
            func = mod if method_name == "forward" else getattr(mod, method_name)
            example_inputs = make_tuple(example_inputs)
            module._c._create_method_from_trace(
                method_name,
                func,
                example_inputs,
                var_lookup_fn,
                strict,
                _force_outplace,
            )
            check_trace_method = module._c._get_method(method_name)

            # Check the trace against new traces created from user-specified inputs
            if check_trace:
                if check_inputs is not None:
                    _check_trace(
                        check_inputs,
                        func,
                        check_trace_method,
                        check_tolerance,
                        strict,
                        _force_outplace,
                        True,
                        _module_class,
                    )
                else:
                    _check_trace(
                        [inputs],
                        func,
                        check_trace_method,
                        check_tolerance,
                        strict,
                        _force_outplace,
                        True,
                        _module_class,
                    )
    finally:
        torch.jit._trace._trace_module_map = old_module_map

    return module


def is_tracing():
    """
    Returns ``True`` in tracing (if a function is called during the tracing of
    code with ``torch.jit.trace``) and ``False`` otherwise.
    """
    return torch._C._is_tracing()


class TracedModule(ScriptModule):
    _disable_script_meta = True

    def __init__(self, orig, id_set=None, _compilation_unit=None):
        # XXX: orig can be a nn.Module or a function!
        super(TracedModule, self).__init__()
        assert isinstance(orig, torch.nn.Module)

        # Copy a subset of `orig` to a temporary nn.Module.
        # This is a way to customize what will actually get compiled by create_script_module
        id_set = set()

        # This allows us to preserve the original module's qualified name by defining a new
        # type with the attribute _jit_override_qualname. In torch._jit_internal._qualified_name
        # we have a special case that will look up this attribute to override whatever qualname
        # we would get from the python type system
        class QualnameWrapper(torch.nn.Module):
            pass

        QualnameWrapper._jit_override_qualname = torch._jit_internal._qualified_name(  # type: ignore
            type(orig)
        )

        tmp_module = QualnameWrapper()

        def check_unique(param):
            if param in id_set:
                raise ValueError(
                    "TracedModules don't support parameter sharing between modules"
                )
            id_set.add(param)

        tmp_module.training = orig.training

        for name, param in orig._parameters.items():
            if param is not None:
                tmp_module._parameters[name] = param
                check_unique(param)
        for name, buf in orig._buffers.items():
            if buf is not None:
                tmp_module._buffers[name] = buf
                check_unique(buf)
        for name, val in orig.__dict__.items():
            if (
                torch._C._jit_is_script_object(val)
                and name not in orig._parameters
                and name not in orig._buffers
            ):
                setattr(tmp_module, name, val)

        if orig._backward_hooks:
            raise ValueError(
                "Modules that have backward hooks assigned can't be compiled: "
                + str(orig)
            )

        for name, submodule in orig._modules.items():
            tmp_module._modules[name] = make_module(
                submodule, TracedModule, _compilation_unit=None
            )

        script_module = torch.jit._recursive.create_script_module(
            tmp_module, lambda module: (), share_types=False
        )

        self.__dict__["_name"] = type(orig).__name__
        self.__dict__["_actual_script_module"] = script_module
        for name in ("_parameters", "_buffers", "_modules"):
            delattr(self, name)

    def forward(self, *args, **kwargs):
        raise RuntimeError("Trace submodules cannot be called.")

    def __getattr__(self, attr):
        if "_actual_script_module" not in self.__dict__:
            return super(TracedModule, self).__getattr__(attr)
        return getattr(self._actual_script_module, attr)

    def __setattr__(self, attr, value):
        if "_actual_script_module" not in self.__dict__:
            return super(TracedModule, self).__setattr__(attr, value)
        setattr(self._actual_script_module, attr, value)

    def _get_name(self):
        return self._name

    def extra_repr(self):
        return "original_name={}".format(self._name)


class TopLevelTracedModule(TracedModule):
    forward = _CachedForward()

    def _reconstruct(self, cpp_module):
        """
        Re-construct an instance of TopLevelTracedModule using an instance of a C++ module.

        Arguments:
            cpp_module: The C++ module that this TopLevelTracedModule will be rebuilt around.
        """
        self.__dict__["_actual_script_module"]._reconstruct(cpp_module)


def _script_if_tracing(fn):
    """
    Compiles ``fn`` when it is first called during tracing. ``torch.jit.script``
    has a non-negligible start up time when it is first called due to
    lazy-initializations of many compiler builtins. Therefore you should not use
    it in library code. However, you may want to have parts of your library work
    in tracing even if they use control flow. In these cases, you should use
    ``@torch.jit._script_if_tracing`` to substitute for
    ``torch.jit.script``.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not is_tracing():
            # Not tracing, don't do anything
            return fn(*args, **kwargs)

        compiled_fn = script(wrapper.__original_fn)  # type: ignore
        return compiled_fn(*args, **kwargs)

    wrapper.__original_fn = fn  # type: ignore
    wrapper.__script_if_tracing_wrapper = True  # type: ignore

    return wrapper


def _get_trace_graph(f, args=(), kwargs=None, strict=True, _force_outplace=False,
                     return_inputs=False, _return_inputs_states=False):
    """
    .. warning::
        This function is internal-only and should only be used by the ONNX
        exporter. If you are trying to get a graph through tracing, please go
        through the public API instead::

            trace = torch.jit.trace(nn.LSTMCell(), (input, hidden))
            trace_graph = trace.graph

    Trace a function or model, returning a tuple consisting of the both the
    *trace* of an execution, as well as the original return value. If return_inputs,
    also returns the trace inputs as part of the tuple

    Tracing is guaranteed not to change the semantics of the function/module
    that is traced.

    Arguments:
        f (torch.nn.Module or function): the function or module
            to be traced.
        args (tuple or Tensor): the positional arguments to pass to the
            function/module to be traced.  A non-tuple is assumed to
            be a single positional argument to be passed to the model.
        kwargs (dict): the keyword arguments to pass to the function/module
            to be traced.

    Example (trace a cell):

    .. testcode::

        trace = torch.jit.trace(nn.LSTMCell(), (input, hidden))
    """
    if kwargs is None:
        kwargs = {}
    if not isinstance(args, tuple):
        args = (args,)
    outs = ONNXTracedModule(f, strict, _force_outplace, return_inputs, _return_inputs_states)(*args, **kwargs)
    return outs
