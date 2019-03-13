import torch._C
from torch import Tensor
from torch.autograd import Variable, function
from torch.serialization import validate_cuda_device
from torch.nn import Module, ModuleList, ParameterList, Parameter, Sequential
import torch.backends.cudnn as cudnn
import torch.jit.annotations
from torch._six import string_classes
from ..nn.modules.utils import _single, _pair, _triple, _quadruple, \
    _list_with_default
import torch.testing
from torch._jit_internal import ignore, _enabled, script, script_method, ScriptModule, \
    _try_get_weak_module, _try_compile_weak_script, _try_get_dispatched_fn, \
    _try_get_overloaded_fn, _try_get_ignored_op, CompilationUnit, ScriptClass, \
    _ConstModuleList, Attribute


import math
from collections import defaultdict, OrderedDict, namedtuple
import textwrap
import sys
import warnings
import itertools
import weakref
import types
import contextlib
import os
import functools
import copy
import numbers
import collections
import re
import inspect
import pickle
if sys.version_info[0] > 2:
    import pathlib


_flatten = torch._C._jit_flatten
_unflatten = torch._C._jit_unflatten
BatchTensor = torch._C._jit.BatchTensor

Future = torch._C.Future
_fork = torch._C.fork
_wait = torch._C.wait


@contextlib.contextmanager
def scope(scope_name):
    tracing_state = torch._C._get_tracing_state()
    if tracing_state:
        tracing_state.push_scope(scope_name)
    try:
        yield
    finally:
        if tracing_state:
            tracing_state.pop_scope()


DEFAULT_EXTRA_FILES_MAP = torch._C.ExtraFilesMap()


def load(f, map_location=None, _extra_files=DEFAULT_EXTRA_FILES_MAP):
    r"""
        Load a ``ScriptModule`` previously saved with :func:`save <torch.jit.save>`

        All previously saved modules, no matter their device, are first loaded onto CPU,
        and then are moved to the devices they were saved from. If this fails (e.g. because
        the run time system doesn't have certain devices), an exception is raised.
        However, storages can be dynamically remapped to an alternative set of devices
        using the `map_location` argument. Comparing to :func:`torch.load`, `map_location`
        in this function is simplified, which only accepts a string (e.g., 'cpu', 'cuda:0'),
        or torch.device (e.g., torch.device('cpu'))

        Arguments:
            f: a file-like object (has to implement read, readline, tell, and seek),
                or a string containing a file name
            map_location: can a string (e.g., 'cpu', 'cuda:0'), a device (e.g.,
                torch.device('cpu'))
            _extra_files: map from filename to content. The extra
                filenames given in the map would be loaded and their content
                would be stored in the provided map.


        Returns:
            A ``ScriptModule`` object.

        Example:
            >>> torch.jit.load('scriptmodule.pt')
            # Load ScriptModule from io.BytesIO object
            >>> with open('scriptmodule.pt', 'rb') as f:
                    buffer = io.BytesIO(f.read())
            # Load all tensors to the original device
            >>> torch.jit.load(buffer)
            # Load all tensors onto CPU, using a device
            >>> torch.jit.load(buffer, map_location=torch.device('cpu'))
            # Load all tensors onto CPU, using a string
            >>> torch.jit.load(buffer, map_location='cpu')
            # Load with extra files.
            >>> files = {'metadata.json' : ''}
            >>> torch.jit.load('scriptmodule.pt', _extra_files = files)
            >>> print (files['metadata.json'])
    """
    m = ScriptModule()

    def module_lookup(names):
        curr = m
        for name in names:
            if not hasattr(curr, name):
                setattr(curr, name, ScriptModule())
            curr = getattr(curr, name)
        return curr
    if isinstance(f, string_classes):
        if not os.path.exists(f):
            raise ValueError("The provided filename {} does not exist".format(f))
    if isinstance(map_location, string_classes):
        map_location = torch.device(map_location)
    elif not (map_location is None or
              isinstance(map_location, torch.device)):
        raise ValueError("map_location should be either None, string or torch.device, "
                         "but got type: " + str(type(map_location)))
    if (str(map_location).startswith('cuda')):
        validate_cuda_device(map_location)

    if isinstance(f, str) or \
            (sys.version_info[0] == 2 and isinstance(f, unicode)) or \
            (sys.version_info[0] == 3 and isinstance(f, pathlib.Path)):
        torch._C.import_ir_module(module_lookup, f, map_location, _extra_files)
    else:
        torch._C.import_ir_module_from_buffer(module_lookup, f.read(), map_location, _extra_files)

    return m


def save(m, f, _extra_files=DEFAULT_EXTRA_FILES_MAP):
    """
        Saves a ScriptModule to a file.

        Args:
            m: a ScriptModule to save
            f: a file-like object (has to implement write and flush) or a string
               containing a file name
            _extra_files: Map from filename to contents which will be stored as part of 'f'

        .. warning::
            If you are using Python 2, torch.save does NOT support StringIO.StringIO
            as a valid file-like object. This is because the write method should return
            the number of bytes written; StringIO.write() does not do this.

            Please use something like io.BytesIO instead.

        Example:
            >>> m = torch.jit.ScriptModule()
            >>> # Save to file
            >>> torch.jit.save(m, 'scriptmodule.pt')
            >>> # Save to io.BytesIO buffer
            >>> buffer = io.BytesIO()
            >>> torch.jit.save(m, buffer)
            >>> # Save with extra files
            >>> extra_files = torch._C.ExtraFilesMap()
            >>> extra_files['foo.txt'] = 'bar'
            >>> torch.jit.save(m, 'scriptmodule.pt', _extra_files=extra_files)
    """
    if isinstance(f, str) or \
            (sys.version_info[0] == 2 and isinstance(f, unicode)) or \
            (sys.version_info[0] == 3 and isinstance(f, pathlib.Path)):
        m.save(f, _extra_files=_extra_files)
    else:
        ret = m.save_to_buffer(_extra_files=_extra_files)
        f.write(ret)


def get_trace_graph(f, args=(), kwargs=None, _force_outplace=False, return_inputs=False):
    """
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

    Example: Trace a cell.

        >>> trace, out = jit.trace(nn.LSTMCell(), (input, hidden))
        >>> print(trace)
    """
    if kwargs is None:
        kwargs = {}
    if not isinstance(args, tuple):
        args = (args,)
    return LegacyTracedModule(f, _force_outplace, return_inputs)(*args, **kwargs)


def _unique_state_dict(module, keep_vars=False):
    state_dict = module.state_dict(keep_vars=keep_vars)
    filtered_dict = type(state_dict)()
    seen_ids = set()
    for k, v in state_dict.items():
        if id(v) in seen_ids:
            continue
        seen_ids.add(id(v))
        filtered_dict[k] = v
    return filtered_dict


def _create_interpreter_name_lookup_fn(frames_up=1):
    def _get_interpreter_name_for_var(var):
        frame = inspect.currentframe()
        i = 0
        while i < frames_up + 1:
            frame = frame.f_back
            i += 1

        f_locals = frame.f_locals
        f_globals = frame.f_globals

        for k, v in f_locals.items():
            if isinstance(v, torch.Tensor) and var is v:
                return k if k != 'self' else ''
        for k, v in f_globals.items():
            if isinstance(v, torch.Tensor) and var is v:
                return k if k != 'self' else ''
        return ''
    return _get_interpreter_name_for_var


class LegacyTracedModule(Module):
    def __init__(self, inner, force_outplace=False, return_inputs=False):
        super(LegacyTracedModule, self).__init__()
        # inner may be a Module, or it may be an arbitrary callable
        # If it's a Module, we get its parameters automatically, which lets
        # us avoid a special casing functions versus modules.
        self.inner = inner
        self._force_outplace = force_outplace
        self._return_inputs = return_inputs

    def forward(self, *args):
        in_vars, in_desc = _flatten(args)
        # NOTE: use full state, because we need it for BatchNorm export
        # This differs from the compiler path, which doesn't support it at the moment.
        module_state = list(_unique_state_dict(self, keep_vars=True).values())
        trace, all_trace_inputs = torch._C._tracer_enter(*(in_vars + module_state))
        ret_inputs = tuple(x.clone() for x in all_trace_inputs)
        torch._C._tracer_set_force_outplace(self._force_outplace)
        torch._C._tracer_set_get_unique_name_fn(_create_interpreter_name_lookup_fn())
        try:
            trace_inputs = _unflatten(all_trace_inputs[:len(in_vars)], in_desc)
            out = self.inner(*trace_inputs)
            out_vars, _ = _flatten(out)
            torch._C._tracer_exit(tuple(out_vars))
        except Exception:
            torch._C._tracer_abandon()
            raise
        if self._return_inputs:
            return trace, out, ret_inputs
        else:
            return trace, out


def _clone_inputs(args):
    def clone_input(a):
        if a is None:
            return None
        elif isinstance(a, torch.Tensor):
            # TODO: figure out one liner to .clone() and set requires_grad
            v = Variable(a.data.clone(), requires_grad=a.requires_grad)
            if a.grad is not None:
                v.grad = clone_input(v.grad)
            return v
        else:
            return a.clone()
    return function._nested_map(lambda x: isinstance(x, torch.Tensor),
                                clone_input, condition_msg="tensors")(args)


# This is purely for developer debugging.  We are not going to advertise it.
_JIT_DUMP = os.environ.get('PYTORCH_JIT_DUMP', False)
_JIT_TIME = os.environ.get('PYTORCH_JIT_TIME', False)  # CUDA-only timing
_JIT_DISABLE = os.environ.get('PYTORCH_JIT_DISABLE', False)
_JIT_STATS = os.environ.get('PYTORCH_JIT_STATS', False)


def _dump_trace(trace_name, pass_name, input_key, trace):
    if not _JIT_DUMP:
        return

    import torch.contrib._graph_vis as graph_vis

    filename = "{}_{}".format(trace_name, pass_name)
    # TODO: Also paste out the backtrace when the trace was compiled
    # (and maybe also when it was run?)
    with open(filename + ".ir", "w") as f:
        f.write("Input key: {}\n\n{}".format(input_key, str(trace)))
    graph_vis.write(trace.graph(), filename + ".html")


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
    if not isinstance(model, torch._C.CompiledFunction):
        raise TypeError("Cannot verify an uncompiled module.  Add @torch.jit.compile to compile it")
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
            out = (out, )
        if loss_fn == torch.sum and len(out) != 1:
            raise ValueError(("Model returns {} outputs, but default loss function "
                              "(torch.sum) can only handle a single output").format(len(out)))
        out_vars, _ = _flatten(out)
        saved_outs = [v.data.clone() for v in out_vars]
        loss = loss_fn(*out)
        grads = torch.autograd.grad([loss], in_vars)
        # TODO: I'm not sure if the clone here is necessary but it is safer
        saved_grads = [v.data.clone() for v in grads]
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
    return '\n'.join(['\t' + line for line in s.splitlines()])


class TracingCheckError(Exception):
    def __init__(self, graph_diff_error, tensor_compare_error, extra_msg=None):
        self.message = 'Tracing failed sanity checks!\n'
        if extra_msg is not None:
            self.message += extra_msg + '\n'
        if graph_diff_error is not None:
            self.message += 'ERROR: Graphs differed across invocations!\n'
            self.message += indent(graph_diff_error) + '\n'
        if tensor_compare_error is not None:
            self.message += 'ERROR: Tensor-valued Constant nodes differed in value ' \
                            'across invocations. This often indicates that the tracer has' \
                            ' encountered untraceable code.\n'
            self.message += indent(tensor_compare_error) + '\n'
        super(TracingCheckError, self).__init__(self.message)


# Check the traced module against a set of user-provided validation inputs
@torch.no_grad()
def _check_trace(check_inputs, func, executor_options, module, check_tolerance, force_outplace):
    # Note: tracing is independent of optimizations, which consume the trace
    executor_options['optimize'] = False
    for inputs in check_inputs:
        if isinstance(inputs, torch.Tensor):
            inputs = (inputs,)
        check_mod = torch.jit.trace(
            func,
            _clone_inputs(inputs),
            check_trace=False,
            _force_outplace=force_outplace,
            **executor_options)

        def graph_diagnostic_info():
            mod_canonicalized = torch._C._jit_pass_canonicalize(module.graph)
            torch._C._jit_pass_erase_shape_information(mod_canonicalized)
            check_canonicalized = torch._C._jit_pass_canonicalize(check_mod.graph)
            torch._C._jit_pass_erase_shape_information(check_canonicalized)

            graph_diff_errors = None
            if str(mod_canonicalized) != str(check_canonicalized):
                import difflib
                graph_diff = difflib.ndiff(str(mod_canonicalized).splitlines(True),
                                           str(check_canonicalized).splitlines(True))
                graph_diff_errors = 'Graph diff:\n' + indent(''.join(graph_diff)) + '\n'

                for n_mod, n_check in zip(mod_canonicalized.nodes(), check_canonicalized.nodes()):
                    if str(n_mod) != str(n_check):
                        graph_diff_errors += 'First diverging operator:\n'
                        node_diff = difflib.ndiff(str(n_mod).splitlines(True),
                                                  str(n_check).splitlines(True))
                        source_printout = 'Node diff:\n' + indent(''.join(node_diff)) + '\n'
                        mod_stack = n_mod.getSourceLocation()
                        if mod_stack:
                            source_printout += 'Trace source location:\n' + indent(mod_stack) + '\n'
                        check_stack = n_check.getSourceLocation()
                        if check_stack:
                            source_printout += 'Check source location:\n' + indent(check_stack) + '\n'
                        graph_diff_errors += source_printout

                        break  # For now, only print out the first pair of nodes that diverges

            tensor_compare_errors = None
            # Check Tensor-valued constant nodes
            for n_mod, n_check in zip(mod_canonicalized.nodes(), check_canonicalized.nodes()):
                if n_mod.kind() != n_check.kind():
                    break  # Graphs have already diverged

                if n_mod.kind() == 'prim::Constant' and not (n_mod.mustBeNone() or n_check.mustBeNone()):
                    if n_mod.kindOf('value') != 't' or n_check.kindOf('value') != 't':
                        continue

                    mod_tensor_val = n_mod.t('value')
                    check_tensor_val = n_check.t('value')

                    try:
                        torch.testing.assert_allclose(mod_tensor_val, check_tensor_val)
                    except (RuntimeError, AssertionError) as e:
                        if tensor_compare_errors is None:
                            tensor_compare_errors = ''
                        tensor_compare_errors += 'Node:\n' + indent(str(n_mod)) + '\n'
                        compare_stack = n_mod.getSourceLocation()
                        if compare_stack:
                            tensor_compare_errors += 'Source Location:\n' + indent(compare_stack) + '\n'
                        tensor_compare_errors += 'Comparison exception: ' + indent(str(e))

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
                raise TracingCheckError(*graph_diagnostic_info(),
                                        extra_msg='Encountered an exception while running the ' + running_what +
                                                  ' with test inputs.\nException:\n' + indent(str(e)))

        has_warned = [False]

        def maybe_warn_nondeterministic():
            if has_warned[0]:
                return
            has_warned[0] = True
            nondeterm_ops = [op for op in module.graph.nodes() if op.isNondeterministic()]
            if len(nondeterm_ops) > 0:
                nondeterministic_ops_warning = "Trace had nondeterministic nodes. Nodes:\n"
                nondeterministic_ops_warning += "\n".join([indent(str(op)) for op in nondeterm_ops][:20])
                nondeterministic_ops_warning += "\nThis may cause errors in trace checking. To disable trace checking,"\
                                                " pass check_trace=False to torch.jit.trace()"
                warnings.warn(nondeterministic_ops_warning, category=TracerWarning, stacklevel=5)

        def compare_outputs(original, reference, match_what):
            all_ok = True
            for i, (orig, ref) in enumerate(zip(original, reference)):
                try:
                    torch.testing.assert_allclose(orig.double(), ref.double(), rtol=check_tolerance,
                                                  atol=torch.testing._get_default_tolerance(orig, ref)[1])
                except AssertionError as e:
                    maybe_warn_nondeterministic()
                    warnings.warn('Output nr ' + str(i + 1) + '. of the traced function does not match '
                                  'the corresponding output of the ' + match_what + '. Detailed error:\n' + str(e),
                                  category=TracerWarning, stacklevel=4)
                    all_ok = False

            return all_ok

        traced_outs = run_mod_and_filter_tensor_outputs(module, inputs, 'trace')
        fn_outs = run_mod_and_filter_tensor_outputs(func, inputs, 'Python function')
        if compare_outputs(traced_outs, fn_outs, 'Python function'):
            check_outs = run_mod_and_filter_tensor_outputs(check_mod, inputs, 'repeated trace')
            compare_outputs(traced_outs, check_outs, 'repeated trace')

        diag_info = graph_diagnostic_info()
        if any(info is not None for info in diag_info):
            raise TracingCheckError(*diag_info)


class TracerWarning(Warning):
    @staticmethod
    def ignore_lib_warnings():
        # We ignore warnings from all submodules excluding the JIT, because we need them e.g. for _check_trace
        warnings.filterwarnings('ignore', category=TracerWarning, module='torch.(?!jit)')


# We ignore the tracer warnings coming form inside the library, because all our shape
# checks in nn will trigger them.
TracerWarning.ignore_lib_warnings()
torch._C._tracer_warn_use_python()


def trace(func,
          example_inputs,
          optimize=True,
          check_trace=True,
          check_inputs=None,
          check_tolerance=1e-5,
          _force_outplace=False,
          _module_class=None):
    """
    Trace a function and return an executable trace that will be optimized
    using just-in-time compilation.

    .. warning::

        Tracing only correctly records functions and modules which are not data
        dependent (e.g., have conditionals on data in tensors) and do not have
        any untracked external dependencies (e.g., perform input/output or
        access global variables). If you trace such models, you may silently get
        incorrect results on subsequent invocations of the model. The tracer
        will try to emit warnings when doing something that may cause an
        incorrect trace to be produced.

    Arguments:
        func (callable or torch.nn.Module):  a python function or torch.nn.Module
                                             that will be run with example_inputs.
                                             arguments and returns to func must be Tensors
                                             or (possibly nested) tuples that
                                             contain tensors.
        example_inputs (tuple):  a tuple of example inputs that will be passed to the function
                                 while tracing. The resulting trace can be run with
                                 inputs of different types and shapes assuming the traced operations
                                 support those types and shapes. example_inputs may also be a single
                                 Tensor in which case it is automatically wrapped in a tuple

    Keyword arguments:
        optimize (bool, optional): whether or not to apply optimizations.  Default: ``True``.
        check_trace (bool, optional): check if the same inputs run through
                                      traced code produce the same outputs. Default: ``True``. You might want
                                      to disable this if, for example, your network contains non-
                                      deterministic ops or if you are sure that the network is correct despite
                                      a checker failure.

        check_inputs (list of tuples, optional): A list of tuples of input arguments that should be used
                                                 to check the trace against what is expected. Each tuple
                                                 is equivalent to a seet of input arguments that would
                                                 be specified in ``args``. For best results, pass in a
                                                 set of checking inputs representative of the space of
                                                 shapes and types of inputs you expect the network to see.
                                                 If not specified, the original ``args`` is used for checking
        check_tolerance (float, optional): Floating-point comparison tolerance to use in the checker procedure.
                                           This can be used to relax the checker strictness in the event that
                                           results diverge numerically for a known reason, such as operator fusion.

    Returns:
        A ``ScriptModule`` object with a single ``forward()`` method containing the traced code.
        When func is a ``torch.nn.Module``, the returned ``ScriptModule`` will have the same set of
        sub-modules and parameters as func.

    Example:
       >>> def f(x):
       ...     return x * 2
       >>> traced_f = torch.jit.trace(f, torch.rand(1))

    """
    if not _enabled:
        return func
    executor_options = {'optimize': bool(optimize)}
    # Special case for common case of passing a single Tensor
    if isinstance(example_inputs, torch.Tensor):
        example_inputs = (example_inputs,)
    # done primarily so that weird iterables fail here and not pybind11 code
    elif not isinstance(example_inputs, tuple):
        example_inputs = tuple(example_inputs)
    if _module_class:
        module = _module_class(func, **executor_options)
    else:
        module = TopLevelTracedModule(func, **executor_options)
    var_lookup_fn = _create_interpreter_name_lookup_fn(0)
    module._create_method_from_trace('forward', func, example_inputs,
                                     var_lookup_fn, _force_outplace)

    # Check the trace against new traces created from user-specified inputs
    if check_trace:
        if check_inputs is not None:
            _check_trace(check_inputs, func, executor_options, module, check_tolerance, _force_outplace)
        else:
            _check_trace([example_inputs], func, executor_options, module, check_tolerance, _force_outplace)

    return module


def batch(batch_size=1, optimize=True, _frames_up=0):
    def decorator(fn):
        if not _enabled:
            return fn
        import torch.jit.batchop
        mod = script(fn, optimize, _frames_up)
        res_graph = torch.to_batch_graph(mod.graph)
        res_mod = ScriptModule()
        res_mod._create_method_from_graph('forward', res_graph)

        def wrapper(*args):
            new_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    arg = BatchTensor(arg, batch_size)
                if isinstance(arg, BatchTensor):
                    new_args.extend([arg.get_data(), arg.get_mask(), arg.get_dims()])
                else:
                    new_args.append(arg)
            res = res_mod(*new_args)
            assert len(res) % 3 == 0
            if len(res) % 3 != 0:
                raise "non-batched-tensor output is not supported yet"
            result = [BatchTensor(*res[i * 3: i * 3 + 3]) for i in range(len(res) // 3)]
            if len(result) == 1:
                return result[0]
            return result
        wrapper.__doc__ = fn.__doc__
        return wrapper
    return decorator


def _get_methods(cls):
    import inspect
    # In Python 3 unbound methods are functions, but in Python 2 they are methods
    return inspect.getmembers(cls, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x))


_compiled_methods_whitelist = {
    'forward', 'register_buffer', 'register_parameter', 'add_module',
    '_apply', 'apply', 'cuda', 'cpu', 'to', 'type', 'float', 'double', 'half',
    'state_dict', 'load_state_dict', '_load_from_state_dict',
    '_named_members', 'parameters', 'named_parameters',
    'buffers', 'named_buffers', 'children', 'named_children', 'modules',
    'named_modules', 'zero_grad', 'share_memory', '_get_name', 'extra_repr',
    '_slow_forward', '_tracing_name', 'eval', 'train',
}


def _make_fail(name):
    def fail(self, *args, **kwargs):
        raise RuntimeError(name + " is not supported on ScriptModules")
    return fail


for name, method in _get_methods(torch.nn.Module):
    if name.startswith('__'):
        continue
    if name not in ScriptModule.__dict__ and name not in _compiled_methods_whitelist:
        setattr(ScriptModule, method.__name__, _make_fail(name))


class TracedModule(ScriptModule):
    __frozen = False

    def __init__(self, orig, id_set=None, optimize=True):
        # XXX: orig can be a nn.Module or a function!
        super(TracedModule, self).__init__(optimize=optimize)
        if id_set is None:
            id_set = set()

        if not isinstance(orig, torch.nn.Module):
            self._name = orig.__name__
            orig = torch.nn.Module()
        else:
            self._name = 'TracedModule[' + type(orig).__name__ + ']'

        def check_unique(param):
            if param in id_set:
                raise ValueError("TracedModules don't support parameter sharing between modules")
            id_set.add(param)

        self.training = orig.training

        for name, param in orig._parameters.items():
            if param is not None:
                self._parameters[name] = param
                check_unique(param)
        for name, buf in orig._buffers.items():
            if buf is not None:
                self._buffers[name] = buf
                check_unique(buf)

        if orig._backward_hooks or orig._forward_hooks or orig._forward_pre_hooks:
            raise ValueError("Modules that have hooks assigned can't be compiled")

        for name, submodule in orig._modules.items():
            if isinstance(submodule, ScriptModule) and not isinstance(submodule, TracedModule):
                self._modules[name] = submodule.copy()
            else:
                self._modules[name] = TracedModule(submodule, id_set, optimize=optimize)

        self._freeze()

    def forward(self, *args, **kwargs):
        raise RuntimeError('Trace submodules cannot be called.')

    def _freeze(self):
        self.__frozen = True

    def _get_name(self):
        return self._name

    def __setattr__(self, attr, value):
        if not self.__frozen or hasattr(self, attr):
            return super(TracedModule, self).__setattr__(attr, value)
        raise RuntimeError("Cannot set new properties on a traced module.")


class TopLevelTracedModule(TracedModule):
    def forward(self, *args, **kwargs):
        return self._get_method('forward')(*args, **kwargs)


_builtin_table = None

_modules_containing_builtins = (torch, torch._C._nn)


def _unwrap_optional(x):
    assert x is not None, "Unwrapping null optional"
    return x


# lazily built to ensure the correct initialization order
def _get_builtin_table():
    global _builtin_table
    if _builtin_table is not None:
        return _builtin_table
    _builtin_table = {}

    def register_all(mod):
        for name in dir(mod):
            v = getattr(mod, name)
            if callable(v):
                _builtin_table[id(v)] = "aten::" + name
    for mod in _modules_containing_builtins:
        register_all(mod)

    _builtin_table[id(warnings.warn)] = "aten::warn"
    _builtin_table[id(_single)] = "aten::_single"
    _builtin_table[id(_pair)] = "aten::_pair"
    _builtin_table[id(_triple)] = "aten::_triple"
    _builtin_table[id(_quadruple)] = "aten::_quadruple"
    _builtin_table[id(_list_with_default)] = "aten::list_with_default"
    _builtin_table[id(_unwrap_optional)] = "aten::_unwrap_optional"
    _builtin_table[id(cudnn.is_acceptable)] = "aten::cudnn_is_acceptable"
    _builtin_table[id(torch._C._infer_size)] = "aten::_infer_size"
    _builtin_table[id(torch.nn.functional._no_grad_embedding_renorm_)] = "aten::_no_grad_embedding_renorm_"

    _builtin_table[id(math.floor)] = "aten::floor"
    _builtin_table[id(torch.nn.functional.interpolate)] = "aten::__interpolate"
    _builtin_table[id(torch.nn.functional.upsample_nearest)] = "aten::__upsample_nearest"
    _builtin_table[id(torch.nn.functional.upsample)] = "aten::__upsample"
    _builtin_table[id(torch.nn.functional.upsample_bilinear)] = "aten::__upsample_bilinear"
    _builtin_table[id(torch.nn.functional.assert_int_or_pair)] = "aten::_assert_int_or_pair"
    _builtin_table[id(torch.nn.utils.rnn.get_packed_sequence)] = "aten::_pack_sequence"

    return _builtin_table


def _register_builtin(fn, op):
    _get_builtin_table()[id(fn)] = op


def _find_builtin(fn):
    return _get_builtin_table().get(id(fn))


_register_builtin(len, 'aten::len')
_register_builtin(_wait, 'aten::wait')

# torch.jit.Error
Error = torch._C.JITException


class _disable_tracing(object):
    def __enter__(self):
        self.state = torch._C._get_tracing_state()
        torch._C._set_tracing_state(None)

    def __exit__(self, *args):
        torch._C._set_tracing_state(self.state)
        self.state = None


# for use in python if using annotate
def annotate(the_type, the_value):
    # noop in python
    return the_value


if not torch._C._jit_init():
    raise RuntimeError("JIT initialization failed")
