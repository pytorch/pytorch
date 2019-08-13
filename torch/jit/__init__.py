import torch._C
from torch.autograd import Variable, function
from torch.serialization import validate_cuda_device
from torch.nn import Module, ModuleList, Parameter, Sequential
from torch.jit.frontend import get_jit_class_def, get_jit_def, get_default_args
import torch.backends.cudnn as cudnn
import torch.jit.annotations
import torch._jit_internal as _jit_internal
from torch._jit_internal import _qualified_name
from torch._six import PY2, PY37, with_metaclass, get_function_from_type, \
    string_classes
from ..nn.modules.utils import _single, _pair, _triple, _quadruple, \
    _list_with_default
import torch.testing

import math
from collections import OrderedDict, namedtuple
import textwrap
import sys
import warnings
import weakref
import types
import contextlib
import os
import functools
import copy
import collections
import inspect
import pickle

# These are imported so users can access them from the `torch.jit` module
from torch._jit_internal import Final, _overload  # noqa: F401
from torch._jit_internal import ignore, export  # noqa: F401

if sys.version_info[0] > 2:
    import pathlib

def _parse_env(name, default, true_message, false_message):
    value = os.environ.get(name)
    if value is None:
        return default
    if value.lower() in {'1', 'true', 'yes'}:
        return True
    elif value.lower() in {'0', 'false', 'no'}:
        return False
    if value == '1v':
        print(true_message)
        return True
    elif value == '0v':
        print(false_message)
        return False
    raise ValueError('Unknown setting of {}. Try using 0 or 1.'.format(name))


_enabled = _parse_env('PYTORCH_JIT', True, "> Using PyTorch JIT", "> PyTorch JIT DISABLED")
_flatten = torch._C._jit_flatten
_unflatten = torch._C._jit_unflatten
_jit_script_class_compile = torch._C._jit_script_class_compile

# The Python CompilationUnit. All functions and modules defined in Python will
# live in here. It's defined in Python because doing in cpp creates static
# destruction order issues.
_python_cu = torch._C.CompilationUnit()

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

@contextlib.contextmanager
def optimized_execution(should_optimize):
    """
    A context manager that controls whether the JIT's executor will run
    optimizations before executing a function.
    """
    stored_flag = torch._C._get_graph_executor_optimize()
    torch._C._set_graph_executor_optimize(should_optimize)
    try:
        yield
    finally:
        torch._C._set_graph_executor_optimize(stored_flag)


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

        Example: ::

            torch.jit.load('scriptmodule.pt')

            # Load ScriptModule from io.BytesIO object
            with open('scriptmodule.pt', 'rb') as f:
                buffer = io.BytesIO(f.read())

            # Load all tensors to the original device
            torch.jit.load(buffer)

            # Load all tensors onto CPU, using a device
            torch.jit.load(buffer, map_location=torch.device('cpu'))

            # Load all tensors onto CPU, using a string
            torch.jit.load(buffer, map_location='cpu')

            # Load with extra files.
            files = {'metadata.json' : ''}
            torch.jit.load('scriptmodule.pt', _extra_files = files)
            print (files['metadata.json'])
    """
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

    cu = torch._C.CompilationUnit()
    if isinstance(f, str) or \
            (sys.version_info[0] == 2 and isinstance(f, unicode)) or \
            (sys.version_info[0] == 3 and isinstance(f, pathlib.Path)):
        cpp_module = torch._C.import_ir_module(cu, f, map_location, _extra_files)
    else:
        cpp_module = torch._C.import_ir_module_from_buffer(cu, f.read(), map_location, _extra_files)

    return ScriptModule(_cpp_module=cpp_module)


def save(m, f, _extra_files=DEFAULT_EXTRA_FILES_MAP):
    """
        Save an offline version of this module for use in a separate process. The saved
        module serializes all of the methods, submodules, parameters, and attributes of this
        module. It can be loaded into the C++ API using ``torch::jit::load(filename)`` or into the Python
        API with :func:`load <torch.jit.load>`.

        To be able to save a module, it must not make any calls to native Python functions.
        This means that all submodules must be subclasses of ``torch.jit.ScriptModule`` as well.

        .. DANGER::
           All modules, no matter their device, are always loaded onto the CPU during loading.
           This is different from :func:`load <torch.jit.load>`'s semantics and may change in the future.

        Arguments:
            m: a ScriptModule to save
            f: a file-like object (has to implement write and flush) or a string
               containing a file name
            _extra_files: Map from filename to contents which will be stored as part of 'f'

        .. warning::
            If you are using Python 2, ``torch.save`` does NOT support ``StringIO.StringIO``
            as a valid file-like object. This is because the write method should return
            the number of bytes written; ``StringIO.write()`` does not do this.

            Please use something like ``io.BytesIO`` instead.

        Example: ::

            import torch
            import io


            class MyModule(torch.nn.Module):
                def forward(self, x):
                    return x + 10

            m = torch.jit.script(MyModule())

            # Save to file
            torch.jit.save(m, 'scriptmodule.pt')

            # Save to io.BytesIO buffer
            buffer = io.BytesIO()
            torch.jit.save(m, buffer)

            # Save with extra files
            extra_files = torch._C.ExtraFilesMap()
            extra_files['foo.txt'] = 'bar'
            torch.jit.save(m, 'scriptmodule.pt', _extra_files=extra_files)
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
    # since Parameter.data always creates a new torch.Tensor instance,
    # id(v) doesn't work with it. So we always get the Parameter or Buffer
    # as values, and deduplicate the params using Parameters and Buffers
    state_dict = module.state_dict(keep_vars=True)
    filtered_dict = type(state_dict)()
    seen_ids = set()
    for k, v in state_dict.items():
        if id(v) in seen_ids:
            continue
        seen_ids.add(id(v))
        if keep_vars:
            filtered_dict[k] = v
        else:
            filtered_dict[k] = v.data
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
        try:
            trace, all_trace_inputs = torch._C._tracer_enter(*(in_vars + module_state))
        except Exception as e:
            torch._C._tracer_abandon()
            raise e
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
def _check_trace(check_inputs, func, traced_func, check_tolerance,
                 force_outplace, is_trace_module, _module_class):
    # Note: tracing is independent of optimizations, which consume the trace
    for inputs in check_inputs:

        if isinstance(inputs, torch.Tensor):
            inputs = (inputs,)

        if is_trace_module:
            copied_dict = {}
            for name, data in inputs.items():
                copied_dict[name] = _clone_inputs(data)
            check_mod = torch.jit.trace_module(
                func.__self__ if hasattr(func, '__self__') else func,
                copied_dict,
                check_trace=False,
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
                _force_outplace=force_outplace,
                _module_class=_module_class,
            )
            check_mod_func = check_mod

        def graph_diagnostic_info():
            mod_canonicalized = torch._C._jit_pass_canonicalize(traced_func.graph)
            torch._C._jit_pass_erase_shape_information(mod_canonicalized)
            check_canonicalized = torch._C._jit_pass_canonicalize(check_mod_func.graph)
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
                        mod_stack = n_mod.sourceRange()
                        if mod_stack:
                            source_printout += 'Trace source location:\n' + indent(mod_stack) + '\n'
                        check_stack = n_check.sourceRange()
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
                        compare_stack = n_mod.sourceRange()
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
            nondeterm_ops = [op for op in traced_func.graph.nodes() if op.isNondeterministic()]
            if len(nondeterm_ops) > 0:
                nondeterministic_ops_warning = "Trace had nondeterministic nodes. "
                nondeterministic_ops_warning += "Did you forget call .eval() on your model? Nodes:\n"
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

        traced_outs = run_mod_and_filter_tensor_outputs(traced_func, inputs, 'trace')
        fn_outs = run_mod_and_filter_tensor_outputs(func, inputs, 'Python function')
        if compare_outputs(traced_outs, fn_outs, 'Python function'):
            check_outs = run_mod_and_filter_tensor_outputs(check_mod_func, inputs, 'repeated trace')
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


def make_tuple(example_inputs):
    if isinstance(example_inputs, (torch.Tensor, dict)):
        return (example_inputs,)
    # done primarily so that weird iterables fail here and not pybind11 code
    if not isinstance(example_inputs, tuple):
        return tuple(example_inputs)
    return example_inputs


def make_module(mod, _module_class, _compilation_unit):
    if _module_class is None:
        _module_class = TopLevelTracedModule
    return _module_class(mod, _compilation_unit=_compilation_unit)

def wrap_check_inputs(check_inputs):
    if check_inputs is None:
        return None

    return [{'forward' : c} for c in check_inputs]

def trace(func,
          example_inputs,
          optimize=None,
          check_trace=True,
          check_inputs=None,
          check_tolerance=1e-5,
          _force_outplace=False,
          _module_class=None,
          _compilation_unit=_python_cu):
    """
    Trace a function and return an executable ``ScriptModule`` or ``torch.jit._C.Function``
    that will be optimized using just-in-time compilation.

    .. warning::

        Tracing only correctly records functions and modules which are not data
        dependent (e.g., do not have conditionals on data in tensors) and do not have
        any untracked external dependencies (e.g., perform input/output or
        access global variables). If you trace such models, you may silently get
        incorrect results on subsequent invocations of the model. The tracer
        will try to emit warnings when doing something that may cause an
        incorrect trace to be produced.

    Arguments:
        func (callable or torch.nn.Module):  a Python function or ``torch.nn.Module``
                                             that will be run with ``example_inputs``.
                                             arguments and returns to ``func`` must be tensors
                                             or (possibly nested) tuples that
                                             contain tensors.
        example_inputs (tuple):  a tuple of example inputs that will be passed to the function
                                 while tracing. The resulting trace can be run with
                                 inputs of different types and shapes assuming the traced operations
                                 support those types and shapes. ``example_inputs`` may also be a single
                                 Tensor in which case it is automatically wrapped in a tuple

    Keyword arguments:
        check_trace (bool, optional): check if the same inputs run through
                                      traced code produce the same outputs. Default: ``True``. You might want
                                      to disable this if, for example, your network contains non-
                                      deterministic ops or if you are sure that the network is correct despite
                                      a checker failure.

        check_inputs (list of tuples, optional): A list of tuples of input arguments that should be used
                                                 to check the trace against what is expected. Each tuple
                                                 is equivalent to a set of input arguments that would
                                                 be specified in ``example_inputs``. For best results, pass in a
                                                 set of checking inputs representative of the space of
                                                 shapes and types of inputs you expect the network to see.
                                                 If not specified, the original ``example_inputs`` are used for checking
        check_tolerance (float, optional): Floating-point comparison tolerance to use in the checker procedure.
                                           This can be used to relax the checker strictness in the event that
                                           results diverge numerically for a known reason, such as operator fusion.

    Returns:
        if ``callable`` is ``nn.Module`` or ``forward()`` of ``nn.Module``, ``trace`` returns
        a ``ScriptModule`` object with a single ``forward()`` method containing the traced code.
        The returned ``ScriptModule`` will have the same set of sub-modules and parameters as the
        original ``nn.Module``.
        If ``callable`` is a standalone function, ``trace`` returns ``torch.jit._C.Function``

    Example::

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(self, x):
                return self.conv(x)

            def weighted_kernel_sum(self, weight):
                return weight * self.conv.weight

        example_weight = torch.rand(1, 1, 3, 3)
        example_forward_input = torch.rand(1, 1, 3, 3)
        n = Net()
        # the following two calls are equivalent
        module = torch.jit.trace_module(n, example_forward_input)
        module = torch.jit.trace_module(n.forward, example_forward_input)

    """
    if not _enabled:
        return func
    if optimize is not None:
        warnings.warn("`optimize` is deprecated and has no effect. Use `with torch.jit.optimized_execution() instead")

    if isinstance(func, torch.jit.ScriptModule):
        # it is hard to trace it because the forward method on ScriptModule is already defined, so it
        # would result in an error.
        warnings.warn('The input to trace is already a ScriptModule, tracing it is a no-op. Returning the object as is.')
        return func

    if isinstance(func, torch.nn.Module):
        return trace_module(func, {'forward': example_inputs}, None,
                            check_trace, wrap_check_inputs(check_inputs),
                            check_tolerance, _force_outplace, _module_class)

    if (hasattr(func, '__self__') and isinstance(func.__self__, torch.nn.Module) and
            func.__name__ == 'forward'):

        return trace_module(func.__self__, {'forward': example_inputs}, None,
                            check_trace, wrap_check_inputs(check_inputs),
                            check_tolerance, _force_outplace, _module_class)

    # Special case for common case of passing a single Tensor
    if isinstance(example_inputs, (torch.Tensor, dict)):
        example_inputs = (example_inputs,)
    # done primarily so that weird iterables fail here and not pybind11 code
    elif not isinstance(example_inputs, tuple):
        example_inputs = tuple(example_inputs)

    var_lookup_fn = _create_interpreter_name_lookup_fn(0)

    if (hasattr(func, '__self__') and isinstance(func.__self__, torch.nn.Module)):
        raise AttributeError("trace doesn't support compiling individual module's functions.\n"
                             "Please use trace_module")

    name = _qualified_name(func)
    if name == '<lambda>':
        name = '_lambda'  # make name a valid identifier
    traced = torch._C._create_function_from_trace(name, func, example_inputs,
                                                  var_lookup_fn,
                                                  _force_outplace)

    # Check the trace against new traces created from user-specified inputs
    if check_trace:
        if check_inputs is not None:
            _check_trace(check_inputs, func, traced, check_tolerance, _force_outplace, False, _module_class)
        else:
            _check_trace([example_inputs], func, traced, check_tolerance, _force_outplace, False, _module_class)

    return traced


def trace_module(mod,
                 inputs,
                 optimize=None,
                 check_trace=True,
                 check_inputs=None,
                 check_tolerance=1e-5,
                 _force_outplace=False,
                 _module_class=None,
                 _compilation_unit=_python_cu):
    """
    Trace a module and return an executable ``ScriptModule`` that will be optimized
    using just-in-time compilation.

    .. warning::

        Tracing only correctly records functions and modules which are not data
        dependent (e.g., do not have conditionals on data in tensors) and do not have
        any untracked external dependencies (e.g., perform input/output or
        access global variables). If you trace such models, you may silently get
        incorrect results on subsequent invocations of the model. The tracer
        will try to emit warnings when doing something that may cause an
        incorrect trace to be produced.

    Arguments:
        mod (torch.nn.Module):           a ``torch.nn.Module`` containing methods whose names are
                                         specified in ``example_inputs``. The given methods will be compiled
                                         as a part of a single `ScriptModule`
        example_inputs (dict):           a dict containing sample inputs indexed by method names in ``mod``
                                         The inputs will be passed to methods whose names correspond to inputs'
                                         keys while tracing.
                                         ``{ 'forward' : example_forward_input, 'method2': example_method2_input}``
    Keyword arguments:
        check_trace (bool, optional): check if the same inputs run through
                                      traced code produce the same outputs. Default: ``True``. You might want
                                      to disable this if, for example, your network contains non-
                                      deterministic ops or if you are sure that the network is correct despite
                                      a checker failure.

        check_inputs (list of dicts, optional): A list of dicts of input arguments that should be used
                                                 to check the trace against what is expected. Each tuple
                                                 is equivalent to a set of input arguments that would
                                                 be specified in ``example_inputs``. For best results, pass in a
                                                 set of checking inputs representative of the space of
                                                 shapes and types of inputs you expect the network to see.
                                                 If not specified, the original ``example_inputs`` are used for checking
        check_tolerance (float, optional): Floating-point comparison tolerance to use in the checker procedure.
                                           This can be used to relax the checker strictness in the event that
                                           results diverge numerically for a known reason, such as operator fusion.

    Returns:
        A ``ScriptModule`` object with a single ``forward()`` method containing the traced code.
        When ``func`` is a ``torch.nn.Module``, the returned ``ScriptModule`` will have the same set of
        sub-modules and parameters as ``func``.

    Example::

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(self, x):
                return self.conv(x)

            def weighted_kernel_sum(self, weight):
                return weight * self.conv.weight

        example_weight = torch.rand(1, 1, 3, 3)
        example_forward_input = torch.rand(1, 1, 3, 3)
        inputs = {'forward' : example_forward_input, 'weighted_kernel_sum' : example_weight}
        n = Net()
        module = torch.jit.trace_module(n, inputs)

    """
    if not _enabled:
        return mod
    if optimize is not None:
        warnings.warn("`optimize` is deprecated and has no effect. Use `with torch.jit.optimized_execution() instead")

    var_lookup_fn = _create_interpreter_name_lookup_fn(0)

    if not isinstance(mod, torch.nn.Module):
        raise AttributeError("expected torch.nn.Module as the first argument")

    if not isinstance(inputs, dict):
        raise AttributeError("expected a dictionary of (method_name, input) pairs")

    module = make_module(mod, _module_class, _compilation_unit)

    for method_name, example_inputs in inputs.items():
        # this is needed since Module.__call__ sets up some extra tracing
        func = mod if method_name == "forward" else getattr(mod, method_name)
        example_inputs = make_tuple(example_inputs)
        module._c._create_method_from_trace(method_name, func, example_inputs, var_lookup_fn, _force_outplace)
        check_trace_method = module._c._get_method(method_name)

        # Check the trace against new traces created from user-specified inputs
        if check_trace:
            if check_inputs is not None:
                _check_trace(check_inputs, func, check_trace_method,
                             check_tolerance, _force_outplace, True, _module_class)
            else:
                _check_trace([inputs], func, check_trace_method,
                             check_tolerance, _force_outplace, True, _module_class)

    return module


class CompilationUnit(object):
    def __init__(self, lang=None, _frames_up=0):
        self._c = torch._C.CompilationUnit()
        if lang is not None:
            self.define(lang, _frames_up=_frames_up + 1)

    def define(self, lang, rcb=None, _frames_up=0):
        if not rcb:
            rcb = _jit_internal.createResolutionCallback(_frames_up + 1)
        self._c.define(lang, rcb)

    def __getattr__(self, attr):
        r = self._c.find_function(attr)
        if r is None:
            raise AttributeError("'CompilationUnit' has no attribute '{}'".format(attr))
        return r


def _try_get_dispatched_fn(fn):
    if not callable(fn):
        return None
    return _jit_internal.boolean_dispatched.get(fn)


def _try_get_overloaded_fn(mod, field):
    return mod._overloads.get(field, None) if isinstance(mod, ScriptModule) else None


class ScriptWarning(Warning):
    pass


def _create_constant_iterable_module(module):
    modules = OrderedDict()

    for key, submodule in module._modules.items():
        if isinstance(submodule, (ModuleList, Sequential)):
            # Make each item in the module a constant
            modules[key] = _create_constant_iterable_module(submodule)
        else:
            modules[key] = _convert_to_script_module(submodule)

    if isinstance(module, Sequential):
        return _ConstSequential(Sequential(modules))
    elif isinstance(module, ModuleList):
        return _ConstModuleList(modules)
    else:
        raise RuntimeError("Only nn.ModuleList and nn.Sequential can be made "
                           "into constant modules, found {}".format(module))


def _make_strong_submodule(field, module, parent):
    if field not in parent._modules:
        # It's not a submodule, don't do anything
        return None

    # Convert the module to a ScriptModule
    new_strong_submodule = _convert_to_script_module(module)

    # Install the ScriptModule on the python side
    parent._modules._python_modules[field] = new_strong_submodule

    return new_strong_submodule


def _try_compile_fn(fn, loc):
    if _jit_internal.is_ignored_fn(fn):
        # Don't do anything for @ignore'd functions
        return None

    if isinstance(fn, torch.nn.Module):
        # Since modules are callable pybind recognizes them as functions, but
        # don't do anything for them
        return None

    if not inspect.isfunction(fn) and not inspect.ismethod(fn):
        raise RuntimeError("`{}` is not a function. Recursive scripting only supports "
                           "Python functions or methods currently.\n"
                           "Consider manually annotating `{}` with @torch.jit.script.".format(fn, fn))

    # We don't have the actual scope where the function was defined, but we can
    # extract the necessary info from the closed over variables on the function
    # object
    rcb = _jit_internal.createResolutionCallbackFromClosure(fn)
    return torch.jit.script(fn, _rcb=rcb)


@contextlib.contextmanager
def _disable_emit_hooks():
    hooks = torch._C._jit_get_emit_hooks()
    torch._C._jit_set_emit_hooks(None, None)
    yield
    torch._C._jit_set_emit_hooks(hooks[0], hooks[1])


def _create_method_from_fn(module, fn):
    if _jit_internal.is_ignored_fn(fn):
        return None
    if not inspect.ismethod(fn):
        return None
    stub = script_method(fn, _jit_internal.createResolutionCallbackFromClosure(fn))
    with _disable_emit_hooks():
        # We don't want to call the hooks here since the graph that is calling
        # this function is not yet complete
        _create_methods_from_stubs(module, (stub,))
    return stub


# ScriptClasses must be new-style classes because we construct them using their
# __new__ method.
def _is_new_style_class(cls):
    if hasattr(cls, '__class__'):
        return ('__dict__' in dir(cls) or hasattr(cls, '__slots__'))


def whichmodule(obj):
    """Find the module an object belong to."""
    module_name = getattr(obj, '__module__', None)
    # Protect the iteration by using a list copy of sys.modules against dynamic
    # modules that trigger imports of other modules upon calls to getattr.
    for name, module in list(sys.modules.items()):
        if name == '__main__' or module is None:
            continue
        try:
            if _getattribute(module, name)[0] is obj:
                return module_name
        except AttributeError:
            pass
    return '__main__'


def _compile_and_register_class(obj, rcb, qualified_name):
    ast = get_jit_class_def(obj, obj.__name__)
    _jit_script_class_compile(qualified_name, ast, rcb)
    _add_script_class(obj, qualified_name)

def script(obj, optimize=None, _frames_up=0, _rcb=None):
    r"""
    Scripting a function or ``nn.Module`` will inspect the source code, compile
    it as TorchScript code using the TorchScript compiler, and return a ``ScriptModule`` or
    ``torch._C.Function``.

    **Scripting a function**
        The ``@torch.jit.script`` decorator will construct a ``torch._C.Function``.

        Example (scripting a function)::

            import torch
            @torch.jit.script
            def foo(x, y):
                if x.max() > y.max():
                    r = x
                else:
                    r = y
                return r

    **Scripting an nn.Module**
        Scripting an ``nn.Module`` by default will compile the ``forward`` method and recursively
        compile any methods, submodules, and functions called by ``forward``. If a ``nn.Module`` only uses
        features supported in TorchScript, no changes to the original module code should be necessary. ``script``
        will construct ``torch.jit.ScriptModule`` that has copies of the attributes, parameters, and methods of
        the original module.

        Example (scripting a simple module with a Parameter)::

            import torch

            class MyModule(torch.nn.Module):
                def __init__(self, N, M):
                    super(MyModule, self).__init__()
                    # This parameter will be copied to the new ScriptModule
                    self.weight = torch.nn.Parameter(torch.rand(N, M))

                    # When this submodule is used, it will be compiled
                    self.linear = torch.nn.Linear(N, M)

                def forward(self, input):
                    output = self.weight.mv(input)

                    # This calls the `forward` method of the `nn.Linear` module, which will
                    # cause the `self.linear` submodule to be compiled to a `ScriptModule` here
                    output = self.linear(output)
                    return output

            scripted_module = torch.jit.script(MyModule())

        Example (scripting a module with traced submodules)::

            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class MyModule(nn.Module):
                def __init__(self):
                    super(MyModule, self).__init__()
                    # torch.jit.trace produces a ScriptModule's conv1 and conv2
                    self.conv1 = torch.jit.trace(nn.Conv2d(1, 20, 5), torch.rand(1, 1, 16, 16))
                    self.conv2 = torch.jit.trace(nn.Conv2d(20, 20, 5), torch.rand(1, 20, 16, 16))

                def forward(self, input):
                  input = F.relu(self.conv1(input))
                  input = F.relu(self.conv2(input))
                  return input

            scripted_module = torch.jit.script(MyModule())

        To compile a method other than ``forward`` (and recursively compile anything it calls), add
        the ``@torch.jit.export`` decorator to the method. To opt out of compilation use ``@torch.jit.ignore``.

        Example (an exported and ignored method in a module)::

            import torch
            import torch.nn as nn

            class MyModule(nn.Module):
                def __init__(self):
                    super(MyModule, self).__init__()

                @torch.jit.export
                def some_entry_point(self, input):
                    return input + 10

                @torch.jit.ignore
                def python_only_fn(self, input):
                    # This function won't be compiled, so any Python APIs can be
                    # used
                    import pdb
                    pdb.set_trace()

                def forward(self, input):
                    if self.training:
                        self.python_only_fn(input)
                    return input * 99

            scripted_module = torch.jit.script(MyModule())
            print(scripted_module.some_entry_point(torch.randn(2, 2)))
            print(scripted_module(torch.randn(2, 2)))
    """
    if not _enabled:
        return obj

    if optimize is not None:
        warnings.warn("`optimize` is deprecated and has no effect. Use `with torch.jit.optimized_execution() instead")

    if isinstance(obj, torch.nn.Module):
        return _convert_to_script_module(obj)

    qualified_name = _qualified_name(obj)
    if inspect.isclass(obj):
        # If this type is a `nn.Module` subclass, they probably meant to pass
        # an instance instead of a Module
        if issubclass(obj, torch.nn.Module):
            raise RuntimeError("Type '{}' cannot be compiled since it inherits"
                               " from nn.Module,"
                               " pass an instance instead".format(obj))

        if not _is_new_style_class(obj):
            raise RuntimeError("TorchScript classes must be new-style classes. "
                               "Please inherit from 'object'")
        if _rcb is None:
            _rcb = _jit_internal.createResolutionCallback(_frames_up + 1)
        _compile_and_register_class(obj, _rcb, qualified_name)
        return obj
    else:
        _check_directly_compile_overloaded(obj)
        ast = get_jit_def(obj)
        if _rcb is None:
            _rcb = _gen_rcb(obj, _frames_up)
        fn = torch._C._jit_script_compile(qualified_name, ast, _rcb, get_default_args(obj))
        # Forward docstrings
        fn.__doc__ = obj.__doc__
        return fn

def _gen_rcb(obj, _frames_up):
    _frames_up = _frames_up + 1  # for invoking _gen_rcb()

    closure_rcb = _jit_internal.createResolutionCallbackFromClosure(obj)
    stack_rcb = _jit_internal.createResolutionCallback(_frames_up + 1)

    def _rcb(name):
        # since type comments aren't captured in the function's closures,
        # we still need to try to the rcb based on stack frames if the
        # closure rcb fails
        result = closure_rcb(name)
        if result:
            return result
        return stack_rcb(name)

    return _rcb

ScriptMethodStub = namedtuple('ScriptMethodStub', ('resolution_callback', 'def_', 'original_method'))


def script_method(fn, _rcb=None):
    if not _enabled:
        return fn
    # NOTE: we need to traverse two frames here because the meta-class frame
    # for ScriptModule will be present, as opposed to invoking @script on a
    # a function or invoking define() on a CompilationUnit.
    # The stack will look like:
    #
    # 0. createResolutionCallback()
    # 1. script_method()
    # 2. ScriptModule metaclass frame
    # 3. Surrounding scope
    #
    # createResolutionCallback internally adds 1 to get us to the scope of this
    # function (the calling function). Adding 2 gets us to the proper surrounding scope.
    if _rcb is None:
        _rcb = _jit_internal.createResolutionCallback(frames_up=2)
    ast = get_jit_def(fn, self_name="ScriptModule")
    return ScriptMethodStub(_rcb, ast, fn)



# These OrderedDictWrapper classes replace the actual OrderedDicts in
# module with versions that get/set properties inside of script::Module.
# This allows us to reuse most of nn.Module while still storing the
# data in C++.
# Each OrderedDict needs to support:
#  x not in view
#  x in view
#  view[name] = ...
#  view.values()
#  del view[name]
#  view.items()
#  view.keys()
#  len(view)

class OrderedDictWrapper(object):
    def __init__(self, module):
        self.module = module

    def keys(self):
        return [k for k, v in self.items()]

    def values(self):
        return [v for k, v in self.items()]

    def __len__(self):
        return len(self.values())

    def __delitem__(self, k):
        raise RuntimeError("cannot delete methods or parameters of a script module")

    def items(self):
        raise NotImplementedError

    def __contains__(self, k):
        raise NotImplementedError

    def __getitem__(self, k):
        raise NotImplementedError

    def __setitem__(self, k, v):
        raise NotImplementedError


class OrderedModuleDict(OrderedDictWrapper):
    def __init__(self, module):
        super(OrderedModuleDict, self).__init__(module)
        # contains _both_ script modules and non-script python-only modules

        # because script modules are subclassed in python and the
        # C++ script::Module class will not hold references to them,
        # to ensure that you always get the same python value here
        # we store it in the python dict as well
        self._python_modules = OrderedDict()

    def items(self):
        r = self._python_modules.items()
        return r

    def __contains__(self, k):
        return k in self._python_modules

    def __setitem__(self, k, v):
        if k in self._python_modules:
            raise RuntimeError("Cannot re-assign modules in a ScriptModule, "
                               "tried to replace existing module '{}': {}".format(k, v))
        if isinstance(v, ScriptModule):
            self.module._register_module(k, v._c)

        self._python_modules[k] = v

    def __getitem__(self, k):
        return self._python_modules[k]


class OrderedParameterDict(OrderedDictWrapper):
    def __init__(self, module):
        super(OrderedParameterDict, self).__init__(module)

    def items(self):
        return [(name, param) for name, param in self.module._get_parameters()]

    def __setitem__(self, k, v):
        self.module._register_parameter(k, v, False)

    def __contains__(self, k):
        return self.module._has_parameter(k)

    def __getitem__(self, k):
        if k not in self:
            raise KeyError(k)
        return self.module._get_parameter(k)


class OrderedBufferDict(OrderedDictWrapper):
    def __init__(self, module):
        super(OrderedBufferDict, self).__init__(module)

    def items(self):
        return [(name, param) for name, _, param in
                self.module._get_attributes() if isinstance(param, torch.Tensor)]

    def __setitem__(self, k, v):
        self.module._register_buffer(k, v)

    def __contains__(self, k):
        return self.module._has_buffer(k)

    def __getitem__(self, k):
        if k not in self:
            raise KeyError(k)
        return self.module._get_buffer(k)

# base types that can be constants
# in addition, tuples and lists of these base types are also considered constants
# If you edit this list, then you also need to edit the handlers in
# ConstantValue in jit/script/init.cpp
_constant_types = (bool, float, int, str, type(None), types.FunctionType, torch.device, torch.layout, torch.dtype)


def _get_valid_constant(attr, v):
    if isinstance(v, _constant_types):
        return v
    elif isinstance(v, tuple) or isinstance(v, list):
        return tuple(_get_valid_constant(attr, x) for x in v)
    constants = ", ".join(typ.__name__ for typ in _constant_types)
    raise TypeError(textwrap.dedent("""
        '{}' object for attribute '{}' is not a valid constant.
        Valid constants are:
          1. a nn.ModuleList
          2. a value of type {{{}}}
          3. a list or tuple of (2)
        """.format(type(v).__name__, attr, constants)))


def _create_methods_from_stubs(self, stubs):
    defs = [m.def_ for m in stubs]
    rcbs = [m.resolution_callback for m in stubs]
    defaults = [get_default_args(m.original_method) for m in stubs]
    self._c._create_methods(self, defs, rcbs, defaults)

# For each user-defined class that subclasses ScriptModule this meta-class,
# (1) finds all the methods annotated with @script_method
# in a ScriptModule and removes them from the class attributes, and
# (2) puts a wrapper around the class's __init__ method to register
# all of the script_methods with the module after the original __init__
# has run. This has to occur after the user-defined __init__ so that
# submodules and parameters are initialized _before_ the script compiler
# resolve references to `self.param` or `self.module`.


class ScriptMeta(type):
    # this has to inherit from pybind11's metaclass otherwise we get
    # issues because ScriptModule inherits from torch._C.ScriptModule,
    # a pybind11 type
    def __init__(cls, name, bases, attrs):
        # initialize inherited properties
        cls._methods = {}
        cls._constants_set = set(getattr(cls, '__constants__', ()))
        for base in reversed(bases):
            for k, v in getattr(base, '_methods', {}).items():
                cls._methods[k] = v
            base_constants = getattr(base, '_constants_set', set())
            cls._constants_set = cls._constants_set.union(base_constants)

        # find all the script methods of the current class
        for k, v in sorted(attrs.items()):
            if isinstance(v, ScriptMethodStub):
                delattr(cls, k)
                cls._methods[v.original_method.__name__] = v

        original_init = getattr(cls, '__init__', lambda self: None)
        cls._overloads = dict(getattr(cls, '__overloads__', {}))

        # after the user's __init__ register all the script methods
        # with the module
        @functools.wraps(original_init)
        def init_then_register(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if type(self) == cls:
                # this is the init of the concrete type of self,
                # we have already resolved all _methods
                methods = [v for k, v in sorted(cls._methods.items())]
                _create_methods_from_stubs(self, methods)

        cls.__init__ = init_then_register
        return super(ScriptMeta, cls).__init__(name, bases, attrs)


if _enabled:

    # this is a Python 'non-data descriptor' that causes the first access
    # to ScriptModule's forward to lookup the forward method and stash
    # it in the objects dict. Due to the standard rules for attribute lookup
    # subsequent lookups will just directly return the previously looked up method.
    # This is necessary because nn.Module defines forward as a method. If we
    # did nothing __getattr__ would not be called. Instead we'd get nn.Module.forward
    # which always throws an exception.
    class _CachedForward(object):
        def __get__(self, obj, cls):
            return self.__getattr__('forward')

    class ScriptModule(with_metaclass(ScriptMeta, Module)):
        r"""
        The core data structure in TorchScript is the ``ScriptModule``. It is an
        analogue of torch's ``nn.Module`` and represents an entire model as a tree of
        submodules. Like normal modules, each individual module in a ``ScriptModule`` can
        have submodules, parameters, and methods. In ``nn.Module``\s methods are implemented
        as Python functions, but in ``ScriptModule``\s methods are implemented as
        TorchScript functions,  a statically-typed subset of Python that contains all
        of PyTorch's built-in Tensor operations. This difference allows your
        ScriptModules code to run without the need for a Python interpreter.

        ``ScriptModule``\s be created in two ways:

        **Tracing:**

            Using ``torch.jit.trace`` and ``torch.jit.trace_module``, you can turn an existing module or Python
            function into a TorchScript ``torch._C.Function`` or ``ScriptModule``. You must provide example inputs,
            and we run the function, recording the operations performed on all the tensors.
            * The resulting recording of a standalone function produces ``torch._C.Function``.
            * The resulting recording of ``forward`` function of ``nn.Module`` or ``nn.Module`` produces ``ScriptModule``.

            This module also contains any parameters that the original
            module had as well.

            Example (tracing a function)::

                import torch
                def foo(x, y):
                    return 2 * x + y
                traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))

            .. note::
                Tracing a standalone function will construct a ``torch._C.Function``
                Tracing ``nn.Module``s ``forward`` will construct a ``ScriptModule``

            Example (tracing an existing module)::

                import torch
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

                # all three trace calls below are equivalent
                # and construct `ScriptModule` with a single `forward` method
                module = torch.jit.trace(n.forward, example_forward_input) # produces ScriptModule with `forward`
                module = torch.jit.trace(n, example_forward_input) # produces ScriptModule with `forward`
                module = torch.jit.trace_module(n, inputs) # produces ScriptModule with `forward`

                inputs = {'forward' : example_forward_input, 'weighted_kernel_sum' : example_weight}
                # trace_module produces `ScriptModule` with two methods:
                # `forward` and `weighted_kernel_sum`
                module = torch.jit.trace_module(n, inputs, True, True)

            .. note::

                * The first three trace/trace_module calls are equivalent and return ``ScriptModule``
                with a single ``forward`` method.
                * The last ``trace_module`` call produces a ``ScriptModule`` with two methods.

                Tracing only records operations done when the given function is run on the given
                tensors. Therefore, the returned ``ScriptModule`` will always run the same traced
                graph on any input. This has some important implications when your module is
                expected to run different sets of operations, depending on the input and/or the
                module state. For example,

                    + Tracing will not record any control-flow like if-statements or loops. When
                      this control-flow is constant across your module, this is fine and it often
                      inlines the control-flow decisions. But sometimes the control-flow is
                      actually part of the model itself. For instance, a recurrent network is
                      a loop over the (possibly dynamic) length of an input sequence.

                    + In the returned ``ScriptModule``, operations that have different behaviors
                      in ``training`` and ``eval`` modes will always behave as if it is in the
                      mode it was in during tracing, no matter which mode the ``ScriptModule``
                      is in.

                In cases like these, tracing would not be appropriate and scripting is a better
                choice.

        **Scripting:**

            You can write TorchScript code directly using Python syntax. You do this
            using the ``@torch.jit.script`` decorator for functions and modules. You can
            also call ``torch.jit.script`` directly with the function or module you wish to
            compile. On functions, the body of the function is compiled to TorchScript. If
            applied to an ``nn.Module``, by default the ``forward`` method and any methods it
            calls are compiled, and all buffer and Parameters of the original module are copied
            to a new ``ScriptModule``. You should not need to construct a ``ScriptModule`` manually.
            TorchScript itself is a subset of the Python language, so not all
            features in Python work, but we provide enough functionality to compute on
            tensors and do control-dependent operations.
        """
        def __init__(self, optimize=None, _qualified_name=None, _compilation_unit=None, _cpp_module=None):
            if _qualified_name is None:
                _qualified_name = _jit_internal._qualified_name(self.__class__)
            if _compilation_unit is None:
                _compilation_unit = _python_cu
            if optimize is not None:
                warnings.warn("`optimize` is deprecated and has no effect. Use `with torch.jit.optimized_execution() instead")

            # If we were give a _cpp_module, use that one as the backing cpp
            # module instead of creating a fresh one.
            if _cpp_module is not None:
                self.__dict__['_c'] = _cpp_module
            else:
                self.__dict__['_c'] = torch._C.ScriptModule(_qualified_name, _compilation_unit, True)

            Module._Module__construct(self)
            Module.__setattr__(self, "training", True)

            self._parameters = OrderedParameterDict(self._c)
            self._buffers = OrderedBufferDict(self._c)
            self._modules = OrderedModuleDict(self._c)

            # If we were given a _cpp_module, recursively create Python
            # ScriptModules that mirror the submodule hierarchy.
            # This has to go last due to quirks in module initialization.
            if _cpp_module is not None:
                for (name, cpp_mod) in self._c._get_modules():
                    setattr(self, name, ScriptModule(_cpp_module=cpp_mod))

        @property
        def graph(self):
            return self.forward.graph

        @property
        def code(self):
            return self.forward.code

        def save(self, *args, **kwargs):
            return self._c.save(*args, **kwargs)

        def save_to_buffer(self, *args, **kwargs):
            return self._c.save_to_buffer(*args, **kwargs)

        def get_debug_state(self, *args, **kwargs):
            return self._c.get_debug_state()

        forward = _CachedForward()

        def __getattr__(self, attr):
            if '_c' not in self.__dict__:
                raise RuntimeError("ScriptModule has not been initialized, did you forget to call super's init?")
            if self._c._has_attribute(attr):
                return self._c._get_attribute(attr)
            if self._c._has_method(attr):
                if attr in self.__class__._methods:
                    original_method = self.__class__._methods[attr].original_method
                    script_method = self._c._get_method(attr)
                    script_method = functools.wraps(original_method)(script_method)
                else:
                    script_method = self._c._get_method(attr)
                # cache method so future calls do not go through __getattr__
                # to improve invocation performance
                self.__dict__[attr] = script_method
                return script_method
            return super(ScriptModule, self).__getattr__(attr)

        def __setattr__(self, attr, value):
            if attr not in self._constants_set:
                if attr == 'training':
                    if self._c._has_attribute('training'):
                        self.__dict__['training'] = value
                        self._c._set_attribute('training', value)
                        return
                if isinstance(value, Attribute):
                    the_type = torch.jit.annotations.ann_to_type(value.type)
                    try:
                        self._c._register_attribute(attr, the_type, value.value)
                    except RuntimeError:
                        raise RuntimeError("Could not register attribute '{}' of type '{}' for a value of type '{}'"
                                           .format(attr, value.type, type(value.value)))
                    return
                return super(ScriptModule, self).__setattr__(attr, value)

            if hasattr(self, attr):
                raise RuntimeError("attempting to re-assign constant '{}' in {}".format(attr, type(self).__name__))

            def conv_module_to_const(module_value):
                if not isinstance(module_value, (ModuleList, Sequential)):
                    return module_value
                for i in range(len(module_value)):
                    module_value[i] = conv_module_to_const(module_value[i])
                if isinstance(module_value, Sequential):
                    return _ConstSequential(module_value)
                else:
                    return _ConstModuleList(module_value)

            if isinstance(value, (ModuleList, Sequential)):
                # special case for list of modules. Modules need to be registered with their
                # parent module. To do this, we create a ConstModuleList, which is itself a module, that
                # contains each of these modules as submodules. The ConstModuleList then
                # is set as an attribute of the parent module.
                super(ScriptModule, self).__setattr__(attr, conv_module_to_const(value))
            else:
                super(ScriptModule, self).__setattr__(attr, _get_valid_constant(attr, value))

        def __dir__(self):
            return sorted(Module.__dir__(self) + self._c._method_names())

        def define(self, lang):
            # We use frames_up=1 to get to the proper surrounding scope. The stack
            # will look like:
            # 0. createResolutionCallback
            # 1. define()
            # 2. surrounding scope.
            #
            # createResolutionCallback internally adds 1 to get us to our frame, then
            # we add 1 to get to the proper surrounding scope.
            rcb = _jit_internal.createResolutionCallback(frames_up=1)
            self._c._define(self, lang, rcb)

        def copy(self):
            return ScriptModule(_cpp_module=self._c._clone())

        def __getstate__(self):
            raise pickle.PickleError(
                "ScriptModules cannot be deepcopied using copy.deepcopy or saved using torch.save. " +
                "Mixed serialization of script and non-script modules is not supported. " +
                "For purely script modules use my_script_module.save(<filename>) instead.")

        def graph_for(self, *args, **kwargs):
            return self.forward.graph_for(*args, **kwargs)

    class WeakScriptModuleProxy(ScriptModule):
        # TODO: [weak script refactor]
        # WeakScriptModule proxy should be deleted since its functionality is
        # subsumed by recursive scripting, and the copying code in init moved
        # to a function to create a ScriptModule from an nn.Module without
        # making a WeakScriptModuleProxy
        """
        Copies the parameters, buffers, constants, attributes, and submodules
        of an nn.Module into itself.
        """
        def __init__(self, original, stubs):
            # Guards behavior of __setattr__ and __getattr__ so ScriptModule
            # __init__ can run correctly
            self.__dict__['_initialized'] = False
            super(WeakScriptModuleProxy, self).__init__(_qualified_name=_qualified_name(type(original)))
            # Store a weak reference to the original module
            self.__dict__["_original"] = weakref.ref(original)

            constants_set = set(getattr(original, "__constants__", []))
            self.__dict__["_constants_set"] = {}

            if not hasattr(original, '_parameters'):
                raise RuntimeError("'{}' has not been initialized, did you forget to call 'super()'?"
                                   .format(type(original).__name__))

            # Copy Parameters and Modules
            for name in dir(original):
                item = getattr(original, name)
                if item is None and name in original._parameters:
                    # XXX: treat None value simply as module attributes instead of adding them to the parameter list
                    # TODO: need to handle this more generally when non-tensor attributes added to module
                    object.__setattr__(self, name, item)
                elif item is self:
                    continue
                elif isinstance(item, (Parameter, Module, Attribute)):
                    ScriptModule.__setattr__(self, name, item)

            # Copy buffers
            for name in original._buffers:
                if original._buffers[name] is None:
                    object.__setattr__(self, name, None)
                else:
                    self.register_buffer(name, original._buffers[name])

            # Constants annotated via `Final[T]` rather than being added to `__constants__`
            for name, ann in getattr(original, '__annotations__', {}).items():
                if torch._jit_internal.is_final(ann):
                    constants_set.add(name)

            # Copy constants
            self.__dict__["_constants_set"] = constants_set
            for name in self.__dict__["_constants_set"]:
                if hasattr(original, name):
                    if (name in original._parameters or name in original._buffers) and item is not None:
                        # for 'None' parameters/buffers, don't actually add their values if it exists
                        continue
                    ScriptModule.__setattr__(self, name, getattr(original, name))

            # Copy annotations, pull types from `__annotations__` or try to infer
            # the type if possible
            class_annotations = getattr(original, '__annotations__', {})
            for name in dir(original):
                if name in ("training", "__dict__"):
                    # TODO: removing this skip should let us remove the code to add training as an
                    # attribute in python_sugared_value.cpp
                    continue
                if hasattr(self, name):
                    # Don't re-copy properties
                    continue
                item = getattr(original, name)
                if name in class_annotations:
                    the_type = torch.jit.annotations.ann_to_type(class_annotations[name])
                else:
                    the_type = torch._C._jit_try_infer_type(item)
                if the_type is not None:
                    self._c._register_attribute(name, the_type, item)

            # Copy overloads
            self.__dict__["_overloads"] = dict(getattr(original, "__overloads__", {}))

            self.__dict__["_initialized"] = True
            self.__dict__["_original_type"] = type(original)
            _create_methods_from_stubs(self, stubs)

        def __getattr__(self, attr):
            # Try to get the attribute directly, if that fails, fall back to the
            # weak module itself
            try:
                return ScriptModule.__getattr__(self, attr)
            except AttributeError as e:
                # unwrap the original
                original_module = self.__dict__["_original"]()
                if original_module and self.__dict__["_initialized"]:
                    # get attr from original if it is still alive
                    return getattr(original_module, attr)
                elif self.__dict__["_initialized"]:
                    # original module is dead, try looking up the value on the
                    # original type
                    fn = getattr(self.__dict__["_original_type"], attr, None)
                    if fn is not None and inspect.isroutine(fn):
                        # bind the function to this instance and return it
                        return fn.__get__(self, self.__dict__["_original_type"])
                # If it's not on this module and it wasn't on the original
                # module (or the original is dead), throw the exception
                raise e

        def __setattr__(self, attr, value):
            # Once constructed, no new properties can be set

            if not self.__dict__["_initialized"]:
                # If constructing, don't fall back to original module
                return ScriptModule.__setattr__(self, attr, value)

            if hasattr(self, attr):
                return ScriptModule.__setattr__(self, attr, value)
            else:
                raise AttributeError("Cannot set new attribute '{}' on "
                                     "weak script module once it has been "
                                     "created".format(attr))

else:
    class ScriptModule(torch.nn.Module):
        def __init__(self):
            super(ScriptModule, self).__init__()


def _convert_to_script_module(mod):
    """
    Makes a ScriptModule from an nn.Module. If `_methods` is provided,
    these methods are treated as @script_methods. If not, it defaults to
    `('forward',)`. Methods accessed in forward are scripted on demand.
    """
    if isinstance(mod, ScriptModule):
        return mod

    if isinstance(mod, (ModuleList, Sequential)):
        # Create constant versions for the iterable modules
        return _create_constant_iterable_module(mod)

    methods = ()
    if hasattr(mod, 'forward'):
        if mod.forward.__func__ == torch.nn.Module.forward:
            raise RuntimeError("No forward method was defined on {}".format(mod))
        if not _jit_internal.is_ignored_fn(mod.forward):
            methods = ('forward',)
    exported = []
    for name in dir(mod):
        item = getattr(mod, name)
        if callable(item):
            if _jit_internal.get_torchscript_modifier(item) is _jit_internal.FunctionModifiers.EXPORT:
                exported.append(name)
    methods = methods + tuple(exported)

    def make_stub(method):
        func = get_function_from_type(type(mod), method)
        return script_method(func, _jit_internal.createResolutionCallbackFromClosure(func))

    stubs = list(map(make_stub, methods))
    return WeakScriptModuleProxy(mod, stubs)


def _get_methods(cls):
    import inspect
    # In Python 3 unbound methods are functions, but in Python 2 they are methods
    return inspect.getmembers(cls, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x))


_compiled_methods_whitelist = {
    'forward', 'register_buffer', 'register_parameter', 'add_module',
    '_apply', 'apply', 'cuda', 'cpu', 'to', 'type', 'float', 'double', 'half',
    'state_dict', '_save_to_state_dict', 'load_state_dict',
    '_load_from_state_dict', '_named_members', 'parameters', 'named_parameters',
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

    def __init__(self, orig, id_set=None, _compilation_unit=None):
        # XXX: orig can be a nn.Module or a function!
        super(TracedModule, self).__init__(_qualified_name=_jit_internal._qualified_name(orig.__class__),
                                           _compilation_unit=_compilation_unit)
        if id_set is None:
            id_set = set()

        assert(isinstance(orig, torch.nn.Module))
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

        if orig._backward_hooks:
            raise ValueError("Modules that have backward hooks assigned can't be compiled: " + str(self))

        for name, submodule in orig._modules.items():
            if isinstance(submodule, ScriptModule):
                self._modules[name] = submodule
            else:
                self._modules[name] = TracedModule(submodule, id_set)

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


if _enabled:
    class TopLevelTracedModule(TracedModule):
        forward = _CachedForward()


class _ConstModuleList(ScriptModule):
    def __init__(self, modules):
        super(_ConstModuleList, self).__init__()

        if isinstance(modules, OrderedDict):
            for key, module in modules.items():
                if isinstance(module, torch.nn.Module):
                    module = _convert_to_script_module(module)
                self.add_module(key, module)
        else:
            for i, module in enumerate(modules):
                if isinstance(module, torch.nn.Module):
                    module = _convert_to_script_module(module)
                self.add_module(str(i), module)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return _ConstModuleList(list(self._modules.values())[idx])
        else:
            if not (-len(self) <= idx < len(self)):
                raise IndexError('index {} is out of range'.format(idx))
            if idx < 0:
                idx += len(self)
            return self._modules[str(idx)]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __dir__(self):
        keys = super(_ConstModuleList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys


class _ConstSequential(_ConstModuleList):
    __constants__ = ['mods']

    def __init__(self, mods):
        super(_ConstSequential, self).__init__(mods._modules)

        # we define the forward method via self.define rather than
        # making it a direct class member (with a @script) annotation
        # because, in optimized runtime environments where only .pyc files
        # are shipped, we cant retrieve the source code.
        # TODO: find a workaround for this and remove this hack
        self.define("""
        def forward(self, input):
            for m in self:
                input = m(input)
            return input
        """)


def _unwrap_optional(x):
    assert x is not None, "Unwrapping null optional"
    return x

_builtin_table = None

_modules_containing_builtins = (torch, torch._C._nn)

_builtin_ops = [
    # Pairs of (function, op_name)
    (_list_with_default, "aten::list_with_default"),
    (_pair, "aten::_pair"),
    (_quadruple, "aten::_quadruple"),
    (_single, "aten::_single"),
    (_triple, "aten::_triple"),
    (_unwrap_optional, "aten::_unwrap_optional"),
    (_wait, 'aten::wait'),
    (cudnn.is_acceptable, "aten::cudnn_is_acceptable"),
    (math.ceil, "aten::ceil"),
    (math.copysign, "aten::copysign"),
    (math.erf, "aten::erf"),
    (math.erfc, "aten::erfc"),
    (math.exp, "aten::exp"),
    (math.expm1, "aten::expm1"),
    (math.fabs, "aten::fabs"),
    (math.floor, "aten::floor"),
    (math.gamma, "aten::gamma"),
    (math.lgamma, "aten::lgamma"),
    (math.log, "aten::log"),
    (math.log10, "aten::log10"),
    (math.log1p, "aten::log1p"),
    (math.pow, "aten::pow"),
    (math.sqrt, "aten::sqrt"),
    (math.isnan, "aten::isnan"),
    (math.asinh, "aten::asinh"),
    (math.atanh, "aten::atanh"),
    (math.cosh, "aten::cosh"),
    (math.sinh, "aten::sinh"),
    (math.tanh, "aten::tanh"),
    (math.acos, "aten::acos"),
    (math.asin, "aten::asin"),
    (math.atan, "aten::atan"),
    (math.atan2, "aten::atan2"),
    (math.cos, "aten::cos"),
    (math.sin, "aten::sin"),
    (math.tan, "aten::tan"),
    (math.asinh, "aten::asinh"),
    (math.atanh, "aten::atanh"),
    (math.acosh, "aten::acosh"),
    (math.sinh, "aten::sinh"),
    (math.cosh, "aten::cosh"),
    (math.tanh, "aten::tanh"),
    (math.fmod, "aten::fmod"),
    (math.modf, "aten::modf"),
    (math.factorial, "aten::factorial"),
    (math.frexp, "aten::frexp"),
    (math.isnan, "aten::isnan"),
    (math.isinf, "aten::isinf"),
    (math.degrees, "aten::degrees"),
    (math.radians, "aten::radians"),
    (math.ldexp, "aten::ldexp"),
    (torch._C._infer_size, "aten::_infer_size"),
    (torch.nn.functional._no_grad_embedding_renorm_, "aten::_no_grad_embedding_renorm_"),
    (torch.nn.functional.assert_int_or_pair, "aten::_assert_int_or_pair"),
    (torch.nn.functional.interpolate, "aten::__interpolate"),
    (torch.nn.functional.upsample_bilinear, "aten::__upsample_bilinear"),
    (torch.nn.functional.upsample_nearest, "aten::__upsample_nearest"),
    (torch.nn.functional.upsample, "aten::__upsample"),
    (torch.nn.init._no_grad_fill_, "aten::_no_grad_fill_"),
    (torch.nn.init._no_grad_normal_, "aten::_no_grad_normal_"),
    (torch.nn.init._no_grad_uniform_, "aten::_no_grad_uniform_"),
    (torch.nn.init._no_grad_zero_, "aten::_no_grad_zero_"),
    (torch._C._get_tracing_state, "aten::_get_tracing_state"),
    (warnings.warn, "aten::warn"),
]


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

    for builtin, aten_op in _builtin_ops:
        _builtin_table[id(builtin)] = aten_op
    if not PY2:
        _builtin_table[id(math.gcd)] = "aten::gcd"
        _builtin_table[id(math.isfinite)] = "aten::isfinite"
    if PY37:
        _builtin_table[id(math.remainder)] = "aten::mathremainder"

    return _builtin_table


def _register_builtin(fn, op):
    _get_builtin_table()[id(fn)] = op


def _find_builtin(fn):
    return _get_builtin_table().get(id(fn))

# qualified_name => ScriptClass mapping
_script_classes = {}


def _add_script_class(cls, name):
    cls.__torch_script_class__ = True
    global _script_classes
    _script_classes[name] = cls


def _get_script_class(name):
    global _script_classes
    if name not in _script_classes:
        raise RuntimeError("Unknown reference to ScriptClass '{}'. "
                           "Did you forget to import it?".format(name))
    return _script_classes[name]

# overloads are registered in _jit_internal and compiled here so that _overload
# can be used in nn/functional.py without an import cycle

# qualified name => list[compiled fns]
_compiled_overloaded_fns = {}

def _compile_function_with_overload(qual_name, impl_fn, overload_decl, overload_defaults):
    impl_ast = torch.jit.get_jit_def(impl_fn)
    _frames_up = 0
    _rcb = _gen_rcb(impl_fn, _frames_up)
    fn = torch._C._jit_script_compile_overload(qual_name, overload_decl, impl_ast, _rcb, overload_defaults)
    return fn

def _get_overload_decl_and_defaults(func):
    signature = torch.jit.annotations.get_signature(func)
    if signature is None:
        raise RuntimeError("Must explicitly add type annotations to overloaded functions: {obj}").format(func)
    return (torch.jit.get_jit_def(func).decl(), get_default_args(func))

def _get_overloads(obj):
    # check for cached compiled fns
    qual_name = _qualified_name(obj)
    global _compiled_overloaded_fns
    compiled_overloads = _compiled_overloaded_fns.get(qual_name, None)
    if compiled_overloads is not None:
        return compiled_overloads

    # check for not yet compiled overloads
    overloads = _jit_internal._get_fn_overloads(qual_name)
    if overloads is None:
        return None
    compiled_fns = []
    # TODO: use default args from the implementation, not from the overload
    # This is more complicated because you can have a default arg with a type
    # incompatible with a type of parameter in an overload, and other validation.
    # This is still an internal api so for now use defaults from overload
    for overload_fn in overloads:
        overload_decl, overload_defaults = _get_overload_decl_and_defaults(overload_fn)
        compiled_fn = _compile_function_with_overload(qual_name, obj, overload_decl, overload_defaults)
        compiled_fns.append(compiled_fn)

    # cache compilation, remove information stored to do compilation
    _compiled_overloaded_fns[qual_name] = compiled_fns
    _jit_internal._clear_fn_overloads(qual_name)
    return compiled_fns

def _check_directly_compile_overloaded(obj):
    qual_name = _qualified_name(obj)
    global _compiled_overloaded_fns
    global _overloaded_fns
    if qual_name in _compiled_overloaded_fns or _jit_internal._get_fn_overloads(qual_name):
        raise RuntimeError("Function {} cannot be directly compiled because it"
                           " is overloaded. It must be used in a context of a function"
                           " where its inputs can determine which overload to call.".format(qual_name))

# torch.jit.Error
Error = torch._C.JITException

def _get_named_tuple_properties(obj):
    assert issubclass(obj, tuple) and hasattr(obj, '_fields')
    fields = list(obj._fields)
    annotations = []
    has_annotations = hasattr(obj, '__annotations__')
    for field in fields:
        if has_annotations and field in obj.__annotations__:
            annotations.append(torch.jit.annotations.ann_to_type(obj.__annotations__[field]))
        else:
            annotations.append(torch._C.TensorType.get())
    return type(obj).__name__, fields, annotations

def _create_named_tuple(t, unqual_name, field_names):
    TupleType = collections.namedtuple(unqual_name, field_names)
    return TupleType(*t)

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


Attribute = collections.namedtuple('Attribute', ['value', 'type'])

last_executed_optimized_graph = torch._C._last_executed_optimized_graph


def _graph_for(self, *args, **kwargs):
    self(*args, **kwargs)
    return last_executed_optimized_graph()

torch._C.ScriptMethod.graph_for = _graph_for
torch._C.Function.graph_for = _graph_for
Function = torch._C.Function

if not torch._C._jit_init():
    raise RuntimeError("JIT initialization failed")
