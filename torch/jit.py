import torch.autograd.function as function
import torch._C
from torch.autograd import Variable
from torch.nn import Module, ParameterList, Parameter
from collections import defaultdict
import itertools
import types
import contextlib
import os
import functools


class Placeholder(object):
    def __init__(self, s):
        self.s = s
    def __str__(self):
        return self.s
    def __repr__(self):
        return self.s


HOLE = Placeholder("HOLE")
VOLATILE = Placeholder("VOLATILE")


# TODO: verify is not implemented yet
def compile(arg=None, nderivs=1, params=tuple(), name=None, verify=False, optimize=True):
    """
    Mark a function or module as eligible for just-in-time compilation.  The
    next time the function/module is executed, it is traced, and the trace is
    compiled into an optimized representation which is run in lieu of the
    original Python code upon subsequent invocations of the function/module.

    The result of this function is stateful, so make sure you invoke it
    only once per code you want to JIT compile.

    .. note::

        A JIT compiled function/module may be compiled multiple times, as
        different inputs can result in different traces.  Currently, the
        JIT compiler conservatively assumes the trace may change if the
        `size` or `requires_grad` of `Variable` inputs change, or if
        any of the non-Variable inputs change.  For example, if you JIT
        compile an RNN which takes the number of hidden units as a parameter,
        we will compile a trace for every RNN length you use at runtime.

    .. warning::

        Just-in-time compilation currently only works for functions/modules
        which are not data dependent (e.g., have conditionals on data in
        tensors) and do not have any untracked external dependencies (e.g.,
        perform input/output or access global variables). If you trace such
        models, you will silently get incorrect results on subsequent
        invocations of the model.  You can use `verify=True` to check that the
        original Python code and optimized code are equivalent.

    Arguments:
        arg (torch.nn.Module or function, optional): the function or module
            to be compiled.  If `None`, `compile` returns a decorator which can be
            applied to the function or module you want to compile.  See the
            examples for how to use both modes.  Default: None.
        verify (bool, optional): if True, upon all invocations of the
            function/module, execute both the compiled and interpreted versions
            of the model, and verify that their results match.  This is an easy
            (albeit slow) way to check if your function/module can be validly
            JIT compiled.  Default: False.
        nderivs (int, optional): the number of derivatives which this function/module
            will be used with.  You MUST accurately specify this number: set it too
            low and you will see an error when you attempt to run `backward`;
            set it too high, and the function/module will never be compiled
            (as we always wait to see all derivatives before compiling.)
            Default: 1 (i.e., we will compile forwards and backwards, but not
            double-backwards).
        optimize (bool, optional): whether or not to apply optimizations.  Default: True.

    Example: Compile as function decorator.

        >>> @jit.compile
        >>> def f(x);
        >>>     return x * 2
        >>> x = Variable(torch.randn(1))
        >>> out1 = f(x)  # interpreted run
        >>> out1.sum().backward()  # won't compile without this line
        >>> out2 = f(x)  # compiled run
        >>> out2.sum().backward()

    Example: Compile as higher order function. (Notice that compile is a *curried*
    function; you first apply it with the function/model to trace, and then
    apply the result with the arguments.)

        >>> compiled_model = jit.compile(nn.LSTMCell())
        >>> out = compiled_model(input, hidden)

    Example: Compile forwards only as function decorator

        >>> @jit.compile(nderivs=0)
        >>> def f(x);
        >>>     return x * 2
        >>> out1 = f(x)  # interpreted run
        >>> out2 = f(x)  # compiled run

    Example: Compile forwards only as higher order function

        >>> compiled_model = jit.compile(nn.LSTMCell(), nderivs=0)
        >>> out1 = compiled_model(input, hidden)  # interpreted run
        >>> out2 = compiled_model(input, hidden)  # compiled run

    Example: Compile a function with extra parameters. (If you compile
    a Module, parameters are automatically computed via `state_dict`).

        >>> lstm = nn.LSTMCell(10, 20)
        >>> @jit.compile(params=lstm.parameters())
        >>> def f(a, b):
        >>>     return lstm(a, b)
    """
    # TODO: handle decorating a class (not an instance)
    def _compile(inner):
        return CompiledModule(inner, params=params, nderivs=nderivs, optimize=optimize, name=name)
    if callable(arg):
        return _compile(arg)
    else:
        return _compile


def trace(arg=None, nderivs=0, params=tuple()):
    """
    Instrument a function or module for tracing, wrapping it in a
    :class:`TracedModule`, whose forward accepts the same arguments as the
    original function/module, but returns a tuple consisting of the
    *trace* of an execution, as well as the original return value.

    Tracing is guaranteed not to change the semantics of the function/module
    that is traced.

    Arguments:
        arg (optional, torch.nn.Module or function): the function or module
            to be traced.  If `None`, `trace` returns a decorator which can be
            applied to the function or module you want to trace.
        nderivs (int, default 0): the number of derivatives to trace.
            Traces of derivatives are recorded into the same trace returned
            after executing the `forward` of the resulting module, but
            are not present until you run `backward()` (an appropriate
            number of times) on the resulting model.
        params (tuple of torch.nn.Parameter): extra parameters for a traced
            function, which do not occur as arguments to the function in
            question.  You generally do not need this for tracing modules, as
            the parameters of a module are automatically computed.

    Example: Trace as higher order function. (Notice that trace is a *curried*
    function; you first apply it with the function/model to trace, and then
    apply the result with the arguments.)

        >>> traced_model = jit.trace(nn.LSTMCell())
        >>> trace, out = traced_model(input, hidden)

    Example: Trace the backwards pass as higher order function.

        >>> traced_model = jit.trace(nn.LSTMCell(), nderivs=1)
        >>> trace, out = traced_model(input, hidden)
        >>> out.sum().backward()
        >>> print(trace)
    """
    # TODO: handle decorating a class (not a callable)
    def _trace(inner):
        return TracedModule(inner, nderivs=nderivs, params=params)
    if callable(arg):
        return _trace(arg)
    else:
        return _trace


# TODO: Formulating it this way means that state_dict of the traced thing
# looks different, etc (because there's extra nesting).  Figure out a
# way to avoid this
class TracedModule(Module):
    def __init__(self, inner, params=tuple(), nderivs=0):
        super(TracedModule, self).__init__()
        # inner may be a Module, or it may be an arbitrary callable
        self.inner = inner
        self.params = ParameterList(list(params))
        self.nderivs = nderivs

    def forward(self, *args):
        # TODO: Possible optimization: use the unflattened
        # output so we don't unflatten it when we get out
        # NB: Not a method because trace_func_raw can't deal
        # with methods
        @_raw_trace(nderivs=self.nderivs)
        def traced_inner(in_vars, in_struct):
            return _flatten(self.inner(*args))

        in_vars, in_struct = _flatten(args, self.state_dict(keep_vars=True).values())
        trace, (out_vars, out_struct) = traced_inner(in_vars, in_struct)
        out, extra = _unflatten(out_vars, out_struct)
        assert len(extra) == 0
        return trace, out


# Functional version that assumes that all parameters are explicitly
# specified
def _raw_trace(nderivs=0):
    def raw_trace(f):
        # f takes two arguments, (in_vars, in_struct) (as determined
        # by _flatten); furthermore, it must be the case that in_vars
        # contains all Variable inputs (including parameters.)  It must
        # produce two outputs, (out_vars, out_struct) (also as determined
        # by _flatten).
        @functools.wraps(f)
        def wrapper(in_vars, in_struct=None):
            trace = torch._C._tracer_enter(in_vars, nderivs)
            out_vars, out_struct = f(in_vars, in_struct)
            torch._C._tracer_exit(out_vars)
            return trace, (out_vars, out_struct)
        return wrapper
    return raw_trace


# Lifecycle of a CompiledModule:
# - It is given an underlying function, which knows how to actually
#   execute the code that we want to compile.
# - When we encounter an input configuration for which we don't
#   have an optimized trace, we run the underlying function, tracing its
#   result.  The trace is not done yet, so we save it into our set of pending
#   traces for that configuration.
# - When we encounter an input configuration whose trace is "ready"
#   (that is, we've seen all of the passes, so the trace contains
#   forwards/backwards/etc), we compile it, and then register this
#   as the compiled trace.
# - When we encounter an input configuration whose trace is compiled,
#   we just directly run the compiled trace.
class CompiledModule(Module):
    _next_id = 0

    def __init__(self, inner, params=tuple(), name=None, **kwargs):
        # TODO: Consider saving the backtrace of this constructor, so it's easier
        # to correlate dump files with invocations in Python
        super(CompiledModule, self).__init__()
        self.inner = inner
        self.params = ParameterList(list(params))
        self.kwargs = kwargs
        self.ktrace_cache = {}
        if name is None:
            name = "jit_compile_{}".format(CompiledModule._next_id)
            CompiledModule._next_id += 1
        self.name = name
        self._next_ktrace_id = 0

    def _process_args(self, args):
        in_vars, in_struct = _flatten(args, self.state_dict(keep_vars=True).values())
        is_volatile, in_vars_key = vars_key(in_vars)
        in_key = (in_vars_key, in_struct)
        return in_vars, in_struct, is_volatile, in_key

    # NB: In principle, there could also be a 'raw' version of this compiler,
    # but since the logic is so complicated, testing code wouldn't benefit much
    def forward(self, *args):
        if _JIT_DISABLE:
            with _time(self.name, "unoptimized"):
                return self.inner(*args)
        in_vars, in_struct, is_volatile, in_key = self._process_args(args)
        ktrace = self.ktrace_cache.get(in_key)
        if ktrace is None:
            ktrace_name = '{}_{}'.format(self.name, self._next_ktrace_id)
            self._next_ktrace_id += 1
            ktrace = TraceForKey(ktrace_name, in_key, self.inner, volatile=is_volatile, **self.kwargs)
            self.ktrace_cache[in_key] = ktrace
        closure = ktrace.maybe_closure()
        if closure is not None:
            # We already compiled it!  Run it directly, and
            # use the saved out_struct to unflatten.
            with _time(ktrace.name, "optimized"):
                out_vars = closure()(*in_vars)
                out_struct = ktrace.out_struct
        else:
            # No compiled trace available.  Run it by hand.
            with _time(ktrace.name, "tracing"):
                out_vars, out_struct = ktrace.add_trace(args, in_vars, in_struct)
        if isinstance(out_vars, Variable):
            out_vars = (out_vars, )
        out, extras = _unflatten(out_vars, out_struct)
        assert len(extras) == 0
        return out

    def has_trace_for(self, *args):
        in_vars, in_struct, is_volatile, in_key = self._process_args(args)
        ktrace = self.ktrace_cache.get(in_key)
        if ktrace is None:
            return False
        return ktrace.maybe_closure() is not None

    # TODO: Provide more compiled code management utility methods


# CompiledModule memoizes multiple traces and switches between them based on
# inputs provided to a call; a TraceForKey logically represents one such trace
# (in reality, a TraceForKey may contain multiple traces, but they all share
# the same input configuration and should be equivalent). Things
# that need to be considered include non-Variable argument (e.g. num_layers=3;
# compared by equality) or Variable flags and sizes. TraceForKey is the object
# that is used to hold a trace / compiled code for a single input configuration
# aka in_key.
class TraceForKey(object):
    # Lifecycle:
    #   - We accumulate 'traces'
    #   - At some point, one of these traces becomes complete ('is_complete'
    #     is True).  This occurs when we run enough backwards on a trace
    #     to complete it (i.e., this is an external event to this object).
    #   - Whenever we want to run this trace, we call 'maybe_closure'.  This
    #     returns None if we don't have a complete trace yet, or the
    #     autograd closure to actually run the trace if we do.
    def __init__(self, name, key, f, nderivs=1, optimize=True, volatile=False):
        self.name = name
        self.key = key
        self.f = f
        # TODO: Not convinced about this volatile special case...
        self.nderivs = nderivs if not volatile else 0
        self.optimize = optimize
        self.traces = []
        self.closure = None
        self.out_struct = None # initialized when we call trace, checked thereafter

    # The signature here is a little goofy; it's a perf optimization
    def add_trace(self, args, in_vars, in_struct):
        # TODO: Deduplicate this code
        @_raw_trace(nderivs=self.nderivs)
        def traced_f(in_vars, in_struct):
            return _flatten(self.f(*args))

        trace, (out_vars, out_struct) = traced_f(in_vars, in_struct)
        if self.out_struct is None:
            self.out_struct = out_struct
        else:
            # TODO: in debug mode, assert the output structs are same
            pass
        self.traces.append(trace)
        return out_vars, out_struct

    def maybe_closure(self):
        if self.closure is not None:
            return self.closure

        # GC expired traces
        self.traces = [t for t in self.traces if not t.is_expired]

        # Search for a complete trace
        complete_trace = None
        for trace in self.traces:
            if trace.is_complete:
                complete_trace = trace
                self.traces = []

        if complete_trace is None:
            return None

        def _run_pass(p, trace):
            pass_name = p.__name__.replace('_jit_pass_', '')
            p(trace)
            _dump_trace(self.name, pass_name, self.key, trace)
            torch._C._jit_pass_lint(trace)

        _dump_trace(self.name, "trace", self.key, complete_trace)

        # It's important to always run DCE, because backward can create a lot of unnecessary nodes
        _run_pass(torch._C._jit_pass_dce, complete_trace)
        if self.optimize:
            _run_pass(torch._C._jit_pass_onnx, complete_trace)
            _run_pass(torch._C._jit_pass_fuse, complete_trace)

        self.closure = torch._C._jit_createAutogradClosure(complete_trace)
        return self.closure


def vars_key(in_vars):
    """
    Compute the key for variables: some properties of variables
    affect the trace, e.g., size and requires_grad.
    """
    is_volatile = any(x.volatile if isinstance(x, Variable) else False for x in in_vars)
    def var_key(x):
        if isinstance(x, Variable):
            grad_key = x.requires_grad
            ty = x.data.type()
        else:
            grad_key = False
            ty = x.type()
        if is_volatile:
            grad_key = VOLATILE
        return ty, grad_key, x.size()
    return is_volatile, tuple(map(var_key, in_vars))


@contextlib.contextmanager
def _fork_rng(enabled=True):
    """
    Forks the RNG, so that when you return, the RNG is reset
    to the state that it was previously in.  This is important
    if we are evaluating a trace twice, and it incorporates
    randomness: if we don't reset the seed, we might get totally
    different results!

    TODO: Randomness in models is a big problem for reproduceability,
    because it means if we start executing models out of order,
    they may behave differently.  Interesting question is whether
    or not backwards pass ever has randomness.  I hope not.
    """
    if not enabled:
        yield
        return

    cpu_rng_state = torch.get_rng_state()
    gpu_rng_state = None
    if torch.cuda.is_available():
        gpu_rng_state = torch.cuda.get_rng_state()

    yield

    torch.set_rng_state(cpu_rng_state)
    if gpu_rng_state is not None:
        torch.cuda.set_rng_state(gpu_rng_state)


# _flatten and _unflatten are inverses
def _unflatten(input, proto):
    def unflatten_helper(input, proto):
        res = []
        if not isinstance(proto, (list, tuple)):
            return input[0], input[1:]
        for e in proto:
            res_e, input = unflatten_helper(input, e)
            res.append(res_e)
        return type(proto)(res), input

    return unflatten_helper(input, proto)


def _flatten(obj, params=tuple()):
    obj_vars = tuple(itertools.chain(function._iter_variables(obj), params))
    obj_struct = function._nested_map(lambda o: isinstance(o, Variable), lambda x: HOLE)(obj)
    return obj_vars, obj_struct


# This is purely for developer debugging.  We are not going to advertise it.
_JIT_DUMP = os.environ.get('PYTORCH_JIT_DUMP', False)
_JIT_TIME = os.environ.get('PYTORCH_JIT_TIME', False)  # CUDA-only timing
_JIT_DISABLE = os.environ.get('PYTORCH_JIT_DISABLE', False)


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
def _time(trace_name, name):
    if not _JIT_TIME or not torch.cuda.is_available():
        yield
        return
    stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stream.record_event(start)
    yield
    stream.record_event(end)
    end.synchronize()
    print("{} {} time: {} ms".format(trace_name, name, start.elapsed_time(end)))


if not torch._C._jit_init():
    raise RuntimeError("JIT initialization failed")
