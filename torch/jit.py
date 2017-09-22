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
import torch.contrib._graph_vis as graph_vis


HOLE = object()
VOLATILE = object()


# TODO: verify is not implemented yet
def compile(arg=None, nderivs=1, params=tuple(), verify=False, optimize=True, enabled=True):
    """
    Mark a function or module as eligible for just-in-time compilation.  The
    next time the function/module is executed, it is traced, and the trace is
    compiled into an optimized representation which is run in lieu of the
    original Python code upon subsequent invocations of the function/module.

    The result of this function is stateful, so make sure you invoke it
    only once per code you want to JIT compile.

    .. note::

        If your function/module takes non-Variable inputs, the JIT compiler
        will compile a trace separately for each distinct input configuration.

    .. warning::

        Just-in-time compilation currently only works for functions/modules
        which do not have dynamic control flow; if you compile a function/module
        which has dynamic control flow, you will silently get incorrect
        results on subsequent invocations of the model.  Use `verify=True` to
        check that the original Python code and optimized code are equivalent.

    Arguments:
        arg (optional, torch.nn.Module or function): the function or module
            to be compiled.  If `None`, `compile` returns a decorator which can be
            applied to the function or module you want to compile.
        verify (bool, default False): if True, upon all invocations of the
            function/module, execute both the compiled and interpreted versions
            of the model, and verify that their results match.  This is an easy
            (albeit slow) way to check if your function/module can be validly
            JIT compiled.
        nderivs (int, default 1): the number of derivatives which this function/module
            will be used with.  You MUST accurately specify this number: set it too
            low and you will see an error when you attempt to run `backward`;
            set it too high, and the function/module will never be compiled
            (as we always wait to see all derivatives before compiling.)
        optimize (bool, default False): whether or not to apply optimizations.
        enabled (bool, default True): whether or not to actually enable the JIT
            compiler.  This is a convenient way to disable a compilation statement
            without deleting the actual `compile` invocation.

    Example: Compile as function decorator.

        >>> @jit.compile
        >>> def f(x);
        >>>     return x * 2
        >>> x = Variable(torch.randn(1))
        >>> out1 = f(x)  # interpreted run
        >>> out1.sum().backward()  # required!
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
    """
    # TODO: handle decorating a class (not an instance)
    def _compile(inner):
        if enabled:
            return CompiledModule(inner, params=params, nderivs=nderivs, optimize=optimize)
        else:
            return inner
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

    Example: Trace as function decorator.

        >>> @jit.trace
        >>> def f(x);
        >>>     return x * 2
        >>> trace, out = f(Variable(torch.randn(1)))

    Example: Trace as higher order function. (Notice that trace is a *curried*
    function; you first apply it with the function/model to trace, and then
    apply the result with the arguments.)

        >>> traced_model = jit.trace(nn.LSTMCell())
        >>> trace, out = traced_model(input, hidden)

    Example: Trace the backwards pass as function decorator.

        >>> @jit.trace(nderivs=1)
        >>> def f(x);
        >>>     return x * 2
        >>> trace, out = f(Variable(torch.randn(1)))
        >>> out.sum().backward()
        >>> print(trace)

    Example: Trace the backwards pass as higher order function.

        >>> traced_model = jit.trace(nn.LSTMCell(), nderivs=1)
        >>> trace, out = traced_model(input, hidden)
        >>> out.sum().backward()
        >>> print(trace)

    Example: Trace a function with extra parameters. (If you trace
    a Module, parameters are automatically computed via `state_dict`).

        >>> lstm = nn.LSTMCell(10, 20)
        >>> @jit.trace(params=lstm.parameters())
        >>> def f(a, b):
        >>>     return lstm(a, b)
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
        @raw_trace(nderivs=self.nderivs)
        def traced_inner(in_vars, in_struct):
            # NB: Uncommenting this line should be equivalent
            # args, _params = _unflatten(in_vars, in_struct)
            return _flatten(self.inner(*args))

        in_vars, in_struct = _flatten(args, self.state_dict(keep_vars=True).values())
        trace, (out_vars, out_struct) = traced_inner(in_vars, in_struct)
        out, extra = _unflatten(out_vars, out_struct)
        assert len(extra) == 0
        return trace, out


# Functional version that assumes that all parameters are explicitly
# specified
def raw_trace(nderivs=0):
    def _raw_trace(f):
        @functools.wraps(f)
        def wrapper(in_vars, in_struct=None):
            trace = torch._C._tracer_enter(in_vars, nderivs)
            out_vars, out_struct = f(in_vars, in_struct)
            torch._C._tracer_exit(out_vars)
            return trace, (out_vars, out_struct)
        return wrapper
    return _raw_trace


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
    def __init__(self, inner, params=tuple(), **kwargs):
        super(CompiledModule, self).__init__()
        self.inner = inner
        self.params = ParameterList(list(params))
        self.kwargs = kwargs
        self.ktrace_cache = {}

    def _process_args(self, args):
        in_vars, in_struct = _flatten(args, self.state_dict(keep_vars=True).values())
        # TODO: I'm not entirely sure about volatile implies nderivs=0 special case
        is_volatile, in_vars_key = vars_key(in_vars)
        in_key = (in_vars_key, in_struct)
        return in_vars, in_struct, is_volatile, in_key

    # NB: In principle, there could also be a 'raw' version of this compiler,
    # but since the logic is so complicated, testing code wouldn't benefit much
    def forward(self, *args):
        in_vars, in_struct, is_volatile, in_key = self._process_args(args)
        ktrace = self.ktrace_cache.get(in_key)
        if ktrace is None:
            ktrace = TraceForKey(self.inner, volatile=is_volatile, **self.kwargs)
            self.ktrace_cache[in_key] = ktrace
        # TODO: avoid virtual method call here
        closure = ktrace.maybe_closure()
        if closure:
            # We already compiled it!  Run it directly, and
            # use the saved out_struct to unflatten.
            out_vars = closure()(*in_vars)
            out_struct = ktrace.out_struct
        else:
            # No compiled trace available.  Run it by hand.
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
# inputs provided to a call; a TraceForKey represents one such trace. Things
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
    def __init__(self, f, nderivs=1, optimize=True, volatile=False):
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
        @raw_trace(nderivs=self.nderivs)
        def traced_f(in_vars, in_struct):
            #args, _params = _unflatten(in_vars, in_struct)
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
        if self.closure:
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
            # TODO: dump the trace with names
            # (this is a little nontrivial because we have to compute a name
            # based on the input names too)
            p(trace)
            torch._C._jit_pass_lint(trace)

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


@contextlib.contextmanager
def _time(name, enabled=True):
    if not enabled or not torch.cuda.is_available():
        yield
        return
    stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stream.record_event(start)
    yield
    stream.record_event(end)
    end.synchronize()
    print("{} time: {} ms".format(name, start.elapsed_time(end)))


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


if not torch._C._jit_init():
    raise RuntimeError("JIT initialization failed")
