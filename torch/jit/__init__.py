import torch.autograd.function as function
import torch._C
from torch.autograd import Variable
from torch.nn import Module, ParameterList, Parameter
from torch._six import raise_from
from collections import defaultdict
from . import passes as _passes
import itertools
import types
import contextlib
import os
import functools
import inspect


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
def compile(arg=None, verify=False, **kwargs):
    """
    Decorator which marks a function or module class as eligible for
    just-in-time compilation.  The next time the function/module is executed, it
    is traced, and the trace is compiled into an optimized representation which
    is run in lieu of the original Python code upon subsequent invocations of
    the function/module.

    .. note::

        A JIT compiled function/module may be compiled multiple times, as
        different inputs can result in different traces.  Currently, the
        JIT compiler conservatively assumes the trace may change if the
        `size` or `requires_grad` of `Variable` inputs change, or if
        any of the non-Variable inputs change.  For example, if you JIT
        compile an RNN which takes the number of hidden units as a parameter,
        we will compile a trace for every RNN length you use at runtime.

        When a module class is JIT compiled, each instantiation of the module
        gets a separate trace cache.

    .. warning::

        Just-in-time compilation currently only works for functions/modules
        which are not data dependent (e.g., have conditionals on data in
        tensors) and do not have any untracked external dependencies (e.g.,
        perform input/output or access global variables). If you trace such
        models, you will silently get incorrect results on subsequent
        invocations of the model.  You can use `verify=True` to check that the
        original Python code and optimized code are equivalent.

    Keyword arguments:
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

    Debug arguments:
        time (bool, optional): if True, whenever we execute the model in question, we
            will also print out some timing information for how long the model
            took to execute.  At the moment, there are three types of timings we
            emit:

                - unoptimized: the time it took to execute the vanilla Python
                  model.  This only occurs when tracing is disabled, e.g., via
                  `enabled=False`

                - tracing: the time it took to execute the vanilla Python model
                  with tracing enabled.

                - optimized: the time it took to execute the optimized model.

            At the moment, all of these timings are for the forward pass only.
            Default: False.
        enabled (bool, optional): if False, compilation is disabled and you
            will get back your original model.  This is a convenient way to
            disable tracing without having to delete the annotation. Default: True.

    Example: Compile as class decorator.

        >>> @jit.compile
        >>> class MyModel(nn.Module):
        >>>     ...
        >>> model = MyModel()
        >>> out1 = model(x)        # interpreted run
        >>> out1.sum().backward()  # won't compile without this line
        >>> out2 = model(x)        # compiled run
        >>> out2.sum().backward()  # also compiled

    Example: Compile forward pass only as class decorator.

        >>> @jit.compile(nderivs=0)
        >>> class MyModel(nn.Module):
        >>>     ...
        >>> model = MyModel()
        >>> out1 = model(x)        # interpreted run
        >>> out2 = model(x)        # compiled run

    Example: Compile as function decorator.  The same modes of use for the class
    decorator are also supported for functions; however, the decorated
    function must declare *all* Variable inputs in its arguments.

        >>> @jit.compile
        >>> def f(x);
        >>>     return x * 2
    """
    # TODO: handle decorating a class (not an instance)
    def _compile(arg):
        if inspect.isclass(arg):
            if issubclass(arg, _CompiledMixin):
                raise TypeError("Cannot compile a model class that already is compiled")

            # NB: It might seem natural to create a subclass here, rather than
            # make a copy of the class to insert the mixin.  Unfortunately, this
            # will break many user classes.  Suppose you have:
            #
            #     @torch.jit.compile
            #     class Foo(Module):
            #         def __init__(self):
            #             super(Foo, self).__init__() # Python 2 syntax!
            #
            # within the class definition, 'Foo' refers to the *decorated*
            # class, not the undecorated class.  This is bad juju if the
            # decorator returns a subclass, since super(Foo, self) is going to
            # refer to the *undecorated* Foo (and thus you have an infinite
            # loop.)  Python 3's argument-less super() does not have this
            # problem, but in general we cannot ask users to rewrite their code.
            #
            # If we create a *copy* of the class (unrelated to the class the
            # user passed in), this problem goes away, because the class
            # __init__ is a part of is indeed Foo.

            # Make a copy of the class, with the extra _CompiledMixin base
            cls = type(arg.__name__, (_CompiledMixin,) + arg.__bases__, dict(arg.__dict__))

            # Monkey-patch forward and __init__ with the compiler versions
            cls.init_compiler(**kwargs)
            return cls
        elif isinstance(arg, Module):
            # It requires work to compile module instances, because you would
            # like the resulting compiled module to look just like the uncompiled
            # version; actually achieving this requires a bit of fanciness.
            # So for now, we just only support the class mechanism.
            raise TypeError("Compiling model instances is not supported.  "
                            "Use @torch.jit.compile on a class instead.")
        elif callable(arg):
            @compile(**kwargs)
            class FuncModule(Module):
                def __init__(self, f):
                    super(FuncModule, self).__init__()
                    self.f = f

                def forward(self, *args):
                    return self.f(*args)

            return FuncModule(arg)
        else:
            raise TypeError("Cannot handle arg with type {}".format(type(arg)))
    if arg is None:
        return _compile
    else:
        return _compile(arg)


def trace(f, args=tuple(), kwargs=None, nderivs=0):
    """
    Trace a function or model, returning a tuple consisting of the both the
    *trace* of an execution, as well as the original return value.

    Tracing is guaranteed not to change the semantics of the function/module
    that is traced.

    Arguments:
        f (torch.nn.Module or function): the function or module
            to be traced.
        args (tuple or Variable): the positional arguments to pass to the
            function/module to be traced.  A non-tuple is assumed to
            be a single positional argument to be passed to the model.
        kwargs (dict): the keyword arguments to pass to the function/module
            to be traced.
        nderivs (int, default 0): the number of derivatives to trace.
            Traces of derivatives are recorded into the same trace returned
            after executing the `forward` of the resulting module, but
            are not present until you run `backward()` (an appropriate
            number of times) on the resulting model.

    Example: Trace the forwards pass only.

        >>> trace, out = jit.trace(nn.LSTMCell(), (input, hidden))
        >>> print(trace)

    Example: Trace the backwards pass too.

        >>> trace, out = jit.trace(nn.LSTMCell(), (input, hidden), nderivs=1)
        >>> out.sum().backward()
        >>> print(trace)
    """
    if kwargs is None:
        kwargs = {}
    if not isinstance(args, tuple):
        args = (args,)
    return TracedModule(f, nderivs=nderivs)(*args, **kwargs)


class TracedModule(Module):
    def __init__(self, inner, nderivs=0):
        super(TracedModule, self).__init__()
        # inner may be a Module, or it may be an arbitrary callable
        # If it's a Module, we get its parameters automatically, which lets
        # us avoid a special casing functions versus modules.
        self.inner = inner
        self.nderivs = nderivs

    def forward(self, *args, **kwargs):
        # TODO: Possible optimization: use the unflattened
        # output so we don't unflatten it when we get out
        # NB: Not a method because _raw_trace can't deal
        # with methods
        @_raw_trace(nderivs=self.nderivs)
        def traced_inner(in_vars, in_struct):
            return _flatten(self.inner(*args, **kwargs))

        kw_items = list(kwargs.items())
        kw_items.sort()
        in_vars, in_struct = _flatten((args, tuple(kw_items)), self.state_dict(keep_vars=True).values())
        trace, (out_vars, out_struct) = traced_inner(in_vars, in_struct)
        out, unmatched = _unflatten(out_vars, out_struct)
        assert len(unmatched) == 0
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


# Lifecycle of a compiler:
#
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
#
# You should never use this class directly; instead, use compile.  However,
# the intended manual usage of this class looks like this:
#
#     class CompiledModel(_CompiledMixin, nn.Module):
#         def forward(self, x):
#             ...
#     CompiledModule.init_compiler()
#     model = CompiledModule()
#
class _CompiledMixin(object):
    # Global over ALL compilations!  This helps us disambig if two Modules have
    # the same __name__ but actually are different
    __next_id = 0

    @classmethod
    def init_compiler(cls, name=None, enabled=True, time=False, **kwargs):
        # Ensure we are not shadowing this method on the class we mixed with
        assert not hasattr(super(_CompiledMixin, cls), "init_compiler")
        # TODO: Consider saving the backtrace of this constructor, so it's easier
        # to correlate dump files with invocations in Python
        #
        # NB: Use private methods/variables here in order to prevent a class
        # we mix with from accidentally scrambling us
        #
        # NB: Class variables are also accessible via self!
        kwargs["time"] = time  # also want to pass onto ktrace
        cls.__ktrace_kwargs = kwargs
        cls.__enabled = enabled
        cls.__time = time
        cls.__model_name = name

        # Monkey patch the constructor and forward functions *inplace*
        cls.__old_forward = cls.forward
        cls.forward = cls.__new_forward
        cls.__old_init = cls.__init__
        cls.__init__ = cls.__new_init

    def __new_init(self, *args, **kwargs):
        try:
            # __old_init is assumed to handle super call
            self.__old_init(*args, **kwargs)
        except TypeError as e:
            # If this fails here, the user probably didn't use this as a class
            # decorator
            if "super" in str(e):
                raise_from(TypeError("torch.jit.compile must be used as a class decorator; "
                                     "using it on an already defined class is not valid."
                                     "\n\nOriginal error: {}".format(str(e))), e)
            else:
                raise
        model_name = self.__model_name if self.__model_name else type(self).__name__
        self.__name = "jit_{}_{}".format(model_name, _CompiledMixin.__next_id)
        _CompiledMixin.__next_id += 1
        self.__ktrace_cache = {}
        self.__next_ktrace_id = 0

    def __process_args(self, args):
        in_vars, in_struct = _flatten(args, self.state_dict(keep_vars=True).values())
        is_volatile, in_vars_key = vars_key(in_vars)
        in_key = (in_vars_key, in_struct)
        return in_vars, in_struct, is_volatile, in_key

    # NB: In principle, there could also be a 'raw' version of this compiler,
    # but since the logic is so complicated, testing code wouldn't benefit much
    def __new_forward(self, *args):
        if _JIT_DISABLE or not self.__enabled:
            with _time(self.__name, "unoptimized", self.__time):
                # Call to the saved old forward function
                return self.__old_forward(*args)
        in_vars, in_struct, is_volatile, in_key = self.__process_args(args)
        ktrace = self.__ktrace_cache.get(in_key)
        if ktrace is None:
            ktrace_name = '{}_{}'.format(self.__name, self.__next_ktrace_id)
            self.__next_ktrace_id += 1
            ktrace = TraceForKey(ktrace_name, in_key, volatile=is_volatile, **self.__ktrace_kwargs)
            self.__ktrace_cache[in_key] = ktrace
        closure = ktrace.maybe_closure()
        if closure is not None:
            # We already compiled it!  Run it directly, and
            # use the saved out_struct to unflatten.
            with _time(ktrace.name, "optimized", self.__time):
                out_vars = closure()(*in_vars)
                out_struct = ktrace.out_struct
        else:
            # No compiled trace available.  Run it by hand.
            with _time(ktrace.name, "tracing", self.__time):
                out_vars, out_struct = ktrace.add_trace(self.__old_forward, args, in_vars, in_struct)
        if isinstance(out_vars, Variable):
            out_vars = (out_vars, )
        out, unmatched = _unflatten(out_vars, out_struct)
        assert len(unmatched) == 0
        return out

    def has_trace_for(self, *args):
        # Ensure we are not shadowing this method on the class we mixed with
        assert not hasattr(super(_CompiledMixin, self), "has_trace_for")
        in_vars, in_struct, is_volatile, in_key = self.__process_args(args)
        ktrace = self.__ktrace_cache.get(in_key)
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
    def __init__(self, name, key, nderivs=1, optimize=True, volatile=False, time=False):
        self.name = name
        self.key = key
        # TODO: Not convinced about this volatile special case...
        self.nderivs = nderivs if not volatile else 0
        self.optimize = optimize
        self.traces = []
        self.closure = None
        self.out_struct = None  # initialized when we call trace, checked thereafter
        self.time = time

    # The signature here is a little goofy; it's a perf optimization.
    # Additionally, f is passed in as an argument (even though it is fixed as
    # class initialization) to avoid a circular reference.
    def add_trace(self, f, args, in_vars, in_struct):
        # TODO: Deduplicate this code
        @_raw_trace(nderivs=self.nderivs)
        def traced_f(in_vars, in_struct):
            return _flatten(f(*args))

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

        with _time(self.name, "compiling", self.time):
            _dump_trace(self.name, "init", self.key, complete_trace)

            # It's important to always run DCE, because backward can create a lot of unnecessary nodes
            _run_pass(torch._C._jit_pass_dce, complete_trace)
            _run_pass(torch._C._jit_pass_onnx, complete_trace)
            _run_pass(_passes._check_inplace, complete_trace)
            if self.optimize:
                _run_pass(torch._C._jit_pass_fuse, complete_trace)

            _dump_trace(self.name, "final", self.key, complete_trace)

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
        gpu_rng_state_all = torch.cuda.get_rng_state_all()

    yield

    torch.set_rng_state(cpu_rng_state)
    if gpu_rng_state is not None:
        torch.cuda.set_rng_state_all(gpu_rng_state_all)


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


if not torch._C._jit_init():
    raise RuntimeError("JIT initialization failed")
