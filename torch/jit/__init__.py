import torch.autograd.function as function
import torch._C
from torch import Tensor
from torch.autograd import Variable
from torch.nn import Module, ParameterList, Parameter
from torch._six import raise_from
from collections import defaultdict
import warnings
import itertools
import types
import contextlib
import os
import functools
import inspect
import copy


_flatten = torch._C._jit_flatten
_unflatten = torch._C._jit_unflatten


def compile(arg=None, nderivs=1, optimize=True, enabled=True):
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
        invocations of the model.

    Keyword arguments:
        nderivs (int, optional): the number of derivatives which this function/module
            will be used with.  You MUST accurately specify this number: set it too
            low and you will see an error when you attempt to run `backward`;
            set it too high, and the function/module will never be compiled
            (as we always wait to see all derivatives before compiling.)
            Default: 1 (i.e., we will compile forwards and backwards, but not
            double-backwards).
        optimize (bool, optional): whether or not to apply optimizations.  Default: ``True``.

    Debug arguments:
        time (bool, optional): if ``True``, whenever we execute the model in question, we
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
            Default: ``False``.
        enabled (bool, optional): if ``False``, compilation is disabled and you
            will get back your original model.  This is a convenient way to
            disable tracing without having to delete the annotation. Default: ``True``.

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
        >>> def f(x):
        >>>     return x * 2
    """
    def _compile(arg):
        if inspect.isclass(arg):
            class CompiledModuleMeta(type):
                def __call__(cls, *args, **kwargs):
                    # NOTE: this is called whenever an instance of this class is created
                    # The super call below will call __new__ and __init__, and we will
                    # patch things later.
                    try:
                        obj = super(CompiledModuleMeta, cls).__call__(*args, **kwargs)
                    except TypeError as e:
                        # If this fails here, the user probably didn't use this as a class decorator
                        if "super" in str(e):
                            raise_from(TypeError("torch.jit.compile must be used as a class decorator; "
                                                 "using it on an already defined class is not valid."
                                                 "\n\nOriginal error: {}".format(str(e))), e)
                        else:
                            raise

                    compiled_fn = torch._C.CompiledFunction(nderivs, optimize,
                                                            obj.forward,
                                                            arg.__name__)
                    compiled_fn.enabled = enabled
                    obj.compiled_fn = compiled_fn
                    obj.forward = lambda *args: compiled_fn(args, list(obj.parameters()))
                    obj.has_trace_for = lambda *args: compiled_fn.has_trace_for(args, list(obj.parameters()))
                    return obj

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

            return type(arg.__name__,
                        (torch._six.with_metaclass(CompiledModuleMeta, *arg.__bases__),),
                        dict(arg.__dict__))
        elif isinstance(arg, Module):
            # It requires work to compile module instances, because you would
            # like the resulting compiled module to look just like the uncompiled
            # version; actually achieving this requires a bit of fanciness.
            # So for now, we just only support the class mechanism.
            raise TypeError("Compiling model instances is not supported.  "
                            "Use @torch.jit.compile on a class instead.")
        elif callable(arg):
            module = type(arg.__name__, (torch.nn.Module,), {'forward': lambda self, *args: arg(*args)})
            return _compile(module)()
        else:
            raise TypeError("Cannot handle arg with type {}".format(type(arg)))
    # Make empty parenthesis optional
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

    def forward(self, *args):
        in_vars, _, _ = _flatten((args, list(self.parameters())))
        return _get_trace(self.inner, args, in_vars, self.nderivs)


# Functional version that assumes that all parameters are explicitly
# specified
def _get_trace(f, args, in_vars, nderivs=0):
    trace = torch._C._tracer_enter(in_vars, nderivs)
    out = f(*args)
    out_vars, _, _ = _flatten(out)
    torch._C._tracer_exit(out_vars)
    return trace, out


def _clone_inputs(args):
    def clone_input(a):
        if a is None:
            return None
        elif isinstance(a, Variable):
            v = Variable(a.data.clone(), requires_grad=a.requires_grad, volatile=a.volatile)
            if a.grad is not None:
                v.grad = clone_input(v.grad)
            return v
        else:
            return a.clone()
    return function._nested_map(lambda o: isinstance(o, Variable) or torch.is_tensor(o),
                                clone_input)(args)


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
        args (tuple or Variable): the positional arguments to pass to the
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
    if not hasattr(model, 'compiled_fn') or not isinstance(model.compiled_fn, torch._C.CompiledFunction):
        raise TypeError("Cannot verify an uncompiled module.  Add @torch.jit.compile to compile it")

    if not isinstance(args, tuple):
        args = (args,)

    saved_args = _clone_inputs(args)
    saved_state = copy.deepcopy(model.state_dict())

    def run_fwd_bwd(args, force_trace=False, assert_compiled=False):
        in_vars, _, _ = _flatten((args, list(model.parameters())))
        # We use a special API to reset the trace and compile it from scratch.
        compiled_fn = model.compiled_fn
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
        out_vars, _, _ = _flatten(out)
        saved_outs = [v.data.clone() for v in out_vars]
        loss = loss_fn(*out)
        grads = torch.autograd.grad([loss], in_vars)
        # TODO: I'm not sure if the clone here is necessary but it is safer
        saved_grads = [v.data.clone() for v in grads]
        return (saved_outs, saved_grads)

    with torch.random.fork_rng(devices, _caller="torch.jit.verify"):
        uncompiled_outs, uncompiled_grads = run_fwd_bwd(args, force_trace=True)
        assert model.has_trace_for(*args)

    model.load_state_dict(saved_state)
    compiled_outs, compiled_grads = run_fwd_bwd(args, assert_compiled=True)

    _verify_equal(uncompiled_outs, compiled_outs)
    _verify_equal(uncompiled_grads, compiled_grads)


def _verify_equal(xs, ys):
    for x, y in zip(xs, ys):
        if x.sub(y).abs().max() > 1e-6:
            raise RuntimeError("JIT and real computation mismatch")


if not torch._C._jit_init():
    raise RuntimeError("JIT initialization failed")
