import torch.autograd.function as function
import torch._C
from torch.autograd import Variable
from torch.nn import Module
import itertools
import types
import contextlib
import os
# Example how to use:
#
# import torch.jit
# model = model.RNNModel(args.model, ...)
# model = torch.jit.traced(model)


def flatten(x):
    return tuple(function._iter_variables(x))


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


def _clone_inputs(all_args):
    return tuple(Variable(a.data.clone(), requires_grad=a.requires_grad, volatile=a.volatile) for a in all_args)


def _verify(flat_trace_out, flat_real_out):
    # test for equality
    for x, y in zip(flat_trace_out, flat_real_out):
        if not (isinstance(x, Variable) and isinstance(y, Variable)):
            raise RuntimeError("non-Variable output")
        if x.data.sub(y.data).abs().max() > 1e-6:
            raise RuntimeError("JIT and real computation mismatch")


# holds run() to run the function and self.inputs which
# are all the variable inputs
class Traceable(object):
    _next_trace_id = 0
    _dump_traces = os.environ.get('PYTORCH_JIT_DUMP', False)

    def __init__(self, function_or_module, trace_name=None, optimize=False, verify=False, time=False, enabled=True):
        """
        time - collect cuda timing stats for perf debugging
        verify - run the original code, and check it is within threshold
        optimize - run optimizations like fusion on the trace before running
        enabled - flag to turn off tracing so you can check timing of stuff that cannot be traced
        """

        if isinstance(function_or_module, Module):
            self._run = function_or_module.forward
            self._additional_inputs = lambda: function_or_module.parameters()
        else:
            self._run = function_or_module
            self._additional_inputs = lambda: ()

        self.trace_name = trace_name
        self.saved_trace = None
        self.saved_closure = None
        self.optimize = optimize
        self.verify = verify
        self.time = time
        self.enabled = enabled
        if self.trace_name is None:
            self.trace_name = "trace_{}".format(Traceable._next_trace_id)
            Traceable._next_trace_id += 1

    def _run_pass(self, p, suffix=''):
        name = p.__doc__
        if Traceable._dump_traces:
            with open("{}_{}_input.ir".format(self.trace_name, name), "w") as f:
                f.write(str(self.saved_trace))
        p(self.saved_trace)
        torch._C._jit_pass_lint(self.saved_trace)
        if Traceable._dump_traces:
            with open("{}_{}_output.ir".format(self.trace_name, name), "w") as f:
                f.write(str(self.saved_trace))

    def run_trace(self, trace_inputs):
        if self.saved_closure is None:
            self.saved_closure = torch._C._jit_createAutogradClosure(
                self.saved_trace)
        with _time("run_trace", self.time):
            assert(self.saved_closure is not None)
            return Variable._execution_engine.run_forward(
                self.saved_closure, trace_inputs)

    def get_trace_inputs(self, args, extra):
        return tuple(x for x in itertools.chain(self._additional_inputs(), flatten(args), extra))

    # create and return a trace, possibly verifying it before returning it
    def record_trace(self, args, extra):
        trace_inputs = self.get_trace_inputs(args, extra)
        if self.verify:
            cloned_inputs = _clone_inputs(trace_inputs)
        with _time("record_trace", self.time), _fork_rng(self.verify):
            self.saved_trace, inputs = torch._C._tracer_enter(trace_inputs)
            out = self._run(*args)
            flat_out = torch._C._tracer_exit(flatten(out))

        torch._C._jit_pass_lint(self.saved_trace)
        if self.optimize:
            self._run_pass("init", torch._C._jit_pass_init)
            self._run_pass("fuse", torch._C._jit_pass_fuse)

        if self.verify:
            flat_trace_out = self.run_trace(cloned_inputs)
            _verify(flat_trace_out, flat_out)

        return self.saved_trace, function._unflatten(flat_out, out)

    def run(self, args, extra):
        # tracing is disabled, run the real thing, possibly timing it
        if not self.enabled:
            with _time("run_real", self.time):
                return self._run(*args)
        # tracing, but no trace exists, create one, possibly verifying it
        # by running it after creating it
        if self.saved_trace is None:
            _, out = self.record_trace(args, extra)
            self.proto = function._to_proto(out)
            return out
        trace_inputs = self.get_trace_inputs(args, extra)

        # just run the already created trace
        if not self.verify:
            return function._unflatten(self.run_trace(trace_inputs), self.proto)

        # verify an already created trace...
        cloned_inputs = _clone_inputs(trace_inputs)
        with _time("run_real", self.time), _fork_rng():
            out_real = self._run(*args)
        flat_trace_out = self.run_trace(cloned_inputs)
        _verify(flat_trace_out, flatten(out_real))
        return out_real


def record_trace(traceable, *args, **kwargs):
    parameters = kwargs.pop('parameters', ())
    return Traceable(traceable, **kwargs).record_trace(
        args, parameters)


def traced(traceable, **traced_kwargs):
    parameters = traced_kwargs.pop('parameters', ())
    t = Traceable(traceable, **traced_kwargs)
    if isinstance(traceable, Module):
        def forward(self, *args):
            return t.run(args, ())
        traceable.forward = types.MethodType(forward, traceable)
        return traceable
    else:
        return lambda *args: t.run(args, traced_kwargs.get('parameters', ()))


if not torch._C._jit_init():
    raise RuntimeError("JIT initialization failed")
