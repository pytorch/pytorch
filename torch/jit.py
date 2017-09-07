import torch.autograd.function as function
import torch._C
from torch.autograd import Variable
from torch.nn import Module
from collections import defaultdict
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
    """
    Flatten an arbitrarily nested structure of Variables into
    a tuple of Variables.
    """
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


def _varify(args):
    return tuple(a if isinstance(a, Variable) else Variable(a, requires_grad=False) for a in args)


def _clone_inputs(all_args):
    for a in all_args:
        if isinstance(a, Variable):
            yield Variable(a.data.clone(), requires_grad=a.requires_grad, volatile=a.volatile)
        else:
            yield a.clone()


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
    VOLATILE = object()

    # Traceable holds multiple traces and switches between them based on
    # inputs provided to a call. Things that need to be considered include
    # non-Variable argument (e.g. num_layers=3; compared by equality) or
    # Variable flags and sizes. TraceInfo is the object that is used to
    # hold a trace for a single input configuration aka input_key.
    class TraceInfo(object):
        def __init__(self, trace_name):
            self.traces = []
            self.complete_trace = None
            self.closure = None
            self.trace_name = trace_name
            self.proto = None

        def _run_pass(self, p):
            name = p.__name__.replace('_jit_pass_', '')
            if Traceable._dump_traces:
                with open("{}_{}_input.ir".format(self.trace_name, name), "w") as f:
                    f.write(str(self.complete_trace))
            p(self.complete_trace)
            # TODO: Make linting optional
            torch._C._jit_pass_lint(self.complete_trace)
            if Traceable._dump_traces:
                with open("{}_{}_output.ir".format(self.trace_name, name), "w") as f:
                    f.write(str(self.complete_trace))

        def compile_trace(self, optimize):
            # It's important to always run DCE, because backward can create a lot of unnecessary nodes
            self._run_pass(torch._C._jit_pass_dce)
            if optimize:
                self._run_pass(torch._C._jit_pass_onnx)
                # TODO: fuse only when using CUDA
                self._run_pass(torch._C._jit_pass_fuse)

            self.closure = torch._C._jit_createAutogradClosure(self.complete_trace)

        def check_traces(self):
            self.traces = [t for t in self.traces if not t.is_expired]
            for trace in self.traces:
                if trace.is_complete:
                    self.complete_trace = trace
                    self.traces = []

    def __init__(self, function_or_module, num_derivatives=1, parameters=None, trace_name=None,
                 optimize=False, verify=False, time=False, enabled=True):
        """
        time - collect cuda timing stats for perf debugging
        verify - run the original code, and check it is within threshold
        optimize - run optimizations like fusion on the trace before running
        enabled - flag to turn off tracing so you can check timing of stuff that cannot be traced
        """

        if isinstance(function_or_module, Module):
            self._run = function_or_module.forward
            self._state_values = lambda: function_or_module.state_dict(keep_vars=True).values()
        else:
            self._run = function_or_module
            param_list = list(parameters) if parameters is not None else []
            self._state_values = lambda: param_list

        if trace_name is None:
            trace_name = "trace_{}".format(Traceable._next_trace_id)
            Traceable._next_trace_id += 1

        self.trace_name = trace_name
        self.optimize = optimize
        self.verify = verify
        self.time = time
        self.enabled = enabled
        self.num_derivatives = num_derivatives
        self.traces = defaultdict(lambda: Traceable.TraceInfo(trace_name))

    def get_input_key(self, args):
        is_volatile = any(arg.volatile if isinstance(arg, Variable) else False for arg in args)
        if is_volatile:
            def get_var_key(var):
                return (var.size(), self.VOLATILE)
        else:
            def get_var_key(var):
                return (var.size(), var.requires_grad)
        return tuple(get_var_key(arg) if isinstance(arg, Variable) else arg for arg in args)

    def get_trace_inputs(self, args, extra=()):
        return tuple(itertools.chain(self._state_values(), flatten(args), extra))

    def run_closure(self, trace_info, args, trace_inputs):
        if self.verify:
            cloned_args = tuple(_clone_inputs(args))
            with _time("run_real", self.time), _fork_rng(self.verify):
                flat_real_out = flatten((self._run(*cloned_args),))

        with _time("run_trace", self.time):
            flat_out = trace_info.closure()(*_varify(trace_inputs))
        if not isinstance(flat_out, tuple):
            flat_out = (flat_out,)

        if self.verify:
            _verify(flat_out, flat_real_out)

        return function._unflatten(flat_out, trace_info.proto)

    def record_trace(self, args, extra=()):
        is_volatile = any(arg.volatile if isinstance(arg, Variable) else False for arg in args)
        trace_inputs = self.get_trace_inputs(args, extra)

        trace = torch._C._tracer_enter(trace_inputs, 0 if is_volatile else self.num_derivatives)
        out = self._run(*args)
        torch._C._tracer_exit(flatten(out))

        return trace, out

    def has_trace_for(self, *args):
        trace_inputs = self.get_trace_inputs(args)
        trace_info = self.traces.get(self.get_input_key(trace_inputs))
        if trace_info is None:
            return False
        trace_info.check_traces()
        return trace_info.complete_trace is not None

    def __call__(self, *args):
        # Run the real thing if tracing is disabled
        if not self.enabled:
            with _time("run_real", self.time):
                return self._run(*args)

        trace_inputs = self.get_trace_inputs(args)
        input_key = self.get_input_key(trace_inputs)
        trace_info = self.traces[input_key]
        # Use the compiled closure if we have it already
        if trace_info.closure is not None:
            return self.run_closure(trace_info, args, trace_inputs)

        # Check if any of the traces in our pool are complete now
        trace_info.check_traces()
        if trace_info.complete_trace is not None:
            trace_info.compile_trace(self.optimize)
            return self.run_closure(trace_info, args, trace_inputs)

        # Otherwise, we have to collect a new trace
        trace, out = self.record_trace(args)
        trace_info.traces.append(trace)
        if trace_info.proto is None:
            trace_info.proto = function._to_proto(out)
        return out


def record_trace(traceable, *args, **kwargs):
    """
    Record a trace for a traceable object (either a function or a Module),
    returning a tuple (trace, output).  Positional arguments are passed
    as arguments to the model, while keyword arguments are used to control
    how we go about performing the trace.

    TODO: document kwargs
    """
    parameters = kwargs.pop('parameters', ())
    return Traceable(traceable, **kwargs).record_trace(args, extra=parameters)


def traced(traceable, **traced_kwargs):
    t = Traceable(traceable, **traced_kwargs)
    if isinstance(traceable, Module):
        traceable.forward = t
        return traceable
    else:
        return t


def trace(**kwargs):
    return lambda traceable: traced(traceable, **kwargs)


if not torch._C._jit_init():
    raise RuntimeError("JIT initialization failed")
