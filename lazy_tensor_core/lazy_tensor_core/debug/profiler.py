import lazy_tensor_core
import lazy_tensor_core.core.lazy_model as ltm
from typing import Optional

_TRACER_MARKED_STEP: bool = False


def set_tracer_marked_step(value: bool):
    global _TRACER_MARKED_STEP
    _TRACER_MARKED_STEP = value


def get_tracer_marked_step() -> bool:
    return _TRACER_MARKED_STEP


def start_server(port: int, only_on_master: Optional[bool] = True) -> object:
    """Start a profiler server on the client side on provided port.

    Users can then use the tensorboard profiler plugin
    (https://github.com/tensorflow/profiler) or the
    :func:`~lazy_tensor_core.debug.profiler.trace` as the client to request
    a profiler from this server.

    Args:
      port (int): the port to start the profiler server on. An exception is
        raised if the provided port is invalid or busy.
      only_on_master (optional(bool)): whether to only startup server from
        local master ordinal.
    Returns:
      A `ProfilerServer` instance that dictates the lifecycle of the profiler
      server. If this object is garbage collected, the profiler server is
      shut down.
    Raises:
      RuntimeError: Raised if the port is invalid or busy already.
    """
    if not only_on_master or ltm.is_master_ordinal():
        return lazy_tensor_core._LAZYC.profiler.start_server(port)


def trace(service_addr: str,
          logdir: str,
          duration_ms: Optional[int] = 1000,
          num_tracing_attempts: Optional[int] = 3,
          host_tracer_level: Optional[int] = 2,
          device_tracer_level: Optional[int] = 1,
          delay_ms: Optional[int] = 0):
    """Performs an on-demand profiling session on provided profiler servers.

    This method will block until it's done with profiling. Both single and
    multi-host profiling is supported. The output of the profiling requests
    are stored in the logdir specified.

    NOTE(b/177595210): 2VM TPU setup + profiler isn't currently supported
    so both the client VM and TPU cannot be profiled concurrently. Ex.
    service_addr = "localhost:9012,10.0.0.2:8466" does not currently work.

    Args:
      service_addr (str): comma delimited string of addresses of the profiling
        servers to profile. ex. "10.0.0.2:8466" or "localhost:9012".
      logdir (str): the path to write profiling output to. Both the profiler
        client and server must have access. ex. "gs://bucket/file/path".
      duration_ms (int): duration in milliseconds for tracing the server.
      num_tracing_attempts (int): number of trials to send profiling request
        in case of failures.
      host_tracer_level (int): CPU tracing level. Values are: 1 - critical info
        only, 2 - info, 3 - verbose.
        device_tracer_level (int): Device (TPU/GPU) tracing level. Values are: 1 -
        enabled, 0 - disabled.
      delay_ms (int): Specifies the services to start profiling delay_ms
        milliseconds after the current time.
    """
    options = {
        'host_tracer_level': host_tracer_level,
        'device_tracer_level': device_tracer_level,
        'delay_ms': delay_ms,
    }
    lazy_tensor_core._LAZYC.profiler.trace(
        service_addr,
        logdir,
        duration_ms=duration_ms,
        num_tracing_attempts=num_tracing_attempts,
        options=options)


class Trace(lazy_tensor_core._LAZYC.profiler.TraceMe):
    """Context manager that produces a trace event for profiling.

    The traces generated can then be collected using the above profiling APIs.
    The profiling server first needs to be started up and then can be sampled
    either using Tensorboard profiler plugin
    (https://github.com/tensorflow/profiler) or the
    :func:`~lazy_tensor_core.debug.profiler.trace` method.

    Note: currently only supports PyTorch client side trace events. i.e.,
    the namespace won't group TPU worker side trace.

    Example usage:
    ```python
    server = xp.start_server(9012)

    with xp.Trace('fwd_context'):
      model(input)
      ltm.mark_step()
    ```
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        super().__init__(name, **kwargs)

    def __enter__(self):
        self.scope = lazy_tensor_core._LAZYC.profiler.scope_pusher(self.name)
        super().__enter__()

    def __exit__(self, type, value, traceback):
        if getattr(self, 'scope', None):
            del self.scope
        super().__exit__(type, value, traceback)


class StepTrace(Trace):
    """Context manager that produces a step trace event for profiling.

    In addition to being regular traces, the generated traces will
    help provide per-step performance statistics.

    Note: currently only supports PyTorch client side trace events. i.e.,
    the namespace won't group TPU worker side trace.

    Example usage:
    ```python
    server = xp.start_server(9012)

    for step, (input, label) in enumerate(loader):
      with xp.StepTrace('train_step', step_num=step):
        model(input)
        ...
    ```
    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name, _r=1, **kwargs)

    def __enter__(self):
        set_tracer_marked_step(True)
        super().__enter__()

    def __exit__(self, type, value, traceback):
        if getattr(self, 'scope', None):
            # In ir.cpp ResetScopeContext we ensure that we have no remaining scope
            # before marking step.
            del self.scope
        ltm.mark_step()
        super().__exit__(type, value, traceback)
