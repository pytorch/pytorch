import collections
import contextlib
import functools
import inspect
import logging
import threading
from typing import Generic, TypeVar, Set, Any

import torch
from torch.futures import Future

from torch._C._distributed_rpc import (
    PyRRef,
    RemoteProfilerManager,
    WorkerInfo,
    get_rpc_timeout,
    _cleanup_python_rpc_handler,
    _delete_all_user_and_unforked_owner_rrefs,
    _destroy_rref_context,
    _get_current_rpc_agent,
    _invoke_remote_builtin,
    _invoke_remote_python_udf,
    _invoke_remote_torchscript,
    _invoke_rpc_builtin,
    _invoke_rpc_python_udf,
    _invoke_rpc_torchscript,
    _is_current_rpc_agent_set,
    _reset_current_rpc_agent,
    _set_and_start_rpc_agent,
)

from .internal import (
    PythonUDF,
    RPCExecMode,
    _internal_rpc_pickler,
    _build_rpc_profiling_key,
)

from .constants import DEFAULT_SHUTDOWN_TIMEOUT, UNSET_RPC_TIMEOUT


logger = logging.getLogger(__name__)


# NB: Ignoring RRef leaks during shutdown. Without this, applications have to
# make sure there is no references to any RRef in the application code and
# Python GC has done its job to delete those RRefs. This is could result in bad
# debugging experiences especially when for large applications. Therefore, by
# default, we are going to ignore RRef leaks during shutdown. This is usually
# fine as shutdown means applications have done training and no longer care
# about states.
#
# To enable RRef leak checking, set this _ignore_rref_leak to False
_ignore_rref_leak = True
_default_pickler = _internal_rpc_pickler


@contextlib.contextmanager
def _use_rpc_pickler(rpc_pickler):
    r"""
    rpc_pickler: (.internal._InternalRPCPickler) Overrides the default RPC pickler
    """
    global _default_pickler
    _default_pickler = rpc_pickler
    try:
        yield
    finally:
        _default_pickler = _internal_rpc_pickler


def _require_initialized(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _is_current_rpc_agent_set():
            raise RuntimeError(
                "RPC has not been initialized. Call "
                "torch.distributed.rpc.init_rpc first."
            )
        return func(*args, **kwargs)

    return wrapper


class AllGatherStates(object):
    def __init__(self):
        # Each `gathered_objects` is an empty dict at beginning.
        # The leader worker is elected as the first worker in a sorted worker
        # name list. Whenever there is a worker entering `_all_gather()`, it
        # runs `_gather_to_leader()` on the leader to add its own name and
        # data obj to this dict. The leader also adds itself's name to the dict
        # on calling `_all_gather()`.
        # Once `set(gathered_objects.keys()) == _ALL_WORKER_NAMES`, the leader
        # will broadcast the gathered dict to all follower workers and set their
        # `gathered_objects` field and the `proceed_signal` field.
        self.gathered_objects = {}
        # All workers wait on this signal until it receives all gathered
        # objects.
        self.proceed_signal = threading.Event()


# States used by `def _all_gather()`.
# `_ALL_WORKER_NAMES` is initialized on initiaizing RPC layer.
_ALL_WORKER_NAMES: Set[Any] = set()
_all_gather_dict_lock = threading.RLock()
_all_gather_sequence_id = 0
_all_gather_sequence_id_to_states: collections.defaultdict = collections.defaultdict(AllGatherStates)


def _init_rpc_states(agent):
    worker_infos = agent.get_worker_infos()
    global _ALL_WORKER_NAMES
    _ALL_WORKER_NAMES = {worker_info.name for worker_info in worker_infos}

    # NB: backend implementation might have already set the rpc_agent.
    if not _is_current_rpc_agent_set():
        _set_and_start_rpc_agent(agent)


def _gather_to_leader(sequence_id, worker_name, obj):
    with _all_gather_dict_lock:
        assert (
            worker_name in _ALL_WORKER_NAMES
        ), "{worker_name} is not expected by leader.".format(worker_name=worker_name)
        states = _all_gather_sequence_id_to_states[sequence_id]
        assert (
            worker_name not in states.gathered_objects
        ), "{worker_name} reported intent sequence id {sequence_id} twice. ".format(
            worker_name=worker_name, sequence_id=sequence_id
        )
        states.gathered_objects[worker_name] = obj
        if _ALL_WORKER_NAMES == set(states.gathered_objects.keys()):
            states.proceed_signal.set()


def _broadcast_to_followers(sequence_id, objects_map):
    with _all_gather_dict_lock:
        states = _all_gather_sequence_id_to_states[sequence_id]

    assert (
        not states.proceed_signal.is_set()
    ), "Termination signal sequence id {} got set twice.".format(sequence_id)
    states.gathered_objects = objects_map
    states.proceed_signal.set()

_thread_local_var = threading.local()

@contextlib.contextmanager
def _wait_all():
    r"""
    A context manager that collects all futures returned by ``rpc_async`` and
    waits them on the context manager's exit; relieving the user of needing
    to explicitly call wait.


    Example::
        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> with rpc._wait_all():
        >>>    fut_1 = rpc.rpc_async(dst, torch.add, (torch.ones(2, 2), 1))
        >>>    fut_2 = rpc.rpc_async(dst, torch.add, (torch.ones(2, 2), 1))
        >>> #fut_1 and fut_2 are waited on
    """
    _thread_local_var.future_list = []
    try:
        yield
    finally:
        try:
            torch.futures.wait_all(_thread_local_var.future_list)
        finally:
            del _thread_local_var.future_list

@_require_initialized
def _all_gather(obj, timeout=UNSET_RPC_TIMEOUT):
    r"""
    This is similar to torch.distributed.all_gather(), but is using RPC. It
    picks the worker with the smallest name (alphabetic order) as the leader.
    Then all followers send their data ``obj`` to the leader. After the leader
    has received all, it will broadcast the results back to all followers. This
    function blocks until all workers have received the gathered results.
    """
    assert (
        _ALL_WORKER_NAMES is not None
    ), "`_ALL_WORKER_NAMES` is not initialized for `def _all_gather`."
    leader_name = sorted(_ALL_WORKER_NAMES)[0]

    self_name = _get_current_rpc_agent().get_worker_info().name

    global _all_gather_sequence_id
    with _all_gather_dict_lock:
        sequence_id = _all_gather_sequence_id
        _all_gather_sequence_id += 1

    is_leader = leader_name == self_name
    if timeout == UNSET_RPC_TIMEOUT:
        timeout = get_rpc_timeout()

    # Phase 1: Followers send it's object to the leader
    if is_leader:
        _gather_to_leader(sequence_id, self_name, obj)
    else:
        rpc_sync(
            leader_name,
            _gather_to_leader,
            args=(sequence_id, self_name, obj),
            timeout=timeout,
        )

    with _all_gather_dict_lock:
        states = _all_gather_sequence_id_to_states[sequence_id]
    states.proceed_signal.wait()

    # Phase 2: Leader broadcast gathered results to all followers
    # Leader's signal is the first to be unblocked, after receiving all
    # followers' data objects.
    if is_leader:
        worker_name_to_response_future_dict = dict()
        for follower_name in _ALL_WORKER_NAMES - {leader_name}:
            fut = rpc_async(
                follower_name,
                _broadcast_to_followers,
                args=(sequence_id, states.gathered_objects),
                timeout=timeout
            )
            worker_name_to_response_future_dict[follower_name] = fut

        errors = []
        for follower_name, fut in worker_name_to_response_future_dict.items():
            try:
                fut.wait()
            except RuntimeError as ex:
                errors.append((follower_name, ex))

        if errors:
            raise RuntimeError(
                f"Followers {[e[0] for e in errors]} timed out in _all_gather "
                f"after {timeout:.2f} seconds. The first exception is {errors[0][1]}"
            )

    return states.gathered_objects


@_require_initialized
def _wait_all_workers():
    r"""
    Block until all local and remote RPC processes reach this method and wait
    for all outstanding work to complete. Every RPC process must call this
    method before exit to perform a graceful shutdown. This should be used to
    terminate the RPC framework, and there is no guarantee that the RPC
    framework will work after this method returns.
    """
    try:
        _all_gather(None, timeout=DEFAULT_SHUTDOWN_TIMEOUT)
    except RuntimeError as ex:
        logger.error(
            f"Failed to respond to 'Shutdown Proceed' in time, got error {ex}"
        )


@_require_initialized
def shutdown(graceful=True):
    r"""
    Perform a shutdown of the RPC agent, and then destroy the RPC agent. This
    stops the local agent from accepting outstanding requests, and shuts
    down the RPC framework by terminating all RPC threads. If ``graceful=True``,
    this will block until all local and remote RPC processes reach this method
    and wait for all outstanding work to complete. Otherwise, if
    ``graceful=False``, this is a local shutdown, and it does not wait for other
    RPC processes to reach this method.

    .. warning::
        For :class:`~torch.futures.Future` objects returned by
        :meth:`~torch.distributed.rpc.rpc_async`, ``future.wait()`` should not
        be called after ``shutdown()``.

    Args:
        graceful (bool): Whether to do a graceful shutdown or not. If True,
                         this will 1) wait until there is no pending system
                         messages for ``UserRRefs`` and delete them; 2) block
                         until all local and remote RPC processes have reached
                         this method and wait for all outstanding work to
                         complete.

    Example::
        Make sure that ``MASTER_ADDR`` and ``MASTER_PORT`` are set properly
        on both workers. Refer to :meth:`~torch.distributed.init_process_group`
        API for more details. For example,

        >>> export MASTER_ADDR=localhost
        >>> export MASTER_PORT=5678

        Then run the following code in two different processes:

        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> # do some work
        >>> result = rpc.rpc_sync("worker1", torch.add, args=(torch.ones(1), 1))
        >>> # ready to shutdown
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> # wait for worker 0 to finish work, and then shutdown.
        >>> rpc.shutdown()
    """
    if graceful:
        _wait_all_workers()
        _delete_all_user_and_unforked_owner_rrefs()
        _get_current_rpc_agent().join(shutdown=True)
    try:
        # This raises a `TORCH_CHECK()` exception on RRef leak detected.
        _destroy_rref_context(_ignore_rref_leak)
    finally:
        _get_current_rpc_agent().shutdown()
        # clean up python rpc handler in shutdown(), see comments in
        # PythonRpcHandler::cleanup(), call it in python API because the
        # cleanup() function has python dependency, it assumes python
        # interpreter exists.
        # No matter if RRef leak exception is raised, this clean-up code
        # must run to avoid destruction segfault in Python 3.5.
        #
        # future.wait() should not be called after shutdown().
        # pythonRpcHandler is cleaned up in shutdown(), after
        # shutdown(), python objects returned from rpc python call can not be
        # resolved.
        _cleanup_python_rpc_handler()
        _reset_current_rpc_agent()


@_require_initialized
def get_worker_info(worker_name=None):
    r"""
    Get :class:`~torch.distributed.rpc.WorkerInfo` of a given worker name.
    Use this :class:`~torch.distributed.rpc.WorkerInfo` to avoid passing an
    expensive string on every invocation.

    Args:
        worker_name (str): the string name of a worker. If ``None``, return the
                           the id of the current worker. (default ``None``)

    Returns:
        :class:`~torch.distributed.rpc.WorkerInfo` instance for the given
        ``worker_name`` or :class:`~torch.distributed.rpc.WorkerInfo` of the
        current worker if ``worker_name`` is ``None``.
    """
    if worker_name is not None:
        return _get_current_rpc_agent().get_worker_info(worker_name)
    else:
        return _get_current_rpc_agent().get_worker_info()


def _to_worker_info(to):
    if isinstance(to, WorkerInfo):
        return to
    elif isinstance(to, str) or isinstance(to, int):
        return get_worker_info(to)
    else:
        raise ValueError("Cannot get WorkerInfo from name {}".format(to))


def _rref_typeof_on_owner(rref, blocking=True):
    rref_type = type(rref.local_value())
    if blocking:
        return rref_type
    else:
        # Wrap result into a completed Future. This is so that if blocking=`False`
        # is specified, we return a future regardless of if this call is on user
        # or owner.
        future = Future[type]()
        future.set_result(rref_type)
        return future


def _rref_typeof_on_user(rref, timeout=UNSET_RPC_TIMEOUT, blocking=True):
    fut = rpc_async(
        rref.owner(),
        _rref_typeof_on_owner,
        args=(rref,),
        timeout=timeout
    )
    if blocking:
        return fut.wait()
    else:
        return fut



T = TypeVar("T")
GenericWithOneTypeVar = Generic[T]


try:
    # Combine the implementation class and the type class.
    class RRef(PyRRef, Generic[T]):
        pass
except TypeError:
    # TypeError: metaclass conflict: the metaclass of a derived class
    # must be a (non-strict) subclass of the metaclasses of all its bases
    # Mypy doesn't understand __class__ (mypy bug #4177)
    class RRefMeta(PyRRef.__class__, GenericWithOneTypeVar.__class__):  # type: ignore[name-defined, misc, valid-type]
        pass

    # Combine the implementation class and the type class.
    # Types for classes expecting a certain generic parameter (mypy bug #7791)
    class RRef(PyRRef, GenericWithOneTypeVar, metaclass=RRefMeta):  # type: ignore[misc, no-redef, valid-type]
        pass


# Install docstrings from `PyRRef` to `RRef`.
#
# This is for the fact that pybind11 generates the parameter
# `self` as type `rpc.PyRRef`, so a `:inherited-members:`
# under `.. autoclass:: RRef` does not work.
# we have to do the following process to replacee `rpc.PyRRef` with `rpc.RRef`.
#
def method_factory(method_name, docstring):
    def method(self, *args, **kwargs):
        return getattr(super(RRef, self), method_name)(*args, **kwargs)

    method.__doc__ = docstring
    return method


for method_name, method in inspect.getmembers(PyRRef):
    # Ignore magic methods, except "__str__".
    if method_name.startswith("_") and method_name != "__str__":
        continue

    # Get pybind11 generated docstring.
    # It's like,
    """
    to_here(self: torch.distributed.rpc.PyRRef, timeout: float=-1.0) -> object

        Blocking call that copies the value of the RRef from the owner
        to the local node and returns it. If the current node is the
        owner, returns a reference to the local value.
    """
    docstring = getattr(method, "__doc__", None)
    assert docstring is not None, "RRef user-facing methods should all have docstrings."

    # Do surgery on pybind11 generated docstrings.
    docstring = docstring.replace("torch.distributed.rpc.PyRRef", "torch.distributed.rpc.RRef")

    # Attach user-facing RRef method with modified docstring.
    new_method = method_factory(method_name, docstring)
    setattr(RRef, method_name, new_method)


@_require_initialized
def remote(to, func, args=None, kwargs=None, timeout=UNSET_RPC_TIMEOUT):
    r"""
    Make a remote call to run ``func`` on worker ``to`` and return an
    :class:`~torch.distributed.rpc.RRef` to the result value immediately.
    Worker ``to`` will be the owner of the returned
    :class:`~torch.distributed.rpc.RRef`, and the worker calling ``remote`` is
    a user. The owner manages the global reference count of its
    :class:`~torch.distributed.rpc.RRef`, and the owner
    :class:`~torch.distributed.rpc.RRef` is only destructed when globally there
    are no living references to it.

    Args:
        to (str or WorkerInfo or int): name/rank/``WorkerInfo`` of the destination worker.
        func (callable): a callable function, such as Python callables, builtin
                         operators (e.g. :meth:`~torch.add`) and annotated
                         TorchScript functions.
        args (tuple): the argument tuple for the ``func`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``func``
                       invocation.

        timeout (float, optional): timeout in seconds for this remote call. If the
                                   creation of this
                                   :class:`~torch.distributed.rpc.RRef` on worker
                                   ``to`` is not successfully processed on this
                                   worker within this timeout, then the next time
                                   there is an attempt to use the RRef (such as
                                   ``to_here()``), a timeout will be raised
                                   indicating this failure. A value of 0 indicates
                                   an infinite timeout, i.e. a timeout error will
                                   never be raised. If not provided, the default
                                   value set during initialization or with
                                   ``_set_rpc_timeout`` is used.

    Returns:
        A user :class:`~torch.distributed.rpc.RRef` instance to the result
        value. Use the blocking API :meth:`torch.distributed.rpc.RRef.to_here`
        to retrieve the result value locally.

    .. warning ::
        The ``remote`` API does not copy storages of argument tensors until
        sending them over the wire, which could be done by a different thread
        depending on the RPC backend type. The caller should make sure that the
        contents of those tensors stay intact until the returned RRef is
        confirmed by the owner, which can be checked using the
        :meth:`torch.distributed.rpc.RRef.confirmed_by_owner` API.

    .. warning ::
        Errors such as timeouts for the ``remote`` API are handled on a
        best-effort basis. This means that when remote calls initiated by
        ``remote`` fail, such as with a timeout error, we take a best-effort
        approach to error handling. This means that errors are handled and set
        on the resulting RRef on an asynchronous basis. If the RRef has not been
        used by the application before this handling (such as ``to_here`` or
        fork call), then future uses of the ``RRef`` will appropriately raise
        errors. However, it is possible that the user application will use the
        ``RRef`` before the errors are handled. In this case, errors may not be
        raised as they have not yet been handled.

    Example::
        Make sure that ``MASTER_ADDR`` and ``MASTER_PORT`` are set properly
        on both workers. Refer to :meth:`~torch.distributed.init_process_group`
        API for more details. For example,

        >>> export MASTER_ADDR=localhost
        >>> export MASTER_PORT=5678

        Then run the following code in two different processes:

        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> rref1 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 3))
        >>> rref2 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 1))
        >>> x = rref1.to_here() + rref2.to_here()
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()

        Below is an example of running a TorchScript function using RPC.

        >>> # On both workers:
        >>> @torch.jit.script
        >>> def my_script_add(t1, t2):
        >>>    return torch.add(t1, t2)

        >>> # On worker 0:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> rref = rpc.remote("worker1", my_script_add, args=(torch.ones(2), 3))
        >>> rref.to_here()
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()
    """
    qualified_name = torch.jit._builtins._find_builtin(func)
    dst_worker_info = _to_worker_info(to)
    should_profile = torch.autograd._profiler_enabled()

    ctx_manager = _enable_rpc_profiler(should_profile, qualified_name, func, RPCExecMode.REMOTE, dst_worker_info)

    with ctx_manager as rf:
        args = args if args else ()
        kwargs = kwargs if kwargs else {}

        is_async_exec = hasattr(func, "_wrapped_async_rpc_function")

        if is_async_exec:
            wrapped = func._wrapped_async_rpc_function
            if isinstance(wrapped, torch.jit.ScriptFunction):
                func = wrapped

        if qualified_name is not None:
            rref = _invoke_remote_builtin(dst_worker_info, qualified_name, timeout, *args, **kwargs)
        elif isinstance(func, torch.jit.ScriptFunction):
            rref = _invoke_remote_torchscript(
                dst_worker_info.name,
                torch._jit_internal._qualified_name(func),
                timeout,
                is_async_exec,
                *args,
                **kwargs,
            )
        else:
            (pickled_python_udf, tensors) = _default_pickler.serialize(
                PythonUDF(func, args, kwargs)
            )
            rref = _invoke_remote_python_udf(
                dst_worker_info,
                pickled_python_udf,
                tensors,
                timeout,
                is_async_exec
            )
        # attach profiling information
        if should_profile:
            assert torch.autograd._profiler_enabled()
            assert rf is not None
            fut = rf._call_end_callbacks_on_future(rref._get_future())
            rref._set_profiling_future(fut)

    return rref

def _invoke_rpc(to, func, rpc_type, args=None, kwargs=None, rpc_timeout=UNSET_RPC_TIMEOUT):
    if not callable(func):
        raise TypeError("function should be callable.")

    qualified_name = torch.jit._builtins._find_builtin(func)
    dst_worker_info = _to_worker_info(to)

    should_profile = torch.autograd._profiler_enabled()

    ctx_manager = _enable_rpc_profiler(should_profile, qualified_name, func, rpc_type, dst_worker_info)

    with ctx_manager as rf:
        args = args if args else ()
        kwargs = kwargs if kwargs else {}

        is_async_exec = hasattr(func, "_wrapped_async_rpc_function")

        if is_async_exec:
            wrapped = func._wrapped_async_rpc_function
            if isinstance(wrapped, torch.jit.ScriptFunction):
                func = wrapped

        if qualified_name is not None:
            fut = _invoke_rpc_builtin(
                dst_worker_info,
                qualified_name,
                rpc_timeout,
                *args,
                **kwargs
            )
        elif isinstance(func, torch.jit.ScriptFunction):
            fut = _invoke_rpc_torchscript(
                dst_worker_info.name,
                torch._jit_internal._qualified_name(func),
                args,
                kwargs,
                rpc_timeout,
                is_async_exec
            )
        else:
            (pickled_python_udf, tensors) = _default_pickler.serialize(
                PythonUDF(func, args, kwargs)
            )
            fut = _invoke_rpc_python_udf(
                dst_worker_info,
                pickled_python_udf,
                tensors,
                rpc_timeout,
                is_async_exec
            )
        if should_profile:
            assert torch.autograd._profiler_enabled()
            assert rf is not None
            # Schedule profiling callbacks to run when the future completes.
            # This returns a future that is completed when the original future
            # completes and the profiling callbacks have been completed as well,
            # to guarantee that fut.wait() completes the profiling. This new
            # future will contain the same value as the original future.
            fut = rf._call_end_callbacks_on_future(fut)
    return fut


@_require_initialized
def rpc_sync(to, func, args=None, kwargs=None, timeout=UNSET_RPC_TIMEOUT):
    r"""
    Make a blocking RPC call to run function ``func`` on worker ``to``. RPC
    messages are sent and received in parallel to execution of Python code. This
    method is thread-safe.

    Args:
        to (str or WorkerInfo or int): name/rank/``WorkerInfo`` of the destination worker.
        func (callable): a callable function, such as Python callables, builtin
                         operators (e.g. :meth:`~torch.add`) and annotated
                         TorchScript functions.
        args (tuple): the argument tuple for the ``func`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``func``
                       invocation.
        timeout (float, optional): timeout in seconds to use for this RPC. If
                                   the RPC does not complete in this amount of
                                   time, an exception indicating it has
                                   timed out will be raised. A value of 0
                                   indicates an infinite timeout, i.e. a timeout
                                   error will never be raised. If not provided,
                                   the default value set during initialization
                                   or with ``_set_rpc_timeout`` is used.

    Returns:
        Returns the result of running ``func`` with ``args`` and ``kwargs``.

    Example::
        Make sure that ``MASTER_ADDR`` and ``MASTER_PORT`` are set properly
        on both workers. Refer to :meth:`~torch.distributed.init_process_group`
        API for more details. For example,

        >>> export MASTER_ADDR=localhost
        >>> export MASTER_PORT=5678

        Then run the following code in two different processes:

        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> ret = rpc.rpc_sync("worker1", torch.add, args=(torch.ones(2), 3))
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()

        Below is an example of running a TorchScript function using RPC.

        >>> # On both workers:
        >>> @torch.jit.script
        >>> def my_script_add(t1, t2):
        >>>    return torch.add(t1, t2)

        >>> # On worker 0:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> ret = rpc.rpc_sync("worker1", my_script_add, args=(torch.ones(2), 3))
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()

    """
    fut = _invoke_rpc(to, func, RPCExecMode.SYNC, args, kwargs, timeout)
    return fut.wait()


@_require_initialized
def rpc_async(to, func, args=None, kwargs=None, timeout=UNSET_RPC_TIMEOUT):
    r"""
    Make a non-blocking RPC call to run function ``func`` on worker ``to``. RPC
    messages are sent and received in parallel to execution of Python code. This
    method is thread-safe. This method will immediately return a
    :class:`~torch.futures.Future` that can be awaited on.

    Args:
        to (str or WorkerInfo or int): name/rank/``WorkerInfo`` of the destination worker.
        func (callable): a callable function, such as Python callables, builtin
                         operators (e.g. :meth:`~torch.add`) and annotated
                         TorchScript functions.
        args (tuple): the argument tuple for the ``func`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``func``
                       invocation.
        timeout (float, optional): timeout in seconds to use for this RPC. If
                                   the RPC does not complete in this amount of
                                   time, an exception indicating it has
                                   timed out will be raised. A value of 0
                                   indicates an infinite timeout, i.e. a timeout
                                   error will never be raised. If not provided,
                                   the default value set during initialization
                                   or with ``_set_rpc_timeout`` is used.


    Returns:
        Returns a :class:`~torch.futures.Future` object that can be waited
        on. When completed, the return value of ``func`` on ``args`` and
        ``kwargs`` can be retrieved from the :class:`~torch.futures.Future`
        object.

    .. warning ::
        Using GPU tensors as arguments or return values of ``func`` is not
        supported since we don't support sending GPU tensors over the wire. You
        need to explicitly copy GPU tensors to CPU before using them as
        arguments or return values of ``func``.

    .. warning ::
        The ``rpc_async`` API does not copy storages of argument tensors until
        sending them over the wire, which could be done by a different thread
        depending on the RPC backend type. The caller should make sure that the
        contents of those tensors stay intact until the returned
        :class:`~torch.futures.Future` completes.

    Example::
        Make sure that ``MASTER_ADDR`` and ``MASTER_PORT`` are set properly
        on both workers. Refer to :meth:`~torch.distributed.init_process_group`
        API for more details. For example,

        >>> export MASTER_ADDR=localhost
        >>> export MASTER_PORT=5678

        Then run the following code in two different processes:

        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> fut1 = rpc.rpc_async("worker1", torch.add, args=(torch.ones(2), 3))
        >>> fut2 = rpc.rpc_async("worker1", min, args=(1, 2))
        >>> result = fut1.wait() + fut2.wait()
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()

        Below is an example of running a TorchScript function using RPC.

        >>> # On both workers:
        >>> @torch.jit.script
        >>> def my_script_add(t1, t2):
        >>>    return torch.add(t1, t2)

        >>> # On worker 0:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> fut = rpc.rpc_async("worker1", my_script_add, args=(torch.ones(2), 3))
        >>> ret = fut.wait()
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()
    """
    fut = _invoke_rpc(to, func, RPCExecMode.ASYNC, args, kwargs, timeout)
    if hasattr(_thread_local_var, "future_list"):
        _thread_local_var.future_list.append(fut)
    return fut


def _enable_rpc_profiler(should_profile, qualified_name, func, rpc_type, dst_worker_info):
    ctx_manager = contextlib.suppress()

    if should_profile:
        # Create appropriate string representation based on type of func
        # (builtin, script, python)
        if qualified_name is None:
            func_name = (
                torch._jit_internal._qualified_name(func)
                if isinstance(func, torch.jit.ScriptFunction)
                else func.__qualname__
            )
        else:
            func_name = qualified_name
        # Build RPC profiling key.
        rpc_profiling_key = _build_rpc_profiling_key(
            rpc_type,
            func_name,
            get_worker_info().name,
            dst_worker_info.name,
        )
        RemoteProfilerManager.set_current_profiling_key(rpc_profiling_key)
        # Mypy doesn't support re-def of a variable not in the same block (#1174)
        ctx_manager = torch.autograd.profiler.record_function(rpc_profiling_key)  # type: ignore[assignment]

    return ctx_manager
