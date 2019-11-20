from . import _invoke_rpc_builtin, _invoke_rpc_python_udf
from . import _invoke_remote_builtin, _invoke_remote_python_udf
from . import _invoke_rpc_script
from . import _start_rpc_agent
from . import _destroy_rref_context, _cleanup_python_rpc_handler
from . import WorkerInfo
from . import backend_registry
from .internal import _internal_rpc_pickler, PythonUDF

import functools
import numbers
import sys
import torch
import torch.distributed as dist


_agent = None


def _require_initialized(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if _agent is None:
            raise RuntimeError(
                "RPC has not been initialized. Call "
                "torch.distributed.rpc.init_rpc first."
            )
        return func(*args, **kwargs)
    return wrapper


def join_rpc():
    r"""
    Block until all local and remote RPC processes reach this method, process
    (send and receive) all pending messages, and then destroy local RPC agent.
    Every RPC process must call this method before exit.
    """
    global _agent

    if _agent:
        _agent.join()
        _agent = None
        _destroy_rref_context()
        # clean up python rpc handler in join_rpc(), see comments in
        # PythonRpcHandler::cleanup(), call it in python API because the
        # cleanup() function has python dependency, it assumes python
        # interpreter exists
        _cleanup_python_rpc_handler()


# TODO: add a context manager to wrap _init_rpc_backend and join_rpc
def _init_rpc_backend(
    backend=backend_registry.BackendType.PROCESS_GROUP,
    store=None,
    name=None,
    rank=-1,
    world_size=-1,
    rpc_agent_options=None,
):
    from . import RpcAgentOptions

    if sys.version_info < (3, 0):
        raise RuntimeError("RPC package does not support Python2.")

    if not isinstance(backend, backend_registry.BackendType):
        raise RuntimeError("`backend` must be a `backend_registry.BackendType`.")

    if not isinstance(store, dist.Store):
        raise RuntimeError("`store` must be a `c10d::Store`. {}".format(store))

    if not isinstance(name, str):
        raise RuntimeError("`name` must be a string. {}".format(name))

    if not isinstance(rank, numbers.Integral):
        raise RuntimeError("`rank` must be an integer. {}".format(rank))

    if not isinstance(world_size, numbers.Integral):
        raise RuntimeError("`world_size` must be an integer. {}".format(world_size))

    if not isinstance(rpc_agent_options, RpcAgentOptions):
        raise RuntimeError(
            "`rpc_agent_options` must be an `RpcAgentOptions`. {}".format(
                rpc_agent_options
            )
        )

    global _agent

    if _agent:
        raise RuntimeError("RPC is already initialized")

    # Initialize RPC.
    _agent = backend_registry.init_backend(
        backend,
        store=store,
        name=name,
        rank=rank,
        world_size=world_size,
        rpc_agent_options=rpc_agent_options,
    )
    _start_rpc_agent(_agent)


@_require_initialized
def get_worker_info(worker_name=None):
    r"""
    Get :class:`~torch.distributed.rpc.WorkerInfo` of a given worker name.
    Use this :class:`~torch.distributed.rpc.WorkerInfo` to avoid passing an
    expensive string on every invocation.

    Arguments:
        worker_name (str): the string name of a worker. If ``None``, return the
                           the id of the current worker. (default ``None``)

    Returns:
        :class:`~torch.distributed.rpc.WorkerInfo` instance for the given
        ``worker_name`` or :class:`~torch.distributed.rpc.WorkerInfo` of the
        current worker if ``worker_name`` is ``None``.
    """
    if worker_name:
        return _agent.get_worker_info(worker_name)
    else:
        return _agent.get_worker_info()


def _to_worker_info(name_or_info):
    if isinstance(name_or_info, WorkerInfo):
        return name_or_info
    elif isinstance(name_or_info, str):
        return get_worker_info(name_or_info)
    else:
        raise ValueError("Cannot get WorkerInfo from name".format(name_or_info))


@_require_initialized
def remote(to, func, args=None, kwargs=None):
    r"""
    Make a remote call to run ``func`` on worker ``to`` and return an
    :class:`~torch.distributed.rpc.RRef` to the result value immediately.
    Worker ``to`` will be the owner of the returned
    :class:`~torch.distributed.rpc.RRef`, and the worker calling ``remote`` is
    a user. The owner manages the global reference count of its
    :class:`~torch.distributed.rpc.RRef`, and the owner
    :class:`~torch.distributed.rpc.RRef` is only destructed when globally there
    are no living references to it.

    Arguments:
        to (str or WorkerInfo): id or name of the destination worker.
        func (callable): builtin functions (like :meth:`torch.add`).
        args (tuple): the argument tuple for the ``func`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``func``
                       invocation.

    Returns:
        A user :class:`~torch.distributed.rpc.RRef` instance to the result
        value. Use the blocking API :meth:`torch.distributed.rpc.RRef.to_here`
        to retrieve the result value locally.

    Example::

        On worker 0:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> worker1 = rpc.get_worker_info("worker1")
        >>> rref1 = rpc.remote(worker1, torch.add, args=(torch.ones(2), 3))
        >>> rref2 = rpc.remote(worker1, torch.add, args=(torch.ones(2), 1))
        >>> x = rref1.to_here() + rref2.to_here()
        >>> rpc.join_rpc()

        On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=0, world_size=2)
        >>> rpc.join_rpc()
    """
    qualified_name = torch.jit._find_builtin(func)

    args = args if args else ()
    kwargs = kwargs if kwargs else {}

    info = _to_worker_info(to)
    if qualified_name is not None:
        return _invoke_remote_builtin(
            _agent, info, qualified_name, *args, **kwargs)
    else:
        (pickled_python_udf, tensors) = _internal_rpc_pickler.serialize(
            PythonUDF(func, args, kwargs))
        return _invoke_remote_python_udf(
            _agent, info, pickled_python_udf, tensors)


def _invoke_rpc(to, func, args=None, kwargs=None):
    if not callable(func):
        raise TypeError("function should be callable.")

    qualified_name = torch.jit._find_builtin(func)

    args = args if args else ()
    kwargs = kwargs if kwargs else {}

    info = _to_worker_info(to)
    if qualified_name is not None:
        fut = _invoke_rpc_builtin(
            _agent, info, qualified_name, *args, **kwargs
        )
    else:
        (pickled_python_udf, tensors) = _internal_rpc_pickler.serialize(
            PythonUDF(func, args, kwargs))
        fut = _invoke_rpc_python_udf(
            _agent, info, pickled_python_udf, tensors)
    return fut


@_require_initialized
def rpc_sync(to, func, args=None, kwargs=None):
    r"""
    Make a blocking RPC call to run function ``func`` on worker ``to``. RPC
    messages are sent and received in parallel to execution of Python code. This
    method is thread-safe.

    Arguments:
        to (str or WorkerInfo): id or name of the destination worker.
        func (callable): any callable function. builtin functions (like
                         :meth:`torch.add`) can be sent over RPC more efficiently.
        args (tuple): the argument tuple for the ``func`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``func``
                       invocation.

    Returns:
        Returns the result of running ``func`` on ``args`` and ``kwargs``.

    Example::

        On worker 0:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> ret = rpc.rpc_sync("worker1", torch.add, args=(torch.ones(2), 3))
        >>> rpc.join_rpc()

        On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=0, world_size=2)
        >>> rpc.join_rpc()
    """
    fut = _invoke_rpc(to, func, args, kwargs)
    return fut.wait()


@_require_initialized
def rpc_async(to, func, args=None, kwargs=None):
    r"""
    Make a non-blocking RPC call to run function ``func`` on worker ``to``. RPC
    messages are sent and received in parallel to execution of Python code. This
    method is thread-safe. This method will immediately return a
    Future<Message> that can be awaited on.

    Arguments:
        to (str or WorkerInfo): id or name of the destination worker.
        func (callable): any callable function. builtin functions (like
                         :meth:`torch.add`) can be sent over RPC more efficiently.
        args (tuple): the argument tuple for the ``func`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``func``
                       invocation.

    Returns:
        Returns a Future<Message> object that can be waited
        on. When completed, the return value of ``func`` on ``args`` and
        ``kwargs`` can be retrieved from the Future<Message> object.

    Example::

        On worker 0:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> worker1 = rpc.get_worker_id("worker1")
        >>> fut1 = rpc.rpc_async(worker1, torch.add, args=(torch.ones(2), 3))
        >>> fut2 = rpc.rpc_async(worker1, min, args=(1, 2))
        >>> result = fut1.wait() + fut2.wait()
        >>> rpc.join_rpc()

        On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=0, world_size=2)
        >>> rpc.join_rpc()
    """
    fut = _invoke_rpc(to, func, args, kwargs)
    return fut


# All below private APIs are for making rpc torch script call.
# Right now, rpc torch script call requires qualified_name to look up
# compliation unit function, so rpc torch script call APIs are different
# from rpc APIs for python call and built in operator call.
# These private APIs are subject to be merged with above public rpc APIs when
# jit supports torch script funciton to be a jit type.
@_require_initialized
def _rpc_sync(to, qualified_name, args=None, kwargs=None):
    r"""
    Make a blocking RPC call to run function ``func`` on worker ``to``. RPC
    messages are sent and received in parallel to execution of Python code. This
    method is thread-safe.

    Arguments:
        to (str): name of the destination worker.
        qualified_name (str): python function annotated with @torch.jit.script
                              (like ``moduleName::torchScriptFuncName``)
                              can be sent over RPC more efficiently.
        args (tuple): the argument tuple for the ``func`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``func``
                       invocation.

    Returns:
        Returns the result of running ``func``on ``args`` and ``kwargs``.

    Example::
        On worker 0:
        >>> import torch.distributed as dist
        >>> import torch.distributed.rpc as rpc
        >>> dist.init_process_group(backend='gloo', rank=0, world_size=2)
        >>> rpc.init_model_parallel("worker0")
        >>> ret = rpc.rpc_sync("worker1", "myModule::myTorchScriptFunc", args=(torch.ones(2), 3))
        >>> rpc.join_rpc()

        On worker 1:
        >>> import torch.distributed as dist
        >>> import torch.distributed.rpc as rpc
        >>> dist.init_process_group(backend='gloo', rank=1, world_size=2)
        >>> rpc.init_model_parallel("worker1")
        >>> rpc.join_rpc()
    """
    args = args if args else ()
    kwargs = kwargs if kwargs else {}
    fut = _invoke_rpc_script(to, qualified_name, *args, **kwargs)
    return fut.wait()


@_require_initialized
def _rpc_async(to, qualified_name, args=None, kwargs=None):
    r"""
    Make a non-blocking RPC call to run function ``func`` on worker ``to``. RPC
    messages are sent and received in parallel to execution of Python code. This
    method is thread-safe. This method will immediately return a
    Future<Message> that can be awaited on.

    Arguments:
        to (str): name of the destination worker.
        qualified_name (str): python function annotated with @torch.jit.script
                              (like ``moduleName::torchScriptFuncName``)
                              can be sent over RPC more efficiently.
        args (tuple): the argument tuple for the ``func`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``func``
                       invocation.

    Returns:
        Returns a PythonFutureWrapper object that can be waited
        on. When completed, the return value of ``func`` on ``args`` and
        ``kwargs`` can be retrieved from the PythonFutureWrapper object.

    Example::

        On worker 0:
        >>> import torch.distributed as dist
        >>> import torch.distributed.rpc as rpc
        >>> dist.init_process_group(backend='gloo', rank=0, world_size=2)
        >>> rpc.init_model_parallel("worker0")
        >>> worker1 = rpc.get_worker_id("worker1")
        >>> fut1 = rpc.rpc_async(worker1, "myModule::myTorchScriptFunc", args=(torch.ones(2), 3))
        >>> fut2 = rpc.rpc_async(worker1, "myModule::myTorchScriptFunc", args=(1, 2))
        >>> result = fut1.wait() + fut2.wait()
        >>> rpc.join_rpc()

        On worker 1:
        >>> import torch.distributed as dist
        >>> import torch.distributed.rpc as rpc
        >>> dist.init_process_group(backend='gloo', rank=1, world_size=2)
        >>> rpc.init_model_parallel("worker1")
        >>> rpc.join_rpc()
    """
    args = args if args else ()
    kwargs = kwargs if kwargs else {}
    fut = _invoke_rpc_script(to, qualified_name, *args, **kwargs)
    return fut
