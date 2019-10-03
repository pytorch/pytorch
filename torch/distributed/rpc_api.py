from . import invoke_rpc_builtin, invoke_rpc_python_udf, invoke_remote_builtin
from . import init_rref_context
from . import ProcessGroupAgent
from . import WorkerId
import torch.distributed.rpc as rpc
from torch.distributed.rpc.internal import _internal_rpc_pickler, PythonUDF

import functools
import sys
import torch
from enum import Enum


_agent = None


def _require_initialized(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if _agent is None:
            raise RuntimeError("RPC has not been initialized. "
                               "Call init_rpc(name) first.")
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


@_require_initialized
def sync_rpc():
    r"""
    Block until all local and remote RPC processes reach this method and finish
    sending all pending RPCs. As this method synchronizes at the process
    level, if multiple threads are spawned, only one of them should call this
    method at a time.
    """

    _agent.sync()

class RpcBackend(Enum):
    PROCESS_GROUP = 1


# TODO: add a context manager to wrap _init_rpc and join_rpc
def _init_rpc(backend=RpcBackend.PROCESS_GROUP,
              self_name=None,
              self_rank=-1,
              init_method=None,
              num_send_recv_threads=4):
    if sys.version_info < (3, 0):
        raise RuntimeError("RPC package does not support Python2.")

    global _agent

    if _agent:
        raise RuntimeError("RPC is already initialized")

    if backend == RpcBackend.PROCESS_GROUP:
        from .distributed_c10d import _get_default_group

        group = _get_default_group()
        if (self_rank != -1) and (self_rank != group.rank()):
            raise RuntimeError("self_rank argument {} doesn't match pg rank {}".format(
                               self_rank, group.rank()))
        # TODO: add try-except and destroy _agent in all processes if any fails.
        _agent = ProcessGroupAgent(self_name, group, num_send_recv_threads)
        init_rref_context(_agent)
    elif rpc.is_backend_registered(backend):
        _agent = rpc.init_backend(
            backend,
            self_rank=self_rank,
            self_name=self_name,
            init_method=init_method,
        )
        init_rref_context(_agent)
    else:
        raise RuntimeError("Unrecognized RPC backend ", backend)


@_require_initialized
def get_worker_id(worker_name=None):
    r"""
    Get worker id of a given worker name. Use this worker id to avoid passing
    an expensive string to ``rpc`` on every invocation.

    Arguments:
        worker_name (str): the string name of a worker. If ``None``, return the
                           the id of the current worker. (default ``None``)
    """
    if worker_name:
        return _agent.get_worker_id(worker_name)
    else:
        return _agent.get_worker_id()


def _to_worker_id(name_or_id):
    if isinstance(name_or_id, WorkerId):
        return name_or_id
    elif isinstance(name_or_id, str):
        return get_worker_id(name_or_id)
    else:
        raise ValueError("Unsupported RPC worker ID type {}".format(name_or_id))


@_require_initialized
def remote(to, func, args=None, kwargs=None):
    r"""
    Make a ``remote`` call to run ``func`` on worker ``to``, and returns an
    ``RRef`` to the result value immediately. Worker ``to`` will be the owner
    of the return ``RRef``, and this worker is a user. The owner manages the
    global reference count of its ``RRef``s, and the owner ``RRef`` is only
    destructed when globally there is no living references to it.

    Arguments:
        to (int or str): id or name of the destination worker.
        func (callable): builtin functions (like ``torch.add``).
        args (tuple): the argument tuple for the ``func`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``func``
                       invocation.

    Returns:
        A user ``RRef`` instance to the result value. Use the blocking API
        ``RRef.to_here()`` to retrieve the result value locally.

    Example::

        On worker 0:
        >>> import torch.distributed as dist
        >>> dist.init_process_group(backend='gloo', rank=0, world_size=2)
        >>> dist.init_rpc("worker0")
        >>> worker1 = dist.get_worker_id("worker1")
        >>> rref1 = dist.remote(worker1, torch.add, args=(torch.ones(2), 3))
        >>> rref2 = dist.remote(worker1, torch.add, args=(torch.ones(2), 1))
        >>> x = rref1.to_here() + rref2.to_here()
        >>> dist.join_rpc()

        On worker 1:
        >>> import torch.distributed as dist
        >>> dist.init_process_group(backend='gloo', rank=1, world_size=2)
        >>> dist.init_rpc("worker1")
        >>> dist.join_rpc()
    """
    qualified_name = torch.jit._find_builtin(func)

    args = args if args else ()
    kwargs = kwargs if kwargs else {}

    return invoke_remote_builtin(
        _agent, _to_worker_id(to), qualified_name, *args, **kwargs)


def _invoke_rpc(to, func, args=None, kwargs=None):
    if not callable(func):
        raise TypeError("function should be callable.")

    qualified_name = torch.jit._find_builtin(func)

    args = args if args else ()
    kwargs = kwargs if kwargs else {}

    if qualified_name is not None:
        fut = invoke_rpc_builtin(
            _agent, _to_worker_id(to), qualified_name, *args, **kwargs
        )
    else:
        (pickled_python_udf, tensors) = _internal_rpc_pickler.serialize(
            PythonUDF(func, args, kwargs))
        fut = invoke_rpc_python_udf(
            _agent, _to_worker_id(to), pickled_python_udf, tensors)
    return fut


@_require_initialized
def rpc_sync(to, func, args=None, kwargs=None):
    r"""
    Make a blocking RPC call to run function ``func`` on worker ``to``. RPC
    messages are sent and received in parallel to execution of Python code. This
    method is thread-safe.

    Arguments:
        to (int or str): id or name of the destination worker.
        func (callable): any callable function. builtin functions (like
                         ``torch.add``) can be sent over RPC more efficiently.
        args (tuple): the argument tuple for the ``func`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``func``
                       invocation.

    Returns:
        Returns the result of running ``func``on ``args`` and ``kwargs``.

    Example::
        On worker 0:
        >>> import torch.distributed as dist
        >>> dist.init_process_group(backend='gloo', rank=0, world_size=2)
        >>> dist.init_model_parallel("worker0")
        >>> ret = dist.rpc_sync("worker1", torch.add, args=(torch.ones(2), 3))
        >>> dist.join_rpc()

        On worker 1:
        >>> import torch.distributed as dist
        >>> dist.init_process_group(backend='gloo', rank=1, world_size=2)
        >>> dist.init_model_parallel("worker1")
        >>> dist.join_rpc()
    """
    fut = _invoke_rpc(to, func, args, kwargs)
    return fut.wait()


@_require_initialized
def rpc_async(to, func, args=None, kwargs=None):
    r"""
    Make a non-blocking RPC call to run function ``func`` on worker ``to``. RPC
    messages are sent and received in parallel to execution of Python code. This
    method is thread-safe. This method will immediately return a
    torch.distributed.FutureMessage that can be awaited on.

    Arguments:
        to (int or str): id or name of the destination worker.
        func (callable): any callable function. builtin functions (like
                         ``torch.add``) can be sent over RPC more efficiently.
        args (tuple): the argument tuple for the ``func`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``func``
                       invocation.

    Returns:
        Returns a ``torch.distributed.FutureMessage`` object that can be waited
        on. When completed, the return value of ``func`` on ``args`` and
        ``kwargs`` can be retrieved from the ``FutureMessage`` object.

    Example::

        On worker 0:
        >>> import torch.distributed as dist
        >>> dist.init_process_group(backend='gloo', rank=0, world_size=2)
        >>> dist.init_model_parallel("worker0")
        >>> worker1 = dist.get_worker_id("worker1")
        >>> fut1 = dist.rpc_async(worker1, torch.add, args=(torch.ones(2), 3))
        >>> fut2 = dist.rpc_async(worker1, min, args=(1, 2))
        >>> result = fut1.wait() + fut2.wait()
        >>> dist.join_rpc()

        On worker 1:
        >>> import torch.distributed as dist
        >>> dist.init_process_group(backend='gloo', rank=1, world_size=2)
        >>> dist.init_model_parallel("worker1")
        >>> dist.join_rpc()
    """
    fut = _invoke_rpc(to, func, args, kwargs)
    return fut
