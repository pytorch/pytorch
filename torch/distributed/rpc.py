from . import invoke_rpc
from . import ProcessGroupAgent

import array
import torch

_agent = None

def _collect_worker_names(name, group):
    from . import all_gather
    from . import get_world_size

    # collect name length
    ws = get_world_size(group)
    encoded_name = list(name.encode('utf-8'))
    name_len = len(encoded_name)
    len_input = torch.ones(1, dtype=torch.int64) * name_len
    len_outputs = [torch.empty(1, dtype=torch.int64) for _ in range(ws)]
    all_gather(len_outputs, len_input, group=group)

    # collect name value
    max_len = torch.stack(len_outputs).max().item()
    name_input = torch.empty(max_len, dtype=torch.uint8)
    name_input[:name_len] = torch.tensor(encoded_name, dtype=torch.uint8)
    name_outputs = [torch.empty(max_len, dtype=torch.uint8) for _ in range(ws)]
    all_gather(name_outputs, name_input, group=group)

    names = []
    for i in range(ws):
        name_tensor = name_outputs[i][:len_outputs[i]]
        names.append(array.array('B', name_tensor.tolist()).tobytes().decode('utf-8'))

    return names


def join_rpc():
    r"""
    Block until all local and remote RPC processes reach this method, process
    (send and receive) all pending messages, and then destroy local RPC agent.
    Every RPC process must call this method before exit.
    """
    global _agent
    _agent.join()
    _agent = None


def init_rpc(name, backend='pg'):
    r"""
    Initialize the local RPC agent which immediately becomes ready to make and
    accept RPCs after this method. The caller needs to make sure the specified
    backend is properly intialized before calling this method. For example, to
    use ``pg`` (ProcessGroup) backend, ``init_process_group`` must be invoked
    prior to this method.

    Arguments:
        name (str): a globally unique name of the local RPC agent. It is
                    encouraged to use names that conform application context.
                    (e.g., ``Trainer3``, ``ParameterServer2``, ``Master``,
                    ``Worker1``, etc.)
        backend (str): type of RPC backend implementation. Currently,
                       process group backend ``"pg"`` is the only available
                       backend implementation. (default: ``"pg"``).
    """
    global _agent
    if backend == 'pg':
        from . import is_initialized

        if not is_initialized():
            raise RuntimeError("Using pg RPC backend requires calling "
                               "init_process_group first.")

        from .distributed_c10d import _default_pg
        # TODO: issue #23232
        names = _collect_worker_names(name, _default_pg)
        name_dict = {names[r] : r for r in range(len(names))}
        _agent = ProcessGroupAgent(name, name_dict, _default_pg)
    else:
        raise RuntimeError("Unrecognized RPC backend ", backend)


def rpc_async(to, func, args=None, kwargs=None):
    r"""
    Asynchronous RPC. Make an RPC call to run function ``func`` on worker
    ``to``, and immediately returns a future object of the return value.

    Arguments:
        to (str): name of the destination worker.
        func (callable): a builtin function (e.g., ``torch.add``).
        args (tuple): the argument tuple for the ``func`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``func``
                       invocation.

    Returns:
        A Future object that can be waited on. When completed, the return value
        of ``func`` on ``args`` and ``kwargs`` can be retrieved from the Future
        object.
    """
    qualified_name = torch.jit._find_builtin(func)
    if qualified_name is None:
        raise RuntimeError("unknown builtin function %s." % func)

    args = args if args else ()
    kwargs = kwargs if kwargs else {}
    return invoke_rpc(_agent, to, qualified_name, *args, **kwargs)

def rpc_sync(to, func, args=None, kwargs=None):
    r"""
    Synchronous RPC. Make an RPC call to run function ``func`` on worker ``to``,
    and block until the return value is locally available.

    Arguments:
        to (str): name of the destination worker.
        func (callable): a builtin function (e.g., ``torch.add``).
        args (tuple): the argument tuple for the ``func`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``func``
                       invocation.

    Returns:
        The return value of ``func`` on ``args`` and ``kwargs``.

    Example::

        On worker 0:
        >>> import torch.distributed as dist
        >>> dist.init_process_group(backend='gloo', ...)
        >>> dist.init_rpc("worker0")
        >>> ret = dist.rpc_sync("worker1", torch.add, torch.ones(2, 2), 3)
        >>> dist.join_rpc()

        One worker 1:
        >>> import torch.distributed as dist
        >>> dist.init_process_group(backend='gloo', ...)
        >>> dist.init_rpc("worker1")
        >>> dist.join_rpc()
    """
    future = rpc_async(to, func, args=args, kwargs=kwargs)
    return future.wait()
