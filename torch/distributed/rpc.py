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
    name_len = len(name)
    len_input = torch.ones(1, dtype=torch.int64) * name_len
    len_outputs = [torch.empty(1, dtype=torch.int64) for _ in range(ws)]
    all_gather(len_outputs, len_input, group=group)

    # collect name value
    max_len = torch.stack(len_outputs).max().item()
    name_input = torch.empty(max_len, dtype=torch.uint8)
    name_input[:name_len] = torch.Tensor(list(name.encode('utf-8')))
    name_outputs = [torch.empty(max_len, dtype=torch.uint8) for _ in range(ws)]
    all_gather(name_outputs, name_input, group=group)

    names = []
    for i in range(ws):
        name_tensor = name_outputs[i][:len_outputs[i]]
        names.append(array.array('B', name_tensor.tolist()).tobytes().decode('utf-8'))

    return names


def init_rpc(name, backend='pg'):
    r"""
    Arguments:
        name (str): name of this worker.
        backend (str): type of RPC backend implementation.
    """
    global _agent
    if backend == 'pg':
        from . import is_initialized

        assert is_initialized(), (
            "Using pg RPC backend requires calling init_process_group first."
        )

        from .distributed_c10d import _default_pg
        # TODO: move this to ProcessGroupAgent constructor
        names = _collect_worker_names(name, _default_pg)
        name_dict = {names[r] : r for r in range(len(names))}
        _agent = ProcessGroupAgent(name, name_dict, _default_pg)
    else:
        raise RuntimeError("Unrecognized RPC backend ", backend)

def destroy_rpc():
    _agent.shutdown()

def rpc_async(to, op, *args, **kargs):
    r"""
    Asynchronized RPC.

    Arguments:
        to (str): name of the destination worker
        op (str): qualified name of the builtin operator (e.g., "aten::add").

    Returns:
        A Future object that can be wait on. When complete, the return value of
        ``op`` on ``args`` and ``kargs`` can be retrieved from the Future
        object. Note that, the return value can only be retrieved once.
    """
    global _agent
    return invoke_rpc(_agent, to, op, *args, **kargs)

def rpc_sync(to, op, *args, **kargs):
    r"""
    Synchronized RPC.

    Arguments:
        to (str): name of the destination worker
        op (str): qualified name of the builtin operator (e.g., "aten::add").

    Returns:
        The return value of ``op`` on ``args`` and ``kargs``.
    """
    future = rpc_async(to, op, *args, **kargs)
    future.wait()
    return future.get()
