"""
torch.distributed provides an MPI-like interface for exchanging tensor
data across multi-machine networks. It supports a few different backends
and initialization methods.
"""
import torch
import warnings

_INITIALIZED_PG = 1
_INITIALIZED_MW = 2
_initialized = 0
_scope = locals()


def _extend_scope(module):
    _scope.update({k: getattr(module, k) for k in dir(module) if not k.startswith('_')})


def is_available():
    return torch._C._has_distributed()


def init_process_group(backend, init_method='env://', **kwargs):
    """Initializes the distributed package.

    Arguments:
        backend (str): Name of the backend to use. Depending on build-time configuration
            valid values include: ``tcp``, ``mpi`` and ``gloo``.
        init_method (str, optional): URL specifying how to initialize the package.
        world_size (int, optional): Number of processes participating in the job.
        rank (int, optional): Rank of the current process.
        group_name (str, optional): Group name. See description of init methods.
    """
    world_size = kwargs.pop('world_size', -1)
    group_name = kwargs.pop('group_name', '')
    rank = kwargs.pop('rank', -1)
    assert len(kwargs) == 0, "got unexpected keyword arguments: %s" % ",".join(kwargs.keys())

    if not is_available():
        raise RuntimeError("PyTorch built without distributed support")

    global _initialized
    if _initialized:
        raise RuntimeError("trying to initialize torch.distributed twice!")
    torch._C._dist_init_process_group(backend, init_method, world_size,
                                      group_name, rank)
    _initialized = _INITIALIZED_PG
    if not torch._C._dist_init_extension(False, reduce_op, group):
        raise RuntimeError("distributed module initialization failed")


def init_master_worker(backend, init_method='env://', **kwargs):
    warnings.warn("""
    ================================================================================
                                        WARNING
    ================================================================================
    Master-worker mode is still experimental. The API will change without
    notice and we're can't guarantee full correctness and expected performance yet.
    We'll announce it once it's ready.
    """)
    world_size = kwargs.pop('world_size', -1)
    group_name = kwargs.pop('group_name', '')
    rank = kwargs.pop('rank', -1)
    assert len(kwargs) == 0, "got unexpected keyword arguments: %s" % ",".join(kwargs.keys())

    if not is_available():
        raise RuntimeError("PyTorch built without distributed support")

    global _initialized
    if _initialized:
        raise RuntimeError("trying to initialize torch.distributed twice!")
    torch._C._dist_init_master_worker(backend, init_method, world_size,
                                      group_name, rank)
    _initialized = _INITIALIZED_MW
    import torch.distributed.collectives as collectives
    import torch.distributed.remote_types as remote_types
    _extend_scope(collectives)
    _extend_scope(remote_types)
    if not torch._C._dist_init_extension(True, reduce_op, group):
        raise RuntimeError("distributed module initialization failed")


class reduce_op(object):
    SUM = object()
    PRODUCT = object()
    MAX = object()
    MIN = object()


class group(object):
    WORLD = object()


class _DistributedRequest(object):
    def __init__(self, request):
        self.request = request

    def is_completed(self):
        return torch._C._dist_request_is_completed(self.request)

    def wait(self):
        torch._C._dist_request_wait(self.request)


def get_rank():
    """Returns the rank of current process.

    Rank is a unique identifier assigned to each process within a distributed
    group. They are always consecutive integers ranging from 0 to ``world_size``.
    """
    assert torch.distributed._initialized
    return torch._C._dist_get_rank()


def get_world_size():
    """Returns the number of processes in the distributed group."""
    assert torch.distributed._initialized
    return torch._C._dist_get_num_processes()


def isend(tensor, dst):
    """Sends a tensor asynchronously.

    Arguments:
        tensor (Tensor): Tensor to send.
        dst (int): Destination rank.

    Returns:
        A distributed request object.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return _DistributedRequest(torch._C._dist_isend(tensor, dst))


def irecv(tensor, src):
    """Receives a tensor asynchronously.

    Arguments:
        tensor (Tensor): Tensor to fill with received data.
        src (int): Source rank.

    Returns:
        A distributed request object.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return _DistributedRequest(torch._C._dist_irecv(tensor, src))


def send(tensor, dst):
    """Sends a tensor synchronously.

    Arguments:
        tensor (Tensor): Tensor to send.
        dst (int): Destination rank.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return torch._C._dist_send(tensor, dst)


def recv(tensor, src=None):
    """Receives a tensor synchronously.

    Arguments:
        tensor (Tensor): Tensor to fill with received data.
        src (int): Source rank.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    if src is None:
        return torch._C._dist_recv_any_source(tensor)
    return torch._C._dist_recv(tensor, src)


def broadcast(tensor, src, group=group.WORLD):
    """Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes
    participating in the collective.

    Arguments:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process, and tensor to be used to save received data otherwise.
        src (int): Source rank.
        group (optional): Group of the collective.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return torch._C._dist_broadcast(tensor, src, group)


def all_reduce(tensor, op=reduce_op.SUM, group=group.WORLD):
    """Reduces the tensor data across all machines in such a way that all get
    the final result.

    After the call ``tensor`` is going to be bitwise identical in all processes.

    Arguments:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        op (optional): One of the values from ``torch.distributed.reduce_op``
            enum.  Specifies an operation used for element-wise reductions.
        group (optional): Group of the collective.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return torch._C._dist_all_reduce(tensor, op, group)


def reduce(tensor, dst, op=reduce_op.SUM, group=group.WORLD):
    """Reduces the tensor data across all machines.

    Only the process with rank ``dst`` is going to receive the final result.

    Arguments:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        op (optional): One of the values from ``torch.distributed.reduce_op``
            enum.  Specifies an operation used for element-wise reductions.
        group (optional): Group of the collective.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return torch._C._dist_reduce(tensor, dst, op, group)


def all_gather(tensor_list, tensor, group=group.WORLD):
    """Gathers tensors from the whole group in a list.

    Arguments:
        tensor_list (list[Tensor]): Output list. It should contain
            correctly-sized tensors to be used for output of the collective.
        tensor (Tensor): Tensor to be broadcast from current process.
        group (optional): Group of the collective.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return torch._C._dist_all_gather(tensor_list, tensor, group)


def gather(tensor, **kwargs):
    """Gathers a list of tensors in a single process.

    Arguments:
        tensor (Tensor): Input tensor.
        dst (int): Destination rank. Required in all processes except the one that
            is receiveing the data.
        gather_list (list[Tensor]): List of appropriately-sized tensors to
            use for received data. Required only in the receiving process.
        group (optional): Group of the collective.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    my_rank = get_rank()
    dst = kwargs.pop('dst', my_rank)
    gather_list = kwargs.pop('gather_list', None)
    _group = kwargs.pop('group', group.WORLD)
    if kwargs:
        raise RuntimeError("got unexpected kwargs")
    if dst == my_rank:
        if gather_list is None:
            raise RuntimeError("gather_list is a required argument in gather destination")
        return torch._C._dist_gather_recv(gather_list, tensor, _group)
    else:
        if gather_list:
            raise RuntimeError("non-empty gather_list can be given only to gather destination")
        return torch._C._dist_gather_send(tensor, dst, _group)


def scatter(tensor, **kwargs):
    """Scatters a list of tensors to all processes in a group.

    Each process will receive exactly one tensor and store its data in the
    ``tensor`` argument.

    Arguments:
        tensor (Tensor): Output tensor.
        src (int): Source rank. Required in all processes except the one that
            is sending the data.
        scatter_list (list[Tensor]): List of tensors to scatter. Required only
            in the process that is sending the data.
        group (optional): Group of the collective.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    my_rank = get_rank()
    src = kwargs.pop('src', my_rank)
    scatter_list = kwargs.pop('scatter_list', None)
    _group = kwargs.pop('group', group.WORLD)
    if kwargs:
        raise RuntimeError("got unexpected kwargs")
    if src == my_rank:
        if scatter_list is None:
            raise RuntimeError("scatter_list is a required argument in scatter source")
        return torch._C._dist_scatter_send(scatter_list, tensor, _group)
    else:
        if scatter_list:
            raise RuntimeError("non-empty can be given only to scatter source")
        return torch._C._dist_scatter_recv(tensor, src, _group)


def barrier(group=group.WORLD):
    """Synchronizes all processes.

    This collective blocks processes until the whole group enters this function.

    Arguments:
        group (optional): Group of the collective.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return torch._C._dist_barrier(group)


def new_group(ranks=None):
    """Creates a new distributed group.

    This function requires that all processes in the main group (i.e. all
    processes that are part of the distributed job) enter this function, even
    if they are not going to be members of the group. Additionally, groups
    should be created in the same order in all processes.

    Arguments:
        ranks (list[int]): List of ranks of group members.

    Returns:
        A handle of distributed group that can be given to collective calls.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    if ranks is None:
        ranks = list(range(get_world_size()))
    return torch._C._dist_new_group(ranks)


def _register_stream(stream):
    if not _initialized:
        raise RuntimeError("torch.distributed needs to be initialized first")
    return torch._C._dist_register_stream(stream)
