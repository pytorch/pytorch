import pickle
import torch
import warnings
from torch._six import string_classes
from datetime import timedelta

# This module is wildcard imported from torch.distributed.
# TODO: specify __all__

from .constants import default_pg_timeout
from .rendezvous import rendezvous, register_rendezvous_handler  # noqa: F401
from . import (
    AllreduceOptions,
    AllreduceCoalescedOptions,
    AllToAllOptions,
    BroadcastOptions,
    GatherOptions,
    ReduceOptions,
    ReduceScatterOptions,
    ScatterOptions,
)
from . import ReduceOp
from . import PrefixStore


_MPI_AVAILABLE = True
_NCCL_AVAILABLE = True
_GLOO_AVAILABLE = True


try:
    from. import ProcessGroupMPI
except ImportError:
    _MPI_AVAILABLE = False

try:
    from. import ProcessGroupNCCL
except ImportError:
    _NCCL_AVAILABLE = False

try:
    from. import ProcessGroupGloo
except ImportError:
    _GLOO_AVAILABLE = False


class Backend(object):
    """
    An enum-like class of available backends: GLOO, NCCL, MPI, and other registered
    backends.

    The values of this class are lowercase strings, e.g., ``"gloo"``. They can
    be accessed as attributes, e.g., ``Backend.NCCL``.

    This class can be directly called to parse the string, e.g.,
    ``Backend(backend_str)`` will check if ``backend_str`` is valid, and
    return the parsed lowercase string if so. It also accepts uppercase strings,
    e.g., ``Backend("GLOO")`` returns ``"gloo"``.

    .. note:: The entry ``Backend.UNDEFINED`` is present but only used as
              initial value of some fields. Users should neither use it directly
              nor assume its existence.
    """
    UNDEFINED = "undefined"
    GLOO = "gloo"
    NCCL = "nccl"
    MPI = "mpi"
    TCP = "tcp"

    def __new__(cls, name):
        if not isinstance(name, string_classes):
            raise ValueError("Backend name must be a string, but got: {}".format(name))
        value = getattr(Backend, name.upper(), Backend.UNDEFINED)

        if value == Backend.TCP:
            raise ValueError("TCP backend has been deprecated. Please use "
                             "Gloo or MPI backend for collective operations "
                             "on CPU tensors.")
        elif value == Backend.UNDEFINED:
            raise ValueError("Invalid backend: '{}'".format(name))
        elif value != Backend.GLOO and value != Backend.NCCL and value != Backend.MPI:
            value = name
        return value

    @classmethod
    def register_backend(cls, name, func):
        """
        Registers a new backend.

        This class method is used by 3rd party cpp extension to register new backend.

        Arguments:
            name (str): Backend name matching with the one in `init_process_group()`.
            func (function): Function handler that instantiates the backend.
                             The function should be implemented in the backend cpp extension
                             and takes four arguments, including prefix_store, rank,
                             world_size, and timeout.

        .. note:: This support of 3rd party backend is experimental and subject to change.

        """
        setattr(Backend, name.upper(), func)

# `_backend`, `dist_backend`, and `reduce_op` are here to maintain backward
# compatibility with pre-c10d distributed package.
# TODO: remove them when users are ready to take a hard dependency on PyTorch 1.
_backend = Backend.UNDEFINED
dist_backend = Backend


class reduce_op(object):
    r"""
    Deprecated enum-like class for reduction operations: ``SUM``, ``PRODUCT``,
    ``MIN``, and ``MAX``.

    :class:`~torch.distributed.ReduceOp` is recommended to use instead.
    """

    def __init__(self):
        # __members__ is a dict storing key-value pairs for enum classes
        for k, v in ReduceOp.__members__.items():
            setattr(self, k, v)
        self.__members__ = ReduceOp.__members__

    def __getattribute__(self, key):
        warnings.warn("torch.distributed.reduce_op is deprecated, please use "
                      "torch.distributed.ReduceOp instead")
        return object.__getattribute__(self, key)

reduce_op = reduce_op()


class group(object):
    WORLD = object()


class GroupMember(object):
    # Alias to group.WORLD for backward compatibility
    WORLD = group.WORLD
    NON_GROUP_MEMBER = object()


# Cached process groups
# For NCCL and GLOO pg, it is a map from ProcessGroup to (Backend, Store)
# For MPI pg, it is a map from ProcessGroup to (Backend, None)
_pg_map = {}
# Process group's names, map from ProcessGroup to str
_pg_names = {}
# Process group's global rank to local rank mapping
_pg_group_ranks = {}

# Default process group state
_default_pg = None
_default_pg_init_method = None

# Process group count for default naming
_group_count = 0


def _rank_not_in_group(group):
    """
    Helper that checks if the current process's rank is not in a given group

    """
    if group == GroupMember.WORLD:
        return False
    return group == GroupMember.NON_GROUP_MEMBER


def _get_group_rank(group, rank):
    """
    Helper that gets a given group's local rank in the group from a given global
    rank

    """
    if group is GroupMember.WORLD:
        raise RuntimeError("group.WORLD does not have local rank to global "
                           "rank mapping")
    if group not in _pg_group_ranks:
        raise RuntimeError("The given group does not exist")
    try:
        group_rank = _pg_group_ranks[group][rank]
    except KeyError:
        raise RuntimeError(f"The global rank {rank} is not part of the group {group}") from None
    return group_rank


def _get_global_rank(group, group_rank):
    """
    Helper that gets a given group's global rank from a given local rank in the
    group

    """
    if group is GroupMember.WORLD:
        raise RuntimeError("group.WORLD does not have local rank to global "
                           "rank mapping")
    group_rank_map = _pg_group_ranks[group]
    for rank, grp_rank in group_rank_map.items():
        if grp_rank == group_rank:
            return rank
    raise RuntimeError("The group rank is not part of the group")


def _check_default_pg():
    """
    Helper that checks if the default ProcessGroup has been initialized, with
    assertion

    """
    assert _default_pg is not None, \
        "Default process group is not initialized"


def _get_group_size(group):
    """
    Helper that gets a given group's world size

    """
    if group is GroupMember.WORLD:
        _check_default_pg()
        return _default_pg.size()
    if group not in _pg_group_ranks:
        raise RuntimeError("The given group does not exist")
    return len(_pg_group_ranks[group])


def _check_single_tensor(param, param_name):
    """
    Helper to check that the parameter ``param_name`` is a single tensor.

    """
    if not isinstance(param, torch.Tensor):
        raise RuntimeError("Invalid function argument. Expected parameter `{}` "
                           "to be of type torch.Tensor.".format(param_name))


def _check_tensor_list(param, param_name):
    """
    Helper to check that the parameter ``param_name`` is a list of tensors.

    """
    if not isinstance(param, list) or \
       not all(isinstance(p, torch.Tensor) for p in param):
        raise RuntimeError("Invalid function argument. Expected parameter `{}` "
                           "to be of type List[torch.Tensor].".format(param_name))


def is_mpi_available():
    """
    Checks if the MPI backend is available.

    """
    return _MPI_AVAILABLE


def is_nccl_available():
    """
    Checks if the NCCL backend is available.

    """
    return _NCCL_AVAILABLE


def is_gloo_available():
    """
    Checks if the Gloo backend is available.

    """
    return _GLOO_AVAILABLE


def is_initialized():
    """
    Checking if the default process group has been initialized

    """
    return _default_pg is not None


def _get_default_group():
    """
    Getting the default process group created by init_process_group

    """
    if not is_initialized():
        raise RuntimeError("Default process group has not been initialized, "
                           "please make sure to call init_process_group.")
    return _default_pg


def _get_default_store():
    """
    Getting the default store created by init_process_group

    """
    if not is_initialized():
        raise RuntimeError("Default process group has not been initialized, "
                           "please make sure to call init_process_group.")
    _, default_store = _pg_map[_default_pg]
    return default_store


def get_backend(group=group.WORLD):
    """
    Returns the backend of the given process group.

    Arguments:
        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific group
            is specified, the calling process must be part of :attr:`group`.

    Returns:
        The backend of the given process group as a lower case string.

    """
    _check_default_pg()

    if group == GroupMember.WORLD:
        pg = _default_pg
    else:
        pg = group
    if _rank_not_in_group(pg):
        raise RuntimeError("Invalid process group specified")
    return _pg_map.get(pg, None)[0]


def init_process_group(backend,
                       init_method=None,
                       timeout=default_pg_timeout,
                       world_size=-1,
                       rank=-1,
                       store=None,
                       group_name=''):
    """
    Initializes the default distributed process group, and this will also
    initialize the distributed package.

    There are 2 main ways to initialize a process group:
        1. Specify ``store``, ``rank``, and ``world_size`` explicitly.
        2. Specify ``init_method`` (a URL string) which indicates where/how
           to discover peers. Optionally specify ``rank`` and ``world_size``,
           or encode all required parameters in the URL and omit them.

    If neither is specified, ``init_method`` is assumed to be "env://".


    Arguments:
        backend (str or Backend): The backend to use. Depending on
            build-time configurations, valid values include ``mpi``, ``gloo``,
            and ``nccl``. This field should be given as a lowercase string
            (e.g., ``"gloo"``), which can also be accessed via
            :class:`Backend` attributes (e.g., ``Backend.GLOO``). If using
            multiple processes per machine with ``nccl`` backend, each process
            must have exclusive access to every GPU it uses, as sharing GPUs
            between processes can result in deadlocks.
        init_method (str, optional): URL specifying how to initialize the
                                     process group. Default is "env://" if no
                                     ``init_method`` or ``store`` is specified.
                                     Mutually exclusive with ``store``.
        world_size (int, optional): Number of processes participating in
                                    the job. Required if ``store`` is specified.
        rank (int, optional): Rank of the current process.
                              Required if ``store`` is specified.
        store(Store, optional): Key/value store accessible to all workers, used
                                to exchange connection/address information.
                                Mutually exclusive with ``init_method``.
        timeout (timedelta, optional): Timeout for operations executed against
            the process group. Default value equals 30 minutes.
            This is applicable for the ``gloo`` backend. For ``nccl``, this is
            applicable only if the environment variable ``NCCL_BLOCKING_WAIT``
            or ``NCCL_ASYNC_ERROR_HANDLING`` is set to 1. When
            ``NCCL_BLOCKING_WAIT`` is set, this is the duration for which the
            process will block and wait for collectives to complete before
            throwing an exception. When ``NCCL_ASYNC_ERROR_HANDLING`` is set,
            this is the duration after which collectives will be aborted
            asynchronously and the process will crash. ``NCCL_BLOCKING_WAIT``
            will provide errors to the user which can be caught and handled,
            but due to its blocking nature, it has a performance overhead. On
            the other hand, ``NCCL_ASYNC_ERROR_HANDLING`` has little
            performance overhead, but crashes the process on errors. This is
            done since CUDA execution is async and it is no longer safe to
            continue executing user code since failed async NCCL operations
            might result in subsequent CUDA operations to run on corrupted
            data. Only one of these two environment variables should be set.
        group_name (str, optional, deprecated): Group name.

    To enable ``backend == Backend.MPI``, PyTorch needs to be built from source
    on a system that supports MPI.

    """
    global _pg_group_ranks
    global _backend
    global _default_pg
    global _default_pg_init_method

    if not isinstance(timeout, timedelta):
        raise RuntimeError("Expected timeout argument to be of type"
                           "datetime.timedelta")

    if _default_pg is not None:
        raise RuntimeError("trying to initialize the default process group "
                           "twice!")

    assert (store is None) or (init_method is None), \
        "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, 'world_size must be positive if using store'
        assert rank >= 0, 'rank must be non-negative if using store'
    elif init_method is None:
        init_method = "env://"

    backend = Backend(backend)

    if backend == Backend.MPI:
        if world_size != -1 or rank != -1:
            warnings.warn(
                "For MPI backend, world_size ({}) and rank ({}) "
                "are ignored since they are assigned by the "
                "MPI runtime.".format(world_size, rank))

        _default_pg = _new_process_group_helper(
            -1,
            -1,
            [],
            Backend.MPI,
            None,
            group_name=group_name,
            timeout=timeout)
    else:
        # backward compatible API
        if store is None:
            rendezvous_iterator = rendezvous(
                init_method, rank, world_size, timeout=timeout
            )
            store, rank, world_size = next(rendezvous_iterator)
            store.set_timeout(timeout)

        _default_pg = _new_process_group_helper(
            world_size,
            rank,
            [],
            backend,
            store,
            group_name=group_name,
            timeout=timeout)

    _pg_group_ranks[_default_pg] = {i: i for i in range(_default_pg.size())}
    _backend = _pg_map[_default_pg][0]
    _default_pg_init_method = init_method

    # barrier at the end to ensure that once we return from this method, all
    # process groups including global variables are updated correctly on all
    # ranks.
    barrier()

def _new_process_group_helper(world_size,
                              rank,
                              group_ranks,
                              backend,
                              store,
                              group_name=None,
                              timeout=default_pg_timeout):
    """
    Create a new distributed process group.

    This function must be called by ALL processes in the global group, even if
    the calling process is not part of the newly created group. In that case,
    this function returns GroupMember.NON_GROUP_MEMBER.

    This function is called with ``group_ranks == []`` for the default group.
    """
    global _pg_map
    global _group_count
    global _pg_names

    if not group_name:
        group_name = str(_group_count)
        _group_count += 1

    if group_name in _pg_names.values():
        raise RuntimeError("The specified group name has already been "
                           "created, please use a different group name")

    if not isinstance(timeout, timedelta):
        raise RuntimeError("Expected timeout argument to be of type"
                           "datetime.timedelta")

    # The list of group ranks is empty if we're creating the default group.
    is_default_group = (len(group_ranks) == 0)

    backend = Backend(backend)
    if backend == Backend.MPI:
        if not is_mpi_available():
            raise RuntimeError(
                "Distributed package doesn't have MPI built in."
                " MPI is only included if you build PyTorch from"
                " source on a host that has MPI installed.")
        pg = ProcessGroupMPI.create(group_ranks)
        if not pg:
            return GroupMember.NON_GROUP_MEMBER
        _pg_map[pg] = (Backend.MPI, None)
        _pg_names[pg] = group_name
    else:
        # If this is a subgroup (which means group_ranks is specified),
        # we check if the current process is a member of the new group.
        if not is_default_group:
            global_rank = _default_pg.rank()
            if global_rank not in group_ranks:
                return GroupMember.NON_GROUP_MEMBER

        # Use the group name as prefix in the default store, such that
        # a single store can be reused by multiple groups.
        prefix_store = PrefixStore(group_name, store)

        if backend == Backend.GLOO:
            pg = ProcessGroupGloo(
                prefix_store,
                rank,
                world_size,
                timeout=timeout)
            _pg_map[pg] = (Backend.GLOO, store)
            _pg_names[pg] = group_name
        elif backend == Backend.NCCL:
            if not is_nccl_available():
                raise RuntimeError("Distributed package doesn't have NCCL "
                                   "built in")
            pg = ProcessGroupNCCL(
                prefix_store,
                rank,
                world_size,
                timeout)
            _pg_map[pg] = (Backend.NCCL, store)
            _pg_names[pg] = group_name
        else:
            pg = getattr(Backend, backend.upper())(
                prefix_store,
                rank,
                world_size,
                timeout)
            _pg_map[pg] = (backend, store)
            _pg_names[pg] = group_name

    return pg


def destroy_process_group(group=group.WORLD):
    """
    Destroy a given process group, and deinitialize the distributed package

    Arguments:
        group (ProcessGroup, optional): The process group to be destroyed, if
                                        group.WORLD is given, all process
                                        groups including the default one will
                                        be destroyed.
    """
    global _pg_map
    global _pg_names
    global _pg_group_ranks
    global _default_pg
    global _default_pg_init_method
    global _group_count

    if group == GroupMember.NON_GROUP_MEMBER:
        return

    if group == GroupMember.WORLD:
        pg = _default_pg
    else:
        pg = group

    if _pg_map.get(pg, None) is None:
        raise RuntimeError("Invalid process group specified")

    if group == GroupMember.WORLD:
        _default_pg = None
        _default_pg_init_method = None
        _pg_map.clear()
        _pg_names.clear()
        _pg_group_ranks.clear()

        # when process group doesn't have an explicit name (only WORLD (default)
        # process group can have an explicit name), we use global _group_counter
        # to generate the name. We need to reset the counter on destruction to
        # allow consistent value to be generated when we re-create process
        # groups after some trainers recover from failure
        #
        # We only reset this when WORLD is being destroyed because if this
        # process group is in good state, we aren't dealing with failures.
        _group_count = 0
    else:
        del _pg_map[pg]
        del _pg_names[pg]
        del _pg_group_ranks[pg]


def get_rank(group=group.WORLD):
    """
    Returns the rank of current process group

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Arguments:
        group (ProcessGroup, optional): The process group to work on

    Returns:
        The rank of the process group
        -1, if not part of the group

    """
    if _rank_not_in_group(group):
        return -1

    _check_default_pg()
    if group == GroupMember.WORLD:
        return _default_pg.rank()

    return _get_group_rank(group, _default_pg.rank())


def get_world_size(group=group.WORLD):
    """
    Returns the number of processes in the current process group

    Arguments:
        group (ProcessGroup, optional): The process group to work on

    Returns:
        The world size of the process group
        -1, if not part of the group

    """
    if _rank_not_in_group(group):
        return -1

    return _get_group_size(group)


def isend(tensor,
          dst,
          group=group.WORLD,
          tag=0):
    """
    Sends a tensor asynchronously.

    Arguments:
        tensor (Tensor): Tensor to send.
        dst (int): Destination rank.
        group (ProcessGroup, optional): The process group to work on
        tag (int, optional): Tag to match send with remote recv

    Returns:
        A distributed request object.
        None, if not part of the group

    """
    _check_single_tensor(tensor, "tensor")
    if _rank_not_in_group(group):
        return

    if group == GroupMember.WORLD:
        _check_default_pg()
        return _default_pg.send([tensor], dst, tag)
    else:
        group_dst_rank = _get_group_rank(group, dst)
        return group.send([tensor], group_dst_rank, tag)


def irecv(tensor,
          src,
          group=group.WORLD,
          tag=0):
    """
    Receives a tensor asynchronously.

    Arguments:
        tensor (Tensor): Tensor to fill with received data.
        src (int): Source rank.
        group (ProcessGroup, optional): The process group to work on
        tag (int, optional): Tag to match recv with remote send

    Returns:
        A distributed request object.
        None, if not part of the group

    """
    _check_single_tensor(tensor, "tensor")
    if _rank_not_in_group(group):
        return

    if group == GroupMember.WORLD:
        _check_default_pg()
        return _default_pg.recv([tensor], src, tag)
    else:
        group_src_rank = _get_group_rank(group, src)
        return group.recv([tensor], group_src_rank, tag)


def send(tensor,
         dst,
         group=group.WORLD,
         tag=0):
    """
    Sends a tensor synchronously.

    Arguments:
        tensor (Tensor): Tensor to send.
        dst (int): Destination rank.
        group (ProcessGroup, optional): The process group to work on
        tag (int, optional): Tag to match send with remote recv

    """
    _check_single_tensor(tensor, "tensor")
    if _rank_not_in_group(group):
        return

    if group == GroupMember.WORLD:
        _check_default_pg()
        _default_pg.send([tensor], dst, tag).wait()
    else:
        group_dst_rank = _get_group_rank(group, dst)
        group.send([tensor], group_dst_rank, tag).wait()


def recv(tensor,
         src=None,
         group=group.WORLD,
         tag=0):
    """
    Receives a tensor synchronously.

    Arguments:
        tensor (Tensor): Tensor to fill with received data.
        src (int, optional): Source rank. Will receive from any
            process if unspecified.
        group (ProcessGroup, optional): The process group to work on
        tag (int, optional): Tag to match recv with remote send

    Returns:
        Sender rank
        -1, if not part of the group

    """
    _check_single_tensor(tensor, "tensor")
    if _rank_not_in_group(group):
        return -1

    if group == GroupMember.WORLD:
        _check_default_pg()
        pg = _default_pg
    else:
        pg = group

    if src is None:
        work = pg.recv_anysource([tensor], tag)
        work.wait()
        src_rank = work._source_rank()
        if group == GroupMember.WORLD:
            return src_rank
        else:
            return _get_global_rank(pg, src_rank)
    else:
        if group == GroupMember.WORLD:
            pg.recv([tensor], src, tag).wait()
        else:
            group_src_rank = _get_group_rank(pg, src)
            pg.recv([tensor], group_src_rank, tag).wait()
        return src


def broadcast_multigpu(tensor_list,
                       src,
                       group=group.WORLD,
                       async_op=False,
                       src_tensor=0):
    """
    Broadcasts the tensor to the whole group with multiple GPU tensors
    per node.

    ``tensor`` must have the same number of elements in all the GPUs from
    all processes participating in the collective. each tensor in the list must
    be on a different GPU

    Only nccl and gloo backend are currently supported
    tensors should only be GPU tensors

    Arguments:
        tensor_list (List[Tensor]): Tensors that participate in the collective
            operation. If ``src`` is the rank, then the specified ``src_tensor``
            element of ``tensor_list`` (``tensor_list[src_tensor]``) will be
            broadcast to all other tensors (on different GPUs) in the src process
            and all tensors in ``tensor_list`` of other non-src processes.
            You also need to make sure that ``len(tensor_list)`` is the same
            for all the distributed processes calling this function.

        src (int): Source rank.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op
        src_tensor (int, optional): Source tensor rank within ``tensor_list``

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    if _rank_not_in_group(group):
        return

    opts = BroadcastOptions()
    opts.rootRank = src
    opts.rootTensor = src_tensor

    if group == GroupMember.WORLD:
        _check_default_pg()
        work = _default_pg.broadcast(tensor_list, opts)
    else:
        group_src_rank = _get_group_rank(group, src)
        opts.rootRank = group_src_rank
        work = group.broadcast(tensor_list, opts)
    if async_op:
        return work
    else:
        work.wait()


def broadcast(tensor,
              src,
              group=group.WORLD,
              async_op=False):
    """
    Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes
    participating in the collective.

    Arguments:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process, and tensor to be used to save received data otherwise.
        src (int): Source rank.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    _check_single_tensor(tensor, "tensor")
    if _rank_not_in_group(group):
        return

    opts = BroadcastOptions()
    opts.rootRank = src
    opts.rootTensor = 0

    if group == GroupMember.WORLD:
        _check_default_pg()
        work = _default_pg.broadcast([tensor], opts)
    else:
        group_src_rank = _get_group_rank(group, src)
        opts.rootRank = group_src_rank
        work = group.broadcast([tensor], opts)
    if async_op:
        return work
    else:
        work.wait()


def all_reduce_multigpu(tensor_list,
                        op=ReduceOp.SUM,
                        group=group.WORLD,
                        async_op=False):
    r"""
    Reduces the tensor data across all machines in such a way that all get
    the final result. This function reduces a number of tensors on every node,
    while each tensor resides on different GPUs.
    Therefore, the input tensor in the tensor list needs to be GPU tensors.
    Also, each tensor in the tensor list needs to reside on a different GPU.

    After the call, all ``tensor`` in ``tensor_list`` is going to be bitwise
    identical in all processes.

    Only nccl and gloo backend is currently supported
    tensors should only be GPU tensors

    Arguments:
        tensor list (List[Tensor]): List of input and output tensors of
            the collective. The function operates in-place and requires that
            each tensor to be a GPU tensor on different GPUs.
            You also need to make sure that ``len(tensor_list)`` is the same for
            all the distributed processes calling this function.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    if _rank_not_in_group(group):
        return

    opts = AllreduceOptions()
    opts.reduceOp = op
    if group == GroupMember.WORLD:
        _check_default_pg()
        work = _default_pg.allreduce(tensor_list, opts)
    else:
        work = group.allreduce(tensor_list, opts)

    if async_op:
        return work
    else:
        work.wait()


def all_reduce(tensor,
               op=ReduceOp.SUM,
               group=group.WORLD,
               async_op=False):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result.

    After the call ``tensor`` is going to be bitwise identical in all processes.

    Arguments:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    _check_single_tensor(tensor, "tensor")
    if _rank_not_in_group(group):
        return

    opts = AllreduceOptions()
    opts.reduceOp = op
    if group == GroupMember.WORLD:
        _check_default_pg()
        work = _default_pg.allreduce([tensor], opts)
    else:
        work = group.allreduce([tensor], opts)

    if async_op:
        return work
    else:
        work.wait()


def all_reduce_coalesced(tensors,
                         op=ReduceOp.SUM,
                         group=group.WORLD,
                         async_op=False):
    """
    WARNING: at this time individual shape checking is not implemented across nodes.
    For example, if the rank 0 node passes [torch.rand(4), torch.rand(2)] and the
    rank 1 node passes [torch.rand(2), torch.rand(2), torch.rand(2)], the allreduce
    operation will proceed without complaint and return erroneous outputs. This lack
    of shape checking results in significant performance improvements but users of this
    function should take extra care to ensure that each node passes in tensors whose
    shapes match across nodes.

    Reduces each tensor in tensors (residing on the same device) across all machines
    in such a way that all get the final result.

    After the call each tensor in tensors is going to bitwise identical
    in all processes.

    Arguments:
        tensors (List[Tensor]): Input and output of the collective. The function
            operates in-place.
        op (Optional[ReduceOp]): One of the values from
            ``torch.distributed.ReduceOp`` enum. Specifies an operation used for
            element-wise reductions.
        group (Optional[ProcessGroup]): The process group to work on.
        async_op (Optional[bool]): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
    _check_tensor_list(tensors, "tensor")
    if _rank_not_in_group(group):
        return

    opts = AllreduceCoalescedOptions()
    opts.reduceOp = op
    if group == GroupMember.WORLD:
        _check_default_pg()
        work = _default_pg.allreduce_coalesced(tensors, opts)
    else:
        work = group.allreduce_coalesced(tensors, opts)

    if async_op:
        return work
    else:
        work.wait()


def reduce_multigpu(tensor_list,
                    dst,
                    op=ReduceOp.SUM,
                    group=group.WORLD,
                    async_op=False,
                    dst_tensor=0):
    """
    Reduces the tensor data on multiple GPUs across all machines. Each tensor
    in ``tensor_list`` should reside on a separate GPU

    Only the GPU of ``tensor_list[dst_tensor]`` on the process with rank ``dst``
    is going to receive the final result.

    Only nccl backend is currently supported
    tensors should only be GPU tensors

    Arguments:
        tensor_list (List[Tensor]): Input and output GPU tensors of the
            collective. The function operates in-place.
            You also need to make sure that ``len(tensor_list)`` is the same for
            all the distributed processes calling this function.
        dst (int): Destination rank
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op
        dst_tensor (int, optional): Destination tensor rank within
                                    ``tensor_list``

    Returns:
        Async work handle, if async_op is set to True.
        None, otherwise

    """
    if _rank_not_in_group(group):
        return

    opts = ReduceOptions()
    opts.reduceOp = op
    opts.rootRank = dst
    opts.rootTensor = dst_tensor

    if group == GroupMember.WORLD:
        _check_default_pg()
        work = _default_pg.reduce(tensor_list, opts)
    else:
        group_dst_rank = _get_group_rank(group, dst)
        opts.rootRank = group_dst_rank
        work = group.reduce(tensor_list, opts)

    if async_op:
        return work
    else:
        work.wait()


def reduce(tensor,
           dst,
           op=ReduceOp.SUM,
           group=group.WORLD,
           async_op=False):
    """
    Reduces the tensor data across all machines.

    Only the process with rank ``dst`` is going to receive the final result.

    Arguments:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        dst (int): Destination rank
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    _check_single_tensor(tensor, "tensor")
    if _rank_not_in_group(group):
        return

    opts = ReduceOptions()
    opts.reduceOp = op
    opts.rootRank = dst

    if group == GroupMember.WORLD:
        _check_default_pg()
        work = _default_pg.reduce([tensor], opts)
    else:
        group_dst_rank = _get_group_rank(group, dst)
        opts.rootRank = group_dst_rank
        work = group.reduce([tensor], opts)

    if async_op:
        return work
    else:
        work.wait()


def all_gather_multigpu(output_tensor_lists,
                        input_tensor_list,
                        group=group.WORLD,
                        async_op=False):
    """
    Gathers tensors from the whole group in a list.
    Each tensor in ``tensor_list`` should reside on a separate GPU

    Only nccl backend is currently supported
    tensors should only be GPU tensors

    Arguments:
        output_tensor_lists (List[List[Tensor]]): Output lists. It should
            contain correctly-sized tensors on each GPU to be used for output
            of the collective, e.g. ``output_tensor_lists[i]`` contains the
            all_gather result that resides on the GPU of
            ``input_tensor_list[i]``.

            Note that each element of ``output_tensor_lists`` has the size of
            ``world_size * len(input_tensor_list)``, since the function all
            gathers the result from every single GPU in the group. To interpret
            each element of ``output_tensor_lists[i]``, note that
            ``input_tensor_list[j]`` of rank k will be appear in
            ``output_tensor_lists[i][k * world_size + j]``

            Also note that ``len(output_tensor_lists)``, and the size of each
            element in ``output_tensor_lists`` (each element is a list,
            therefore ``len(output_tensor_lists[i])``) need to be the same
            for all the distributed processes calling this function.

        input_tensor_list (List[Tensor]): List of tensors(on different GPUs) to
            be broadcast from current process.
            Note that ``len(input_tensor_list)`` needs to be the same for
            all the distributed processes calling this function.

        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    if _rank_not_in_group(group):
        return

    if group == GroupMember.WORLD:
        _check_default_pg()
        work = _default_pg.allgather(output_tensor_lists, input_tensor_list)
    else:
        work = group.allgather(output_tensor_lists, input_tensor_list)

    if async_op:
        return work
    else:
        work.wait()


def _object_to_tensor(obj):
    buffer = pickle.dumps(obj)
    byte_storage = torch.ByteStorage.from_buffer(buffer)
    byte_tensor = torch.ByteTensor(byte_storage)
    local_size = torch.LongTensor([byte_tensor.numel()])
    return byte_tensor, local_size


def _tensor_to_object(tensor, tensor_size):
    buf = tensor.numpy().tobytes()[:tensor_size]
    out = pickle.loads(buf)
    return out


def all_gather_object(object_list, obj, group=group.WORLD):
    """
    Gathers picklable objects from the whole group into a list. Similar to
    :func:`all_gather`, but Python objects can be passed in. Note that the object
    must be picklable in order to be gathered.

    Arguments:
        object_list (list[Any]): Output list. It should be correctly sized as the
            size of the group for this collective and will contain the output.
        object (Any): Pickable Python object to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on

    Returns:
        None. If the calling rank is part of this group, the output of the
        collective will be populated into the input ``object_list``. If the
        calling rank is not part of the group, the passed in ``object_list`` will
        be unmodified.

    .. note:: Note that this API differs slightly from the :func:`all_gather`
        collective since it does not provide an ``async_op`` handle and thus
        will be a blocking call.

    .. warning::
        :func:`all_gather_object` uses ``pickle`` module implicitly, which is
        known to be insecure. It is possible to construct malicious pickle data
        which will execute arbitrary code during unpickling. Only call this
        function with data you trust.
    """
    if _rank_not_in_group(group):
        return

    input_tensor, local_size = _object_to_tensor(obj)
    group_backend = get_backend(group)
    my_rank = get_rank()
    is_nccl_backend = group_backend == Backend.NCCL
    if is_nccl_backend:
        input_tensor, local_size = input_tensor.to(my_rank), local_size.to(my_rank)
    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    group_size = get_world_size(group=group)
    object_sizes_tensor = torch.zeros(group_size, dtype=int).to(
        my_rank if is_nccl_backend else "cpu"
    )
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
    ]
    # Allgather tensor sizes
    all_gather(object_size_list, local_size, group=group)
    max_object_size = max(object_size_list)
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    coalesced_output_tensor = torch.empty(
        max_object_size * group_size, dtype=torch.uint8
    ).to(my_rank if is_nccl_backend else "cpu")
    # Output tensors are nonoverlapping views of coalesced_output_tensor
    output_tensors = [
        coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
        for i in range(group_size)
    ]
    all_gather(output_tensors, input_tensor, group=group)
    # Deserialize outputs back to object.
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.ByteTensor)
        tensor_size = object_size_list[i]
        object_list[i] = _tensor_to_object(tensor, tensor_size)


def gather_object(obj, object_gather_list=None, dst=0, group=group.WORLD):
    """
    Gathers picklable objects from the whole group in a single process.
    Similar to :func:`gather`, but Python objects can be passed in. Note that the
    object must be picklable in order to be gathered.

    Arguments:
        obj (Any): Input object. Must be picklable.
        object_gather_list (list[Any]): Output list. On the ``dst`` rank, it
            should be correctly sized as the size of the group for this
            collective and will contain the output. Must be ``None`` on non-dst
            ranks. (default is ``None``)
        dst (int, optional): Destination rank. (default is 0)
        group: (ProcessGroup, optional): The process group to work on.

    Returns:
        None. On the ``dst`` rank, ``object_gather_list`` will contain the
        output of the collective.

    .. note:: Note that this API differs slightly from the gather collective
        since it does not provide an async_op handle and thus will be a blocking
        call.

    .. note:: Note that this API is not supported when using the NCCL backend.

    .. warning::
        :func:`gather_object` uses ``pickle`` module implicitly, which is
        known to be insecure. It is possible to construct malicious pickle data
        which will execute arbitrary code during unpickling. Only call this
        function with data you trust.
    """
    if _rank_not_in_group(group):
        return

    # Ensure object_gather_list is specified appopriately.
    my_rank = get_rank()
    _validate_output_list_for_rank(my_rank, dst, object_gather_list)
    input_tensor, local_size = _object_to_tensor(obj)
    group_backend = get_backend(group)
    is_nccl_backend = group_backend == Backend.NCCL
    if is_nccl_backend:
        input_tensor, local_size = input_tensor.to(my_rank), local_size.to(my_rank)
    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    group_size = get_world_size(group=group)
    object_sizes_tensor = torch.zeros(group_size, dtype=int).to(
        my_rank if is_nccl_backend else "cpu"
    )
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
    ]
    # Allgather tensor sizes. An all-gather is needed here despite this being a gather,
    # since each rank needs to broadcast a tensor of the same (maximal) size.
    all_gather(object_size_list, local_size, group=group)
    max_object_size = max(object_size_list)
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    # Avoid populating output tensors if the result won't be gathered on this rank.
    if my_rank == dst:
        coalesced_output_tensor = torch.empty(
            max_object_size * group_size, dtype=torch.uint8
        ).to(my_rank if is_nccl_backend else "cpu")
        # Output tensors are nonoverlapping views of coalesced_output_tensor
        output_tensors = [
            coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
            for i in range(group_size)
        ]
    # All ranks call gather with equal-sized tensors.
    gather(
        input_tensor,
        gather_list=output_tensors if my_rank == dst else None,
        dst=dst,
        group=group,
    )
    if my_rank != dst:
        return
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.ByteTensor)
        tensor_size = object_size_list[i]
        object_gather_list[i] = _tensor_to_object(tensor, tensor_size)


def broadcast_object_list(object_list, src, group=group.WORLD):
    """
    Broadcasts picklable objects in ``object_list`` to the whole group. Similar
    to :func:`broadcast`, but Python objects can be passed in.
    Note that all objects in ``object_list`` must be picklable in order to be
    broadcasted.

    Arguments:
        object_list (List[Any]): List of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank will
            be broadcast, but each rank must provide lists of equal sizes.
        src (int): Source rank from which to broadcast ``object_list``.
        group: (ProcessGroup, optional): The process group to work on.

    Returns:
        ``None``. If rank is part of the group, ``object_list`` will contain the
        broadcasted objects from ``src`` rank.

    .. note:: Note that this API differs slightly from the broadcast collective
        since it does not provide an ``async_op`` handle and thus will be a
        blocking call.

    .. warning::
        :func:`broadcast_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.
    """
    if _rank_not_in_group(group):
        return

    my_rank = get_rank()
    # Serialize object_list elements to tensors on src rank.
    if my_rank == src:
        tensor_list, size_list = zip(*[_object_to_tensor(obj) for obj in object_list])
        object_sizes_tensor = torch.cat(size_list)
    else:
        object_sizes_tensor = torch.LongTensor(len(object_list))

    group_backend = get_backend(group)
    is_nccl_backend = group_backend == Backend.NCCL
    if is_nccl_backend:
        object_sizes_tensor = object_sizes_tensor.to(my_rank)

    # Broadcast object sizes
    broadcast(object_sizes_tensor, src=src, group=group)

    # Concatenate and broadcast serialized object tensors
    if my_rank == src:
        object_tensor = torch.cat(tensor_list)
    else:
        object_tensor = torch.ByteTensor(torch.sum(object_sizes_tensor).item())

    if is_nccl_backend:
        object_tensor = object_tensor.to(my_rank)
    broadcast(object_tensor, src=src, group=group)
    # Deserialize objects using their stored sizes.
    offset = 0
    if my_rank != src:
        for i, obj_size in enumerate(object_sizes_tensor):
            obj_view = object_tensor[offset : offset + obj_size]
            obj_view = obj_view.type(torch.ByteTensor)
            offset += obj_size
            object_list[i] = _tensor_to_object(obj_view, obj_size)


def all_gather(tensor_list,
               tensor,
               group=group.WORLD,
               async_op=False):
    """
    Gathers tensors from the whole group in a list.

    Arguments:
        tensor_list (list[Tensor]): Output list. It should contain
            correctly-sized tensors to be used for output of the collective.
        tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    _check_tensor_list(tensor_list, "tensor_list")
    _check_single_tensor(tensor, "tensor")
    if _rank_not_in_group(group):
        return

    if group == GroupMember.WORLD:
        _check_default_pg()
        work = _default_pg.allgather([tensor_list], [tensor])
    else:
        work = group.allgather([tensor_list], [tensor])

    if async_op:
        return work
    else:
        work.wait()

def all_gather_coalesced(output_tensor_lists,
                         input_tensor_list,
                         group=group.WORLD,
                         async_op=False):
    """
    Gathers input tensors from the whole group in a list in a coalesced manner.

    Arguments:
        output_tensor_lists (list[list[Tensor]]): Output list. It should contain
            correctly-sized tensors to be used for output of the collective.
        input_tensor_list (list[Tensor]): Tensors to be broadcast from
            current process. At least one tensor has to be non empty.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    Example:
        we have 2 process groups, 2 ranks.
        rank 0 passes:
            input_tensor_list = [[[1, 1], [1, 1]], [2], [3, 3]]
            output_tensor_lists =
               [[[[-1, -1], [-1, -1]], [-1], [-1, -1]],
                [[[-1, -1], [-1, -1]], [-1], [-1, -1]]]
        rank 1 passes:
            input_tensor_list = [[[3, 3], [3, 3]], [5], [1, 1]]
            output_tensor_lists =
               [[[[-1, -1], [-1, -1]], [-1], [-1, -1]],
                [[[-1, -1], [-1, -1]], [-1], [-1, -1]]]
        both rank 0 and 1 get:
            output_tensor_lists =
               [[[1, 1], [1, 1]], [2], [3, 3]],
                [[3, 3], [3, 3]], [5], [1, 1]]].

    WARNING: at this time individual shape checking is not implemented across nodes.
    For example, if the rank 0 node passes [torch.rand(4), torch.rand(2)] and the
    rank 1 node passes [torch.rand(2), torch.rand(2), torch.rand(2)], the
    all_gather_coalesced operation will proceed without complaint and return
    erroneous outputs. This lack of shape checking results in significant
    performance improvements but users of this function should take extra care
    to ensure that each node passes in tensors whose shapes match across nodes.
    """
    # We only check basic compatibility with C++ params here, C++ code will
    # do shape and type checking.
    if _rank_not_in_group(group):
        return
    _check_tensor_list(input_tensor_list, "tensor_list")
    if not isinstance(output_tensor_lists, list):
        raise RuntimeError("Invalid function argument: "
                           "output_tensor_lists should be a list")
    for output_tensor_list in output_tensor_lists:
        _check_tensor_list(output_tensor_list, "output_tensor_lists")

    if group == GroupMember.WORLD:
        _check_default_pg()
        work = _default_pg.allgather_coalesced(
            output_tensor_lists, input_tensor_list)
    else:
        work = group.allgather_coalesced(output_tensor_lists, input_tensor_list)

    if async_op:
        return work
    else:
        work.wait()

def _validate_output_list_for_rank(my_rank, dst, gather_list):
    if dst == my_rank:
        if not gather_list:
            raise ValueError(
                "Argument ``gather_list`` must be specified on destination rank."
            )
    elif gather_list:
        raise ValueError(
            "Argument ``gather_list`` must NOT be specified "
            "on non-destination ranks."
        )


def gather(tensor,
           gather_list=None,
           dst=0,
           group=group.WORLD,
           async_op=False):
    """
    Gathers a list of tensors in a single process.

    Arguments:
        tensor (Tensor): Input tensor.
        gather_list (list[Tensor], optional): List of appropriately-sized
            tensors to use for gathered data (default is None, must be specified
            on the destination rank)
        dst (int, optional): Destination rank (default is 0)
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    _check_single_tensor(tensor, "tensor")

    # Parameter ``gather_list`` may be left unspecified on non-dst ranks.
    if gather_list:
        _check_tensor_list(gather_list, "gather_list")
    else:
        gather_list = []

    if _rank_not_in_group(group):
        return

    my_rank = get_rank()
    _validate_output_list_for_rank(my_rank, dst, gather_list)
    output_tensors = [gather_list] if dst == my_rank else []
    input_tensors = [tensor]

    opts = GatherOptions()
    opts.rootRank = dst

    if group == GroupMember.WORLD:
        _check_default_pg()
        work = _default_pg.gather(output_tensors, input_tensors, opts)
    else:
        group_dst_rank = _get_group_rank(group, dst)
        opts.rootRank = group_dst_rank
        work = group.gather(output_tensors, input_tensors, opts)

    if async_op:
        return work
    else:
        work.wait()


def scatter(tensor,
            scatter_list=None,
            src=0,
            group=group.WORLD,
            async_op=False):
    """
    Scatters a list of tensors to all processes in a group.

    Each process will receive exactly one tensor and store its data in the
    ``tensor`` argument.

    Arguments:
        tensor (Tensor): Output tensor.
        scatter_list (list[Tensor]): List of tensors to scatter (default is
            None, must be specified on the source rank)
        src (int): Source rank (default is 0)
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    _check_single_tensor(tensor, "tensor")

    # Parameter ``scatter_list`` may be left unspecified on non-src ranks.
    if scatter_list:
        _check_tensor_list(scatter_list, "scatter_list")
    else:
        scatter_list = []

    if _rank_not_in_group(group):
        return

    my_rank = get_rank()
    if src == my_rank:
        if not scatter_list:
            raise ValueError("Argument ``scatter_list`` must be specified "
                             "on source rank.")
        input_tensors = [scatter_list]
        output_tensors = [tensor]
    else:
        if scatter_list:
            raise ValueError("Argument ``scatter_list`` must NOT be specified "
                             "on non-source ranks.")
        input_tensors = []
        output_tensors = [tensor]

    opts = ScatterOptions()
    opts.rootRank = src

    if group == GroupMember.WORLD:
        _check_default_pg()
        work = _default_pg.scatter(output_tensors, input_tensors, opts)
    else:
        group_src_rank = _get_group_rank(group, src)
        opts.rootRank = group_src_rank
        work = group.scatter(output_tensors, input_tensors, opts)

    if async_op:
        return work
    else:
        work.wait()


def reduce_scatter_multigpu(output_tensor_list,
                            input_tensor_lists,
                            op=ReduceOp.SUM,
                            group=group.WORLD,
                            async_op=False):
    """
    Reduce and scatter a list of tensors to the whole group.  Only nccl backend
    is currently supported.

    Each tensor in ``output_tensor_list`` should reside on a separate GPU, as
    should each list of tensors in ``input_tensor_lists``.

    Arguments:
        output_tensor_list (List[Tensor]): Output tensors (on different GPUs)
            to receive the result of the operation.

            Note that ``len(output_tensor_list)`` needs to be the same for all
            the distributed processes calling this function.

        input_tensor_lists (List[List[Tensor]]): Input lists.  It should
            contain correctly-sized tensors on each GPU to be used for input of
            the collective, e.g. ``input_tensor_lists[i]`` contains the
            reduce_scatter input that resides on the GPU of
            ``output_tensor_list[i]``.

            Note that each element of ``input_tensor_lists`` has the size of
            ``world_size * len(output_tensor_list)``, since the function
            scatters the result from every single GPU in the group.  To
            interpret each element of ``input_tensor_lists[i]``, note that
            ``output_tensor_list[j]`` of rank k receives the reduce-scattered
            result from ``input_tensor_lists[i][k * world_size + j]``

            Also note that ``len(input_tensor_lists)``, and the size of each
            element in ``input_tensor_lists`` (each element is a list,
            therefore ``len(input_tensor_lists[i])``) need to be the same for
            all the distributed processes calling this function.

        group (ProcessGroup, optional): The process group to work on.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
    if _rank_not_in_group(group):
        return

    opts = ReduceScatterOptions()
    opts.reduceOp = op

    if group == GroupMember.WORLD:
        _check_default_pg()
        work = _default_pg.reduce_scatter(
            output_tensor_list,
            input_tensor_lists,
            opts
        )
    else:
        work = group.reduce_scatter(
            output_tensor_list,
            input_tensor_lists,
            opts
        )

    if async_op:
        return work
    else:
        work.wait()


def reduce_scatter(output,
                   input_list,
                   op=ReduceOp.SUM,
                   group=group.WORLD,
                   async_op=False):
    """
    Reduces, then scatters a list of tensors to all processes in a group.

    Arguments:
        output (Tensor): Output tensor.
        input_list (list[Tensor]): List of tensors to reduce and scatter.
        group (ProcessGroup, optional): The process group to work on.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
    _check_single_tensor(output, "output")
    _check_tensor_list(input_list, "input_list")
    if _rank_not_in_group(group):
        return

    opts = ReduceScatterOptions()
    opts.reduceOp = op

    if group == GroupMember.WORLD:
        _check_default_pg()
        work = _default_pg.reduce_scatter([output], [input_list], opts)
    else:
        work = group.reduce_scatter([output], [input_list], opts)

    if async_op:
        return work
    else:
        work.wait()


def all_to_all_single(output,
                      input,
                      output_split_sizes=None,
                      input_split_sizes=None,
                      group=group.WORLD,
                      async_op=False):
    """
    Each process splits input tensor and then scatters the split list
    to all processes in a group. Then concatenate the received tensors from all
    the processes in the group and return single output tensor.

    Arguments:
        output (Tensor): Gathered cancatenated output tensor.
        input (Tensor): Input tensor to scatter.
        output_split_sizes: (list[Int], optional): Output split sizes for dim 0
            if specified None or empty, dim 0 of ``output`` tensor must divide
            equally by ``world_size``.
        input_split_sizes: (list[Int], optional): Input split sizes for dim 0
            if specified None or empty, dim 0 of ``input`` tensor must divide
            equally by ``world_size``.
        group (ProcessGroup, optional): The process group to work on.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    .. warning::
        `all_to_all_single` is experimental and subject to change.

    Examples:
        >>> input = torch.arange(4) + rank * 4
        >>> input
        tensor([0, 1, 2, 3])     # Rank 0
        tensor([4, 5, 6, 7])     # Rank 1
        tensor([8, 9, 10, 11])   # Rank 2
        tensor([12, 13, 14, 15]) # Rank 3
        >>> output = torch.empty([4], dtype=torch.int64)
        >>> dist.all_to_all_single(output, input)
        >>> output
        tensor([0, 4, 8, 12])    # Rank 0
        tensor([1, 5, 9, 13])    # Rank 1
        tensor([2, 6, 10, 14])   # Rank 2
        tensor([3, 7, 11, 15])   # Rank 3

        >>> # Essentially, it is similar to following operation:
        >>> scatter_list = list(input.chunk(world_size))
        >>> gather_list  = list(output.chunk(world_size))
        >>> for i in range(world_size):
        >>>   dist.scatter(gather_list[i], scatter_list if i == rank else [], src = i)

        >>> # Another example with uneven split
        >>> input
        tensor([0, 1, 2, 3, 4, 5])                                       # Rank 0
        tensor([10, 11, 12, 13, 14, 15, 16, 17, 18])                     # Rank 1
        tensor([20, 21, 22, 23, 24])                                     # Rank 2
        tensor([30, 31, 32, 33, 34, 35, 36])                             # Rank 3
        >>> input_splits
        [2, 2, 1, 1]                                                     # Rank 0
        [3, 2, 2, 2]                                                     # Rank 1
        [2, 1, 1, 1]                                                     # Rank 2
        [2, 2, 2, 1]                                                     # Rank 3
        >>> output_splits
        [2, 3, 2, 2]                                                     # Rank 0
        [2, 2, 1, 2]                                                     # Rank 1
        [1, 2, 1, 2]                                                     # Rank 2
        [1, 2, 1, 1]                                                     # Rank 3
        >>> output = ...
        >>> dist.all_to_all_single(output, input, output_splits, input_splits)
        >>> output
        tensor([ 0,  1, 10, 11, 12, 20, 21, 30, 31])                     # Rank 0
        tensor([ 2,  3, 13, 14, 22, 32, 33])                             # Rank 1
        tensor([ 4, 15, 16, 23, 34, 35])                                 # Rank 2
        tensor([ 5, 17, 18, 24, 36])                                     # Rank 3
    """
    if _rank_not_in_group(group):
        return

    opts = AllToAllOptions()
    _check_single_tensor(output, "output")
    _check_single_tensor(input, "input")
    output_split_sizes = [] if output_split_sizes is None else output_split_sizes
    input_split_sizes = [] if input_split_sizes is None else input_split_sizes

    if group == GroupMember.WORLD:
        _check_default_pg()
        work = _default_pg.alltoall_base(output, input, output_split_sizes, input_split_sizes, opts)
    else:
        work = group.alltoall_base(output, input, output_split_sizes, input_split_sizes, opts)

    if async_op:
        return work
    else:
        work.wait()

def all_to_all(output_tensor_list,
               input_tensor_list,
               group=group.WORLD,
               async_op=False):
    """
    Each process scatters list of input tensors to all processes in a group and
    return gathered list of tensors in output list.

    Arguments:
        output_tensor_list (list[Tensor]): List of tensors to be gathered one
            per rank.
        input_tensor_list (list[Tensor]): List of tensors to scatter one per rank.
        group (ProcessGroup, optional): The process group to work on.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    .. warning::
        `all_to_all` is experimental and subject to change.

    Examples:
        >>> input = torch.arange(4) + rank * 4
        >>> input = list(input.chunk(4))
        >>> input
        [tensor([0]), tensor([1]), tensor([2]), tensor([3])]     # Rank 0
        [tensor([4]), tensor([5]), tensor([6]), tensor([7])]     # Rank 1
        [tensor([8]), tensor([9]), tensor([10]), tensor([11])]   # Rank 2
        [tensor([12]), tensor([13]), tensor([14]), tensor([15])] # Rank 3
        >>> output = list(torch.empty([4], dtype=torch.int64).chunk(4))
        >>> dist.all_to_all(output, input)
        >>> output
        [tensor([0]), tensor([4]), tensor([8]), tensor([12])]    # Rank 0
        [tensor([1]), tensor([5]), tensor([9]), tensor([13])]    # Rank 1
        [tensor([2]), tensor([6]), tensor([10]), tensor([14])]   # Rank 2
        [tensor([3]), tensor([7]), tensor([11]), tensor([15])]   # Rank 3

        >>> # Essentially, it is similar to following operation:
        >>> scatter_list = input
        >>> gather_list  = output
        >>> for i in range(world_size):
        >>>   dist.scatter(gather_list[i], scatter_list if i == rank else [], src = i)

        >>> input
        tensor([0, 1, 2, 3, 4, 5])                                       # Rank 0
        tensor([10, 11, 12, 13, 14, 15, 16, 17, 18])                     # Rank 1
        tensor([20, 21, 22, 23, 24])                                     # Rank 2
        tensor([30, 31, 32, 33, 34, 35, 36])                             # Rank 3
        >>> input_splits
        [2, 2, 1, 1]                                                     # Rank 0
        [3, 2, 2, 2]                                                     # Rank 1
        [2, 1, 1, 1]                                                     # Rank 2
        [2, 2, 2, 1]                                                     # Rank 3
        >>> output_splits
        [2, 3, 2, 2]                                                     # Rank 0
        [2, 2, 1, 2]                                                     # Rank 1
        [1, 2, 1, 2]                                                     # Rank 2
        [1, 2, 1, 1]                                                     # Rank 3
        >>> input = list(input.split(input_splits))
        >>> input
        [tensor([0, 1]), tensor([2, 3]), tensor([4]), tensor([5])]                   # Rank 0
        [tensor([10, 11, 12]), tensor([13, 14]), tensor([15, 16]), tensor([17, 18])] # Rank 1
        [tensor([20, 21]), tensor([22]), tensor([23]), tensor([24])]                 # Rank 2
        [tensor([30, 31]), tensor([32, 33]), tensor([34, 35]), tensor([36])]         # Rank 3
        >>> output = ...
        >>> dist.all_to_all(output, input)
        >>> output
        [tensor([0, 1]), tensor([10, 11, 12]), tensor([20, 21]), tensor([30, 31])]   # Rank 0
        [tensor([2, 3]), tensor([13, 14]), tensor([22]), tensor([32, 33])]           # Rank 1
        [tensor([4]), tensor([15, 16]), tensor([23]), tensor([34, 35])]              # Rank 2
        [tensor([5]), tensor([17, 18]), tensor([24]), tensor([36])]                  # Rank 3
    """
    if _rank_not_in_group(group):
        return

    opts = AllToAllOptions()
    _check_tensor_list(output_tensor_list, "output_tensor_list")
    _check_tensor_list(input_tensor_list, "input_tensor_list")

    if group == GroupMember.WORLD:
        _check_default_pg()
        work = _default_pg.alltoall(output_tensor_list, input_tensor_list, opts)
    else:
        work = group.alltoall(output_tensor_list, input_tensor_list, opts)

    if async_op:
        return work
    else:
        work.wait()


def barrier(group=group.WORLD,
            async_op=False):
    """
    Synchronizes all processes.

    This collective blocks processes until the whole group enters this function,
    if async_op is False, or if async work handle is called on wait().

    Arguments:
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    """
    if _rank_not_in_group(group):
        return

    if group == GroupMember.WORLD:
        _check_default_pg()
        work = _default_pg.barrier()
    else:
        work = group.barrier()

    if async_op:
        return work
    else:
        work.wait()


def new_group(ranks=None, timeout=default_pg_timeout, backend=None):
    """
    Creates a new distributed group.

    This function requires that all processes in the main group (i.e. all
    processes that are part of the distributed job) enter this function, even
    if they are not going to be members of the group. Additionally, groups
    should be created in the same order in all processes.

    Arguments:
        ranks (list[int]): List of ranks of group members. If ``None``, will be
            set to all ranks. Default is ``None``.
        timeout (timedelta, optional): Timeout for operations executed against
            the process group. Default value equals 30 minutes.
            This is only applicable for the ``gloo`` backend.
        backend (str or Backend, optional): The backend to use. Depending on
            build-time configurations, valid values are ``gloo`` and ``nccl``.
            By default uses the same backend as the global group. This field
            should be given as a lowercase string (e.g., ``"gloo"``), which can
            also be accessed via :class:`Backend` attributes (e.g.,
            ``Backend.GLOO``).

    Returns:
        A handle of distributed group that can be given to collective calls.
    """

    _check_default_pg()

    global _pg_group_ranks

    default_backend, default_store = _pg_map[_default_pg]
    global_rank = _default_pg.rank()
    global_world_size = _default_pg.size()

    # Default to the same backend as the global process group
    # if the backend is not specified.
    if not backend:
        backend = default_backend

    # checks the input ranks
    if ranks is not None:
        ranks = sorted(ranks)
        group_world_size = len(ranks)
        if group_world_size > global_world_size:
            raise RuntimeError("the new group's world size should be less or "
                               "equal to the world size set by "
                               "init_process_group")
        # check ranks' sanity
        for rank in ranks:
            if rank < 0 or rank >= global_world_size:
                raise RuntimeError("The new group's rank should be within the "
                                   "the world_size set by init_process_group")
        if global_rank in ranks:
            group_rank = ranks.index(global_rank)
        else:
            group_rank = None
    else:
        ranks = list(range(global_world_size))
        group_world_size = global_world_size
        group_rank = global_rank

    backend = Backend(backend)
    pg = _new_process_group_helper(group_world_size,
                                   group_rank,
                                   ranks,
                                   backend,
                                   default_store,
                                   timeout=timeout)

    # Create the global rank to group rank mapping
    _pg_group_ranks[pg] = {
        global_rank: group_rank
        for group_rank, global_rank in enumerate(ranks)
    }

    # barrier at the end to ensure that once we return from this method, all
    # process groups including global variables are updated correctly on all
    # ranks.
    barrier()

    return pg
