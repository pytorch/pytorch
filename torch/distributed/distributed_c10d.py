import torch
import warnings
from torch._six import string_classes
from datetime import timedelta

from .rendezvous import rendezvous, register_rendezvous_handler
from . import BroadcastOptions, AllreduceOptions, ReduceOptions, \
    ScatterOptions, GatherOptions
from . import ReduceOp
from . import PrefixStore
from . import ProcessGroupGloo


_MPI_AVAILABLE = True
_NCCL_AVAILABLE = True


try:
    from. import ProcessGroupMPI
except ImportError:
    _MPI_AVAILABLE = False

try:
    from. import ProcessGroupNCCL
except ImportError:
    _NCCL_AVAILABLE = False


class Backend(object):
    """
    An enum-like class of available backends: GLOO, NCCL, and MPI.

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
        return value

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
# For MPI pg, it is a map from ProcessGroup to (Backend, Bool), where bool
# represents if the ProcessGroup objects is part of the group
_pg_map = {}
# Process group's names, map from ProcessGroup to str
_pg_names = {}
# Process group's global rank to local rank mapping
_pg_group_ranks = {}

# Default process group state
_default_pg = None
_default_pg_init_method = None

# Default process group wide timeout, if applicable.
# This currently only applies to the gloo backend. To make an attempt at
# backwards compatibility with THD, we use an extraordinarily high default
# timeout, given that THD did not have timeouts.
_default_pg_timeout = timedelta(minutes=30)

# Process group count for default naming
_group_count = 0


def _rank_not_in_group(group):
    """
    Helper that checks if the current process's rank is not in a given group

    """
    default_backend, _ = _pg_map[get_default_group()]
    if default_backend != Backend.MPI:
        return group == GroupMember.NON_GROUP_MEMBER
    else:
        if group == GroupMember.WORLD:
            return False
        else:
            _, in_group = _pg_map[group]
            return not in_group


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
        raise RuntimeError("The global rank is not part of the group")
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
    Helper that checks if the default ProcessGroup has been initializd, with
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


def is_mpi_available():
    """
    Checks if MPI is available

    """
    return _MPI_AVAILABLE


def is_nccl_available():
    """
    Checks if NCCL is available

    """
    return _NCCL_AVAILABLE


def is_initialized():
    """
    Checking if the default process group has been initialized

    """
    return _default_pg is not None


def get_default_group():
    """
    Getting the default process group created by init_process_group

    """
    if not is_initialized():
        raise RuntimeError("Default process group has not been initialized, "
                           "please make sure to call init_process_group.")
    return _default_pg


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
                       init_method="env://",
                       **kwargs):
    """
    Initializes the default distributed process group, and this will also
    initialize the distributed package

    Arguments:
        backend (str or Backend): The backend to use. Depending on
            build-time configurations, valid values include ``mpi``, ``gloo``,
            and ``nccl``. This field should be given as a lowercase string
            (e.g., ``"gloo"``), which can also be accessed via
            :class:`Backend` attributes (e.g., ``Backend.GLOO``).
        init_method (str, optional): URL specifying how to initialize the
                                     process group.
        world_size (int, optional): Number of processes participating in
                                    the job.
        rank (int, optional): Rank of the current process.
        group_name (str, optional, deprecated): Group name.

    To enable ``backend == Backend.MPI``, PyTorch needs to built from source
    on a system that supports MPI. The same applies to NCCL as well.

    """
    global _pg_map
    global _pg_names
    global _backend
    global _default_pg
    global _default_pg_init_method

    if _default_pg is not None:
        raise RuntimeError("trying to initialize the default process group "
                           "twice!")

    world_size = kwargs.pop('world_size', -1)
    group_name = kwargs.pop('group_name', '')
    rank = kwargs.pop('rank', -1)
    assert len(kwargs) == 0, \
        "got unexpected keyword arguments: %s" % ",".join(kwargs.keys())

    backend = Backend(backend)

    if backend == Backend.MPI:
        if not is_mpi_available():
            raise RuntimeError("Distributed package doesn't have MPI built in")

        _default_pg = ProcessGroupMPI([])
        _pg_map[_default_pg] = (Backend.MPI, True)
        _pg_names[_default_pg] = group_name
    else:
        # backward compatible API
        if init_method != "env://" and world_size != -1 and rank != -1:
            url = "{}?rank={}&world_size={}".format(init_method,
                                                    rank,
                                                    world_size)
            store, _, _ = next(rendezvous(url))
        else:
            store, rank, world_size = next(rendezvous(init_method))

        if backend == Backend.GLOO:
            _default_pg = ProcessGroupGloo(
                store,
                rank,
                world_size,
                timeout=_default_pg_timeout)
            _pg_map[_default_pg] = (Backend.GLOO, store)
            _pg_names[_default_pg] = group_name
        elif backend == Backend.NCCL:
            if not is_nccl_available():
                raise RuntimeError("Distributed package doesn't have NCCL "
                                   "built in")
            _default_pg = ProcessGroupNCCL(store, rank, world_size)
            _pg_map[_default_pg] = (Backend.NCCL, store)
            _pg_names[_default_pg] = group_name

    _backend = _pg_map[_default_pg][0]
    _default_pg_init_method = init_method


def _new_process_group_helper(world_size,
                              rank,
                              group_ranks,
                              in_group=True,
                              group_name=""):
    """
    Create a new distributed process group. And the new process group can be
    used to perform collective operations.

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

    default_backend, default_store = _pg_map[_default_pg]

    if default_backend == Backend.MPI:
        if not is_mpi_available():
            raise RuntimeError("Distributed package doesn't have MPI built in")
        pg = ProcessGroupMPI(group_ranks)
        _pg_map[pg] = (Backend.MPI, in_group)
        _pg_names[pg] = group_name
    else:
        # Create the prefix store
        store = PrefixStore(group_name, default_store)

        if default_backend == Backend.GLOO:
            pg = ProcessGroupGloo(
                store,
                rank,
                world_size,
                timeout=_default_pg_timeout)
            _pg_map[pg] = (Backend.GLOO, store)
            _pg_names[pg] = group_name
        elif default_backend == Backend.NCCL:
            if not is_nccl_available():
                raise RuntimeError("Distributed package doesn't have NCCL "
                                   "built in")
            pg = ProcessGroupNCCL(store, rank, world_size)
            _pg_map[pg] = (Backend.NCCL, store)
            _pg_names[pg] = group_name
        else:
            raise RuntimeError("Unsupported distributed backend by group")
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

    default_backend, _ = _pg_map[get_default_group()]
    if (default_backend != Backend.MPI and
            group == GroupMember.NON_GROUP_MEMBER):
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
    else:
        del _pg_map[pg]
        del _pg_names[pg]
        del _pg_group_ranks[pg]


def get_rank(group=group.WORLD):
    """
    Returns the rank of currrent process group

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
    if _rank_not_in_group(group):
        return -1

    if group == GroupMember.WORLD:
        _check_default_pg()
        pg = _default_pg
    else:
        pg = group

    if src is None:
        rank_tensor = torch.IntTensor([-1])
        pg.recv_anysource([tensor], rank_tensor, tag).wait()
        src_rank = rank_tensor[0].item()
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
            operation. if ``src`` is the rank, then ``src_tensor``th element of
            ``tensor_list`` (``tensor_list[src_tensor]``) will be broadcasted
            to all other tensors (on different GPUs) in the src process and
            all tensors in ``tensor_list`` of other non-src processes.
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
            contain correctly-sized tensors on each GPU to be used for output of
            the collective.
            e.g. ``output_tensor_lists[i]`` contains the all_gather
            result that resides on the GPU of ``input_tensor_list[i]``.
            Note that each element of ``output_tensor_lists[i]`` has the size of
            ``world_size * len(input_tensor_list)``, since the function all
            gathers the result from every single GPU in the group. To interpret
            each element of ``output_tensor_list[i]``, note that
            ``input_tensor_list[j]`` of rank k will be appear in
            ``output_tensor_list[i][rank * world_size + j]``
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


def gather(tensor,
           gather_list,
           dst,
           group=group.WORLD,
           async_op=False):
    """
    Gathers a list of tensors in a single process.

    Arguments:
        tensor (Tensor): Input tensor.
        gather_list (list[Tensor]): List of appropriately-sized tensors to
            use for received data. Required only in the receiving process.
        dst (int): Destination rank. Required in all processes except the one
            that is receiveing the data.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    if _rank_not_in_group(group):
        return

    my_rank = get_rank()
    if dst == my_rank:
        if gather_list is None:
            raise RuntimeError("gather_list is a required argument in gather "
                               "destination")
    else:
        if gather_list:
            raise RuntimeError("non-empty gather_list can be given only "
                               "to gather destination")

    opts = GatherOptions()
    opts.rootRank = dst

    if group == GroupMember.WORLD:
        _check_default_pg()
        work = _default_pg.gather([gather_list], [tensor], opts)
    else:
        group_dst_rank = _get_group_rank(group, dst)
        opts.rootRank = group_dst_rank
        work = group.gather([gather_list], [tensor], opts)

    if async_op:
        return work
    else:
        work.wait()


def scatter(tensor,
            scatter_list,
            src,
            group=group.WORLD,
            async_op=False):
    """
    Scatters a list of tensors to all processes in a group.

    Each process will receive exactly one tensor and store its data in the
    ``tensor`` argument.

    Arguments:
        tensor (Tensor): Output tensor.
        scatter_list (list[Tensor]): List of tensors to scatter. Required only
            in the process that is sending the data.
        src (int): Source rank. Required in all processes except the one that
            is sending the data.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    if _rank_not_in_group(group):
        return

    my_rank = get_rank()
    if src == my_rank:
        if scatter_list is None:
            raise RuntimeError("scatter_list is a required argument in "
                               "scatter source")
    else:
        if scatter_list:
            raise RuntimeError("non-empty can be given only to scatter "
                               "source")

    opts = ScatterOptions()
    opts.rootRank = src

    if group == GroupMember.WORLD:
        _check_default_pg()
        work = _default_pg.scatter([tensor], [scatter_list], opts)
    else:
        group_src_rank = _get_group_rank(group, src)
        opts.rootRank = group_src_rank
        work = group.scatter([tensor], [scatter_list], opts)

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


def new_group(ranks=None):
    """
    Creates a new distributed group.

    This function requires that all processes in the main group (i.e. all
    processes that are part of the distributed job) enter this function, even
    if they are not going to be members of the group. Additionally, groups
    should be created in the same order in all processes.

    Arguments:
        ranks (list[int]): List of ranks of group members.

    Returns:
        A handle of distributed group that can be given to collective calls.
    """

    _check_default_pg()

    global _pg_group_ranks

    default_backend, _ = _pg_map[_default_pg]
    global_rank = _default_pg.rank()
    global_world_size = _default_pg.size()

    # checks the input ranks
    if ranks is not None:
        input_ranks = list(ranks)
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
        input_ranks = []
        ranks = list(range(global_world_size))
        group_world_size = global_world_size
        group_rank = global_rank

    if default_backend == Backend.MPI:
        in_group = global_rank in ranks
        pg = _new_process_group_helper(group_world_size,
                                       group_rank,
                                       input_ranks,
                                       in_group)
    else:
        # Release ranks not in the group
        if global_rank not in ranks:
            return GroupMember.NON_GROUP_MEMBER

        if default_backend != Backend.MPI:
            pg = _new_process_group_helper(group_world_size,
                                           group_rank,
                                           input_ranks)

    # Create the global rank to group rank mapping
    _pg_group_ranks[pg] = {}
    if default_backend == Backend.MPI:
        _pg_group_ranks[pg] = pg.group_ranks()
    else:
        for rank in range(global_world_size):
            if rank in ranks:
                _pg_group_ranks[pg][rank] = ranks.index(rank)
    return pg
