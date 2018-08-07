import torch
import os


def is_available():
    return hasattr(torch._C, "_c10d_init")

if not is_available():
    raise RuntimeError("PyTorch built without distributed support")

if not torch._C._c10d_init():
    raise RuntimeError("Failed to initialize PyTorch distributed support")


from .rendezvous import rendezvous, register_rendezvous_handler
from . import BroadcastOptions, AllreduceOptions, ReduceOptions, \
    ScatterOptions, GatherOptions, ReduceOp


class DistBackend:
    UNDEFINED = -1
    GLOO = 0
    NCCL = 2
    MPI = 3


_pg_map = {}
_default_pg = None


DEFAULT_REDUCE_OPTIONS = AllreduceOptions()


def _check_default_pg(process_group):
    assert _default_pg is not None, \
        "Default process group is not initialized"


def init_process_group(backend,
                       init_method="env://"):
    """Initializes the default distributed process group.

    Arguments:
        backend (str): Name of the backend to use. Depending on build-time
                       configuration valid values include:
                       ``tcp``, ``mpi`` and ``gloo``.
        init_method (str, optional): URL specifying how to initialize the
                                     process group.

    To enable ``backend == mpi``, PyTorch needs to built from source on
    a system that supports MPI. The same applies to NCCL as well.

    """
    global _pg_map
    global _default_pg

    if _default_pg is not None:
        raise RuntimeError("trying to initialize the default process group "
                           "twice!")
    if backend == "mpi":
        _default_pg = c10d.ProcessGroupMPI()
        _pg_map[_default_pg] = (DistBackend.MPI, None)
    elif backend == "gloo":
        store, rank, world_size = next(rendezvous(init_method))
        _default_pg = c10d.ProcessGroupGloo(store, rank, world_size)
        _pg_map[_default_pg] = (DistBackend.GLOO, store)
    elif backend == "nccl":
        store, rank, world_size = next(rendezvous(init_method))
        _default_pg = c10d.ProcessGroupNCCL(store, rank, world_size)
        _pg_map[_default_pg] = (DistBackend.NCCL, store)
    else:
        raise RuntimeError("Invalid distributed backend name: " + backend)


def new_process_group(backend,
                      init_method="env://"):
    """Create a new distributed process group.

    Arguments:
        backend (str): Name of the backend to use. Depending on
                       build-time configuration
                       valid values include: ``tcp``, ``mpi`` and ``gloo``.
        init_method (str, optional): URL specifying how to initialize the
                                     process group.

    To enable ``backend == mpi``, PyTorch needs to built from source on
    a system that supports MPI. The same applies to NCCL as well.

    """
    global _pg_map

    if backend == "mpi":
        pg = c10d.ProcessGroupMPI()
        _pg_map[pg] = (DistBackend.MPI, None)
    elif backend == "gloo":
        store, rank, world_size = next(rendezvous(init_method))
        pg = c10d.ProcessGroupGloo(store, rank, world_size)
        _pg_map[pg] = (DistBackend.GLOO, store)
    elif backend == "nccl":
        store, rank, world_size = next(rendezvous(init_method))
        pg = c10d.ProcessGroupNCCL(store, rank, world_size)
        _pg_map[pg] = (DistBackend.NCCL, store)
    else:
        raise RuntimeError("Invalid distributed backend name: " + backend)
    return pg


def destroy_process_group(process_group=None):
    """
    Destroy a given process group
    """
    global _pg_map
    global _default_pg
    if process_group is None:
        pg = _default_pg
    if _pg_map.get(pg, None) is None:
        raise RuntimeError("Invalid process group specified")
    del _pg_map[pg]
    if process_group is None:
        _default_pg = None


def get_rank(process_group=None):
    """
    Returns the rank of currrent process group

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Arguments:
        process_group (ProcessGroup, optional): The process group to work on

    Returns:
        The rank of the process group

    """
    if process_group is None:
        _check_default_pg(process_group)
        return _default_pg.rank()
    return process_group.rank()


def get_world_size(process_group=None):
    """
    Returns the number of processes in the current process group

    Arguments:
        process_group (ProcessGroup, optional): The process group to work on

    Returns:
        The world size of the process group

    """
    if process_group is None:
        _check_default_pg(process_group)
        return _default_pg.size()
    return process_group.size()


def isend(tensor,
          dst,
          process_group=None):
    """Sends a tensor asynchronously.

    Arguments:
        tensor (Tensor): Tensor to send.
        dst (int): Destination rank.
        process_group (ProcessGroup, optional): The process group to work on

    Returns:
        A distributed request object.

    """
    if process_group is None:
        _check_default_pg(process_group)
        return _default_pg.send([tensor], dst)
    return process_group.send([tensor], dst)


def irecv(tensor,
          src,
          process_group=None):
    """Receives a tensor asynchronously.

    Arguments:
        tensor (Tensor): Tensor to fill with received data.
        src (int): Source rank.
        process_group (ProcessGroup, optional): The process group to work on

    Returns:
        A distributed request object.

    """
    if process_group is None:
        _check_default_pg(process_group)
        return _default_pg.recv([tensor], src)
    return process_group.recv([tensor], src)


def send(tensor,
         dst,
         process_group=None):
    """Sends a tensor synchronously.

    Arguments:
        tensor (Tensor): Tensor to send.
        dst (int): Destination rank.
        process_group (ProcessGroup, optional): The process group to work on

    """
    if process_group is None:
        _check_default_pg(process_group)
        _default_pg.send([tensor], dst).wait()
    process_group.send([tensor], dst).wait()


def recv(tensor,
         src=None,
         process_group=None):
    """Receives a tensor synchronously.

    Arguments:
        tensor (Tensor): Tensor to fill with received data.
        src (int, optional): Source rank. Will receive from any
            process if unspecified.
        process_group (ProcessGroup, optional): The process group to work on

    """
    if process_group is None:
        _check_default_pg(process_group)
        _default_pg.recv([tensor], src).wait()
    process_group.recv([tensor], src).wait()


def broadcast_multigpu(tensor_list,
                       src,
                       src_tensor=0,
                       process_group=None,
                       async_op=False):
    """Broadcasts the tensor to the whole group with multiple GPU tensors
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
        src_tensor (int): Source tensor rank within ``tensor_list``
        process_group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, otherwise

    """
    opts = BroadcastOptions()
    opts.rootRank = src
    opts.rootTensor = src_tensor

    if process_group is None:
        _check_default_pg(process_group)
        work = _default_pg.broadcast(tensor_list, opts)
    else:
        work = process_group.broadcast(tensor_list, opts)

    if async_op:
        return work
    else:
        work.wait()


def broadcast(tensor,
              src,
              process_group=None,
              async_op=False):
    """Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes
    participating in the collective.

    Arguments:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process, and tensor to be used to save received data otherwise.
        src (int): Source rank.
        process_group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op
    """
    opts = BroadcastOptions()
    opts.rootRank = src
    opts.rootTensor = 0

    if process_group is None:
        _check_default_pg(process_group)
        work = _default_pg.broadcast([tensor], opts)
    else:
        work = process_group.broadcast([tensor], opts)

    if async_op:
        return work
    else:
        work.wait()


def all_reduce_multigpu(tensor_list,
                        op=ReduceOp.SUM,
                        process_group=None,
                        async_op=False):
    """Reduces the tensor data across all machines in such a way that all get
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
            ``torch.distributed.c10d.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        process_group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, otherwise

    """
    opts = AllreduceOptions()
    opts.reduceOp = op
    if process_group is None:
        _check_default_pg(process_group)
        work = _default_pg.allreduce(tensor_list, opts)
    else:
        work = process_group.allreduce(tensor_list, opts)

    if async_op:
        return work
    else:
        work.wait()


def all_reduce(tensor,
               op=ReduceOp.SUM,
               process_group=None,
               async_op=False):
    """Reduces the tensor data across all machines in such a way that all get
    the final result.

    After the call ``tensor`` is going to be bitwise identical in all processes.

    Arguments:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        op (optional): One of the values from
            ``torch.distributed.c10d.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        process_group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, otherwise

    """
    opts = AllreduceOptions()
    opts.reduceOp = op
    if process_group is None:
        _check_default_pg(process_group)
        work = _default_pg.allreduce([tensor], opts)
    else:
        work = process_group.allreduce([tensor], opts)

    if async_op:
        return work
    else:
        work.wait()


def reduce_multigpu(tensor_list,
                    dst,
                    dst_tensor=0,
                    op=ReduceOp.SUM,
                    process_group=None,
                    async_op=False):
    """Reduces the tensor data on multiple GPUs across all machines. Each tensor
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
        dst_tensor (int): Destination tensor rank within ``tensor_list``
        op (optional): One of the values from
            ``torch.distributed.c10d.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        process_group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, otherwise

    """
    opts = ReduceOptions()
    opts.reduceOp = op
    opts.rootRank = dst
    opts.rootTensor = dst_tensor
    if process_group is None:
        _check_default_pg(process_group)
        work = _default_pg.reduce(tensor_list, opts)
    else:
        work = process_group.reduce(tensor_list, opts)

    if async_op:
        return work
    else:
        work.wait()


def reduce(tensor,
           dst,
           op=ReduceOp.SUM,
           process_group=None,
           async_op=False):
    """Reduces the tensor data across all machines.

    Only the process with rank ``dst`` is going to receive the final result.

    Arguments:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        dst (int): Destination rank
        op (optional): One of the values from
            ``torch.distributed.c10d.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        process_group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, otherwise

    """
    opts = ReduceOptions()
    opts.reduceOp = op
    opts.rootRank = dst
    if process_group is None:
        _check_default_pg(process_group)
        work = _default_pg.reduce([tensor], opts)
    else:
        work = process_group.reduce([tensor], opts)

    if async_op:
        return work
    else:
        work.wait()


def all_gather_multigpu(output_tensor_lists,
                        input_tensor_list,
                        process_group=None,
                        async_op=False):
    """Gathers tensors from the whole group in a list.
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

        process_group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, otherwise

    """
    if process_group is None:
        _check_default_pg(process_group)
        work = _default_pg.allgather(output_tensor_lists, input_tensor_list)
    else:
        work = process_group.allgather(output_tensor_lists, input_tensor_list)

    if async_op:
        return work
    else:
        work.wait()


def all_gather(tensor_list,
               tensor,
               process_group=None,
               async_op=False):
    """Gathers tensors from the whole group in a list.

    Arguments:
        tensor_list (list[Tensor]): Output list. It should contain
            correctly-sized tensors to be used for output of the collective.
        tensor (Tensor): Tensor to be broadcast from current process.
        process_group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, otherwise

    """
    if process_group is None:
        _check_default_pg(process_group)
        work = _default_pg.allgather([tensor_list], [tensor])
    else:
        work = process_group.allgather([tensor_list], [tensor])

    if async_op:
        return work
    else:
        work.wait()


def gather(tensor,
           gather_list,
           dst,
           process_group=None,
           async_op=False):
    """Gathers a list of tensors in a single process.

    Arguments:
        tensor (Tensor): Input tensor.
        gather_list (list[Tensor]): List of appropriately-sized tensors to
            use for received data. Required only in the receiving process.
        dst (int): Destination rank. Required in all processes except the one
            that is receiveing the data.
        process_group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, otherwise

    """
    opts = GatherOptions()
    opts.rootRank = dst

    if process_group is None:
        _check_default_pg(process_group)
        work = _default_pg.gather([tensor_list], [tensor], opts)
    else:
        work = process_group.gather([tensor_list], [tensor], opts)

    if async_op:
        return work
    else:
        work.wait()


def scatter(tensor,
            scatter_list,
            src,
            process_group=None,
            async_op=False):
    """Scatters a list of tensors to all processes in a group.

    Each process will receive exactly one tensor and store its data in the
    ``tensor`` argument.

    Arguments:
        tensor (Tensor): Output tensor.
        scatter_list (list[Tensor]): List of tensors to scatter. Required only
            in the process that is sending the data.
        src (int): Source rank. Required in all processes except the one that
            is sending the data.
        process_group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, otherwise

    """
    opts = ScatterOptions()
    opts.rootRank = src

    if process_group is None:
        _check_default_pg(process_group)
        work = _default_pg.scatter([tensor], [scatter_list], opts)
    else:
        work = process_group.scatter([tensor], [scatter_list], opts)

    if async_op:
        return work
    else:
        work.wait()


def barrier(process_group=None,
            async_op=False):
    """Synchronizes all processes.

    This collective blocks processes until the whole group enters this function,
    if async_op is False, or if async work handle is called on wait().

    Arguments:
        process_group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, otherwise
    """
    if process_group is None:
        _check_default_pg(process_group)
        work = _default_pg.barrier()
    else:
        work = process_group.barrier()

    if async_op:
        return work
    else:
        work.wait()


def _broadcast(tensor, src, process_group):
    opts = BroadcastOptions()
    opts.reduceOp = op
    return process_group.broadcast([tensor], opts)


def _all_reduce(tensor, process_group, opts=DEFAULT_REDUCE_OPTIONS):
    return process_group.allreduce([tensor], opts)
