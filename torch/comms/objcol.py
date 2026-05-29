# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

import functools
import io
import os
import pickle
from datetime import timedelta
from typing import Any

import torch
from torch._C._comms import TorchComm
from torch.monitor import _WaitCounter


__all__ = [
    "all_gather_object",
    "broadcast_object_list",
    "gather_object",
    "recv_object_list",
    "scatter_object_list",
    "send_object_list",
]


class _Serialization:
    """Serialization helper with serialize and deserialize methods."""

    def __init__(self) -> None:
        self.use_pickle: bool = os.getenv("TORCHCOMMS_SERIALIZATION") == "pickle"

    def serialize(self, f: io.BytesIO, obj: object) -> None:
        if self.use_pickle:
            pickle.Pickler(f).dump(obj)
        else:
            torch.save(obj, f)

    def deserialize(self, f: io.BytesIO, weights_only: bool) -> object:
        if self.use_pickle:
            return pickle.Unpickler(f).load()
        else:
            return torch.load(f, weights_only=weights_only)


@functools.cache
def _get_serialization() -> _Serialization:
    """Returns a cached serialization object with serialize and deserialize methods."""
    return _Serialization()


def _object_to_tensor(
    obj: object, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    with _WaitCounter("pytorch.wait_counter.torchcomms._object_to_tensor").guard():
        f = io.BytesIO()
        serialization = _get_serialization()
        serialization.serialize(f, obj)
        byte_storage = torch.ByteStorage._from_buffer(f.getvalue())  # type: ignore[attr-defined]
        # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor and specifying dtype.
        # Otherwise, it will cause 100X slowdown.
        # See: https://github.com/pytorch/pytorch/issues/65696
        byte_tensor = torch.ByteTensor(byte_storage).to(device)
        local_size = torch.LongTensor([byte_tensor.numel()]).to(device)
        return byte_tensor, local_size


def _tensor_to_object(
    tensor: torch.Tensor, tensor_size: int | torch.Tensor, weights_only: bool
) -> object:
    with _WaitCounter("pytorch.wait_counter.torchcomms._tensor_to_object").guard():
        tensor = tensor.cpu()
        buf = tensor.numpy().tobytes()[:tensor_size]
        serialization = _get_serialization()
        return serialization.deserialize(io.BytesIO(buf), weights_only=weights_only)


def all_gather_object(
    comm: TorchComm,
    object_list: list[Any],
    obj: object,
    timeout: timedelta | None = None,
    weights_only: bool = True,
) -> None:
    """
    Gathers picklable objects from the whole comm into a list.

    Similar to :func:`all_gather`, but Python objects can be passed in.
    Note that the object must be picklable in order to be gathered.

    Args:
        comm: The comm to work on.
        object_list (list[object]): Output list. It should be correctly sized as the
            size of the comm for this collective and will contain the output.
        obj (object): Picklable Python object to be broadcast from current process.
        timeout: (timedelta, optional): Timeout for collective operations. If
            ``None``, will use the default timeout for the backend.
        weights_only (bool, optional): If ``True``, only safe objects such as
            weights are allowed to be deserialized.
            https://docs.pytorch.org/docs/stable/notes/serialization.html#weights-only

    Returns:
        None. If the calling rank is part of this comm, the output of the
        collective will be populated into the input ``object_list``. If the
        calling rank is not part of the comm, the passed in ``object_list`` will
        be unmodified.

    .. note:: Note that this API differs slightly from the :func:`all_gather`
        collective since it does not provide an ``async_op`` handle and thus
        will be a blocking call.

    .. note:: For NCCL-based processed comms, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. warning::
        Object collectives have a number of serious performance and scalability
        limitations.

    .. warning::
        :func:`all_gather_object` uses ``pickle`` module implicitly, which is
        known to be insecure. It is possible to construct malicious pickle data
        which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`all_gather_object` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`all_gather` instead.

    Example::
        >>> # xdoctest: +SKIP("need comm init")
        >>> # Note: comm initialization omitted on each rank.
        >>> from torch.comms import objcol
        >>> # Assumes world_size of 3.
        >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object
        >>> output = [None for _ in gather_objects]
        >>> objcol.all_gather_object(comm, output, gather_objects[comm.get_rank()])
        >>> output
        ['foo', 12, {1: 2}]
    """

    current_device = comm.get_device()
    input_tensor, local_size = _object_to_tensor(obj, current_device)

    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    comm_size = comm.get_size()
    object_sizes_tensor = torch.zeros(
        comm_size, dtype=torch.long, device=current_device
    )
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(comm_size)
    ]
    # Allgather tensor sizes
    comm.all_gather(object_size_list, local_size, async_op=False, timeout=timeout)
    max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    coalesced_output_tensor = torch.empty(
        max_object_size * comm_size, dtype=torch.uint8, device=current_device
    )
    # Output tensors are nonoverlapping views of coalesced_output_tensor
    output_tensors = [
        coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
        for i in range(comm_size)
    ]
    comm.all_gather(output_tensors, input_tensor, async_op=False, timeout=timeout)
    # Deserialize outputs back to object.
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        tensor_size = object_size_list[i]
        object_list[i] = _tensor_to_object(
            tensor, tensor_size, weights_only=weights_only
        )


def gather_object(
    comm: TorchComm,
    obj: object,
    root: int,
    object_gather_list: list[Any] | None = None,
    timeout: timedelta | None = None,
    weights_only: bool = True,
) -> None:
    """
    Gathers picklable objects from the whole comm in a single process.

    Similar to :func:`gather`, but Python objects can be passed in. Note that the
    object must be picklable in order to be gathered.

    Args:
        comm: The comm to work on.
        obj (object): Input object. Must be picklable.
        object_gather_list (list[object]): Output list. On the ``root`` rank, it
            should be correctly sized as the size of the comm for this
            collective and will contain the output. Must be ``None`` on non-root
            ranks. (default is ``None``)
        root (int): Destination rank on ``comm``.
        timeout: (timedelta, optional): Timeout for collective operations. If
            ``None``, will use the default timeout for the backend.
        weights_only (bool, optional): If ``True``, only safe objects such as
            weights are allowed to be deserialized.
            https://docs.pytorch.org/docs/stable/notes/serialization.html#weights-only

    Returns:
        None. On the ``root`` rank, ``object_gather_list`` will contain the
        output of the collective.

    .. note:: Note that this API differs slightly from the gather collective
        since it does not provide an async_op handle and thus will be a blocking
        call.

    .. note:: For NCCL-based processed comms, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. warning::
        Object collectives have a number of serious performance and scalability
        limitations.

    .. warning::
        :func:`gather_object` uses ``pickle`` module implicitly, which is
        known to be insecure. It is possible to construct malicious pickle data
        which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`gather_object` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`gather` instead.

    Example::
        >>> # xdoctest: +SKIP("need comm init")
        >>> # Note: comm initialization omitted on each rank.
        >>> from torch.comms import objcol
        >>> # Assumes world_size of 3.
        >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object
        >>> output = [None for _ in gather_objects]
        >>> objcol.gather_object(
        ...     comm,
        ...     gather_objects[comm.get_rank()],
        ...     root=0,
        ...     object_gather_list=output,
        ... )
        >>> # On rank 0
        >>> output
        ['foo', 12, {1: 2}]
    """
    # Ensure object_gather_list is specified appropriately.
    my_comm_rank = comm.get_rank()
    current_device = comm.get_device()
    input_tensor, local_size = _object_to_tensor(obj, current_device)

    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    comm_size = comm.get_size()
    object_sizes_tensor = torch.zeros(
        comm_size, dtype=torch.long, device=current_device
    )
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(comm_size)
    ]
    # Allgather tensor sizes. An all-gather is needed here despite this being a
    # gather, since each rank needs to broadcast a tensor of the same (maximal)
    # size.
    comm.all_gather(object_size_list, local_size, async_op=False, timeout=timeout)
    max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)

    coalesced_output_tensor = torch.empty(
        max_object_size * comm_size, dtype=torch.uint8, device=current_device
    )
    # Output tensors are nonoverlapping views of coalesced_output_tensor
    output_tensors = [
        coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
        for i in range(comm_size)
    ]
    # All ranks call gather with equal-sized tensors.
    comm.gather(
        input_tensor=input_tensor,
        output_tensor_list=output_tensors,
        root=root,
        async_op=False,
        timeout=timeout,
    )
    if my_comm_rank != root:
        return

    assert object_gather_list is not None, (
        "Must provide object_gather_list on root rank"
    )
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        tensor_size = object_size_list[i]
        object_gather_list[i] = _tensor_to_object(
            tensor, tensor_size, weights_only=weights_only
        )


def send_object_list(
    comm: TorchComm,
    object_list: list[Any],
    dst: int,
    timeout: timedelta | None = None,
) -> None:
    """
    Sends picklable objects in ``object_list`` synchronously.

    Similar to :func:`send`, but Python objects can be passed in.
    Note that all objects in ``object_list`` must be picklable in order to be
    sent.

    Args:
        comm: The comm to work on.
        object_list (List[object]): List of input objects to sent.
            Each object must be picklable. Receiver must provide lists of equal sizes.
        dst (int): Destination rank to send ``object_list`` to.
        timeout: (timedelta, optional): Timeout for collective operations. If
            ``None``, will use the default timeout for the backend.
    Returns:
        ``None``.

    .. note:: For NCCL-based comms, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. warning::
        Object collectives have a number of serious performance and scalability
        limitations.

    .. warning::
        :func:`send_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`send_object_list` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`send` instead.

    Example::
        >>> # xdoctest: +SKIP("need comm init")
        >>> # Note: comm initialization omitted on each rank.
        >>> from torch.comms import objcol
        >>> # Assumes backend is not NCCL
        >>> if comm.get_rank() == 0:
        >>>     # Assumes world_size of 2.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>>     objcol.send_object_list(comm, objects, dst=1)
        >>> else:
        >>>     objects = [None, None, None]
        >>>     objcol.recv_object_list(comm, objects, src=0)
        >>> objects
        ['foo', 12, {1: 2}]
    """
    # Current device selection.
    # To preserve backwards compatibility, ``device`` is default to ``None``
    # in which case we run current logic of device selection, i.e.
    # ``current_device`` is CUDA if backend is NCCL otherwise CPU device. In the
    # case it is not ``None`` we move the size and object tensors to be
    # sent to this device.
    current_device = comm.get_device()
    # Serialize object_list elements to tensors on src rank.
    tensor_list, size_list = zip(
        *[_object_to_tensor(obj, current_device) for obj in object_list]
    )
    object_sizes_tensor = torch.cat(size_list)

    # Send object sizes
    comm.send(object_sizes_tensor, dst=dst, async_op=False, timeout=timeout)

    # Concatenate and send serialized object tensors
    # Note: torch.cat will do an extra memory copy to the current device, if the tensor_list
    # has only one element, we can skip the copy.
    if len(tensor_list) == 1:  # type: ignore[possibly-undefined]
        object_tensor = tensor_list[0]
    else:
        object_tensor = torch.cat(tensor_list)

    comm.send(object_tensor, dst=dst, async_op=False, timeout=timeout)


def recv_object_list(
    comm: TorchComm,
    object_list: list[Any],
    src: int,
    timeout: timedelta | None = None,
    weights_only: bool = True,
) -> None:
    """
    Receives picklable objects in ``object_list`` synchronously.

    Similar to :func:`recv`, but can receive Python objects.

    Args:
        comm: The comm to work on.
        object_list (List[object]): List of objects to receive into.
            Must provide a list of sizes equal to the size of the list being sent.
        src (int): Source rank from which to recv ``object_list``.
        timeout: (timedelta, optional): Timeout for collective operations. If
            ``None``, will use the default timeout for the backend.
        weights_only (bool, optional): If ``True``, only safe objects such as
            weights are allowed to be deserialized.
            https://docs.pytorch.org/docs/stable/notes/serialization.html#weights-only

    Returns: None

    .. note:: For NCCL-based comms, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. warning::
        Object collectives have a number of serious performance and scalability
        limitations.

    .. warning::
        :func:`recv_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`recv_object_list` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`recv` instead.

    Example::
        >>> # xdoctest: +SKIP("need comm init")
        >>> # Note: comm initialization omitted on each rank.
        >>> from torch.comms import objcol
        >>> # Assumes backend is not NCCL
        >>> if comm.get_rank() == 0:
        >>>     # Assumes world_size of 2.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>>     objcol.send_object_list(comm, objects, dst=1)
        >>> else:
        >>>     objects = [None, None, None]
        >>>     objcol.recv_object_list(comm, objects, src=0)
        >>> objects
        ['foo', 12, {1: 2}]
    """
    current_device = comm.get_device()
    object_sizes_tensor = torch.empty(
        len(object_list), dtype=torch.long, device=current_device
    )

    # Receive object sizes
    comm.recv(object_sizes_tensor, src=src, async_op=False, timeout=timeout)

    # Tensor to receive serialized objects into.
    object_tensor = torch.empty(  # type: ignore[call-overload]
        torch.sum(object_sizes_tensor).item(),  # type: ignore[arg-type]
        dtype=torch.uint8,
        device=current_device,
    )

    comm.recv(object_tensor, src=src, async_op=False, timeout=timeout)
    # Deserialize objects using their stored sizes.
    offset = 0
    for i, obj_size in enumerate(object_sizes_tensor):
        obj_view = object_tensor[offset : offset + obj_size]
        obj_view = obj_view.type(torch.uint8)
        offset += obj_size
        object_list[i] = _tensor_to_object(
            obj_view, obj_size, weights_only=weights_only
        )


def broadcast_object_list(
    comm: TorchComm,
    object_list: list[Any],
    root: int,
    timeout: timedelta | None = None,
    weights_only: bool = True,
) -> None:
    """
    Broadcasts picklable objects in ``object_list`` to the whole comm.

    Similar to :func:`broadcast`, but Python objects can be passed in.
    Note that all objects in ``object_list`` must be picklable in order to be
    broadcast.

    Args:
        comm: The comm to work on.
        object_list (List[object]): List of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank will
            be broadcast, but each rank must provide lists of equal sizes.
        root (int): Source rank from which to broadcast ``object_list``.
        timeout: (timedelta, optional): Timeout for collective operations. If
            ``None``, will use the default timeout for the backend.
        weights_only (bool, optional): If ``True``, only safe objects such as
            weights are allowed to be deserialized.
            https://docs.pytorch.org/docs/stable/notes/serialization.html#weights-only

    Returns:
        ``None``. If rank is part of the comm, ``object_list`` will contain the
        broadcasted objects from ``src`` rank.

    .. note:: For NCCL-based comms, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. note:: Note that this API differs slightly from the :func:`broadcast`
        collective since it does not provide an ``async_op`` handle and thus
        will be a blocking call.

    .. warning::
        Object collectives have a number of serious performance and scalability
        limitations.

    .. warning::
        :func:`broadcast_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`broadcast_object_list` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`broadcast` instead.

    Example::
        >>> # xdoctest: +SKIP("need comm init")
        >>> # Note: comm initialization omitted on each rank.
        >>> from torch.comms import objcol
        >>> if comm.get_rank() == 0:
        >>>     # Assumes world_size of 3.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>> else:
        >>>     objects = [None, None, None]
        >>> # Assumes backend is not NCCL
        >>> objcol.broadcast_object_list(comm, objects, src=0, device=device)
        >>> objects
        ['foo', 12, {1: 2}]
    """
    current_device = comm.get_device()
    my_comm_rank = comm.get_rank()
    # Serialize object_list elements to tensors on src rank.
    if my_comm_rank == root:
        tensor_list, size_list = zip(
            *[_object_to_tensor(obj, current_device) for obj in object_list]
        )
        object_sizes_tensor = torch.cat(size_list)
    else:
        object_sizes_tensor = torch.empty(
            len(object_list), dtype=torch.long, device=current_device
        )

    # Broadcast object sizes
    comm.broadcast(object_sizes_tensor, root=root, async_op=False, timeout=timeout)

    # Concatenate and broadcast serialized object tensors
    # Note: torch.cat will do an extra memory copy to the current device, if the tensor_list
    # has only one element, we can skip the copy.
    if my_comm_rank == root:
        if len(tensor_list) == 1:  # type: ignore[possibly-undefined]
            object_tensor = tensor_list[0]  # pyre-fixme[61]
        else:
            object_tensor = torch.cat(tensor_list)  # pyre-fixme[61]
    else:
        object_tensor = torch.empty(  # type: ignore[call-overload]
            torch.sum(object_sizes_tensor).item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
            device=current_device,
        )

    comm.broadcast(object_tensor, root=root, async_op=False, timeout=timeout)
    # Deserialize objects using their stored sizes.
    offset = 0
    if my_comm_rank != root:
        for i, obj_size in enumerate(object_sizes_tensor):
            obj_view = object_tensor[offset : offset + obj_size]
            obj_view = obj_view.type(torch.uint8)
            offset += obj_size
            object_list[i] = _tensor_to_object(
                obj_view, obj_size, weights_only=weights_only
            )


def scatter_object_list(
    comm: TorchComm,
    root: int,
    scatter_object_output_list: list[Any],
    scatter_object_input_list: list[Any] | None = None,
    timeout: timedelta | None = None,
    weights_only: bool = True,
) -> None:
    """
    Scatters picklable objects in ``scatter_object_input_list`` to the whole comm.

    Similar to :func:`scatter`, but Python objects can be passed in. On
    each rank, the scattered object will be stored as the first element of
    ``scatter_object_output_list``. Note that all objects in
    ``scatter_object_input_list`` must be picklable in order to be scattered.

    Args:
        comm: The comm to work on.
        scatter_object_output_list (List[object]): Non-empty list whose first
            element will store the object scattered to this rank.
        scatter_object_input_list (List[object], optional): List of input objects to scatter.
            Each object must be picklable. Only objects on the ``root`` rank will
            be scattered, and the argument can be ``None`` for non-root ranks.
        root (int): Source rank from which to scatter ``scatter_object_input_list``.
        timeout: (timedelta, optional): Timeout for collective operations. If
            ``None``, will use the default timeout for the backend.
        weights_only (bool, optional): If ``True``, only safe objects such as
            weights are allowed to be deserialized.
            https://docs.pytorch.org/docs/stable/notes/serialization.html#weights-only

    Returns:
        ``None``. If rank is part of the comm, ``scatter_object_output_list``
        will have its first element set to the scattered object for this rank.

    .. note:: Note that this API differs slightly from the scatter collective
        since it does not provide an ``async_op`` handle and thus will be a
        blocking call.

    .. warning::
        Object collectives have a number of serious performance and scalability
        limitations.

    .. warning::
        :func:`scatter_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`scatter_object_list` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`scatter` instead.

    Example::
        >>> # xdoctest: +SKIP("need comm init")
        >>> # Note: comm initialization omitted on each rank.
        >>> from torch.comms import objcol
        >>> if comm.get_rank() == 0:
        >>>     # Assumes world_size of 3.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>> else:
        >>>     # Can be any list on non-root ranks, elements are not used.
        >>>     objects = [None, None, None]
        >>> output_list = [None]
        >>> objcol.scatter_object_list(comm, output_list, objects, root=0)
        >>> # Rank i gets objects[i]. For example, on rank 2:
        >>> output_list
        [{1: 2}]
    """

    if (
        not isinstance(scatter_object_output_list, list)
        or len(scatter_object_output_list) < 1
    ):
        raise ValueError(
            "Expected argument scatter_object_output_list to be a list of size at least 1."
        )

    my_comm_rank = comm.get_rank()
    current_device = comm.get_device()
    if my_comm_rank == root:
        if scatter_object_input_list is None:
            raise ValueError(
                "source rank must provide non-None scatter_object_input_list"
            )
        tensor_list, tensor_sizes = zip(
            *[
                _object_to_tensor(obj, current_device)
                for obj in scatter_object_input_list
            ]
        )
        tensor_list, tensor_sizes = list(tensor_list), list(tensor_sizes)

        # root rank broadcasts the maximum tensor size. This is because all ranks are
        # expected to call into scatter() with equal-sized tensors.
        max_tensor_size = max(tensor_sizes)  # type: ignore[possibly-undefined]
        for tensor in tensor_list:  # type: ignore[possibly-undefined]
            tensor.resize_(max_tensor_size)
    else:
        max_tensor_size = torch.tensor([0], dtype=torch.long, device=current_device)
    comm.broadcast(max_tensor_size, root=root, async_op=False, timeout=timeout)

    # Scatter actual serialized objects
    output_tensor = torch.empty(  # pyrefly: ignore[no-matching-overload]
        max_tensor_size.item(), dtype=torch.uint8, device=current_device
    )
    comm.scatter(
        output_tensor,
        input_tensor_list=[] if my_comm_rank != root else tensor_list,  # type: ignore[possibly-undefined]
        root=root,
        async_op=False,
        timeout=timeout,
    )

    # Scatter per-object sizes to trim tensors when deserializing back to object
    obj_tensor_size = torch.tensor([0], dtype=torch.long, device=current_device)
    comm.scatter(
        obj_tensor_size,
        input_tensor_list=[] if my_comm_rank != root else tensor_sizes,  # type: ignore[possibly-undefined]
        root=root,
        async_op=False,
        timeout=timeout,
    )

    # Deserialize back to object
    scatter_object_output_list[0] = _tensor_to_object(
        output_tensor,
        obj_tensor_size,
        weights_only=weights_only,
    )
