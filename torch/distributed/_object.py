import io
import torch


def _object_to_tensor(obj):
    """
    Serialize Python object to torch.ByteTensor.
    """
    buf = io.BytesIO()
    torch.save(obj, buf)
    return torch.ByteTensor(list(buf.getvalue()))


def _tensor_to_object(tensor):
    """
    Deserialize Python object from torch.ByteTensor.
    """
    buf = io.BytesIO(bytearray(tensor.tolist()))
    buf.seek(0)
    return torch.load(buf)


def _broadcast_object(process_group, obj=None, root=0):
    """
    Broadcast Python object from root to all processes in the process group.

    Arguments:
        process_group (ProcessGroup): The process group to use.
        obj (object, optional): The object to broadcast.
                                Must be set only on the root process.
        root (int, optional): The rank of the root process.

    Returns:
        The broadcast object. On the root process it returns a copy of
        the `obj` argument (after serialization and deserialization).

    """
    tensor = None
    length = torch.empty([1], dtype=torch.uint8)
    if process_group.rank() == root:
        tensor = _object_to_tensor(obj)
        length[0] = tensor.size(0)

    # Broadcast length of the serialized data.
    process_group.broadcast(length, root=root).wait()

    # Broadcast serialized data itself.
    if process_group.rank() != root:
        tensor = torch.empty([length[0]], dtype=torch.uint8)
    process_group.broadcast(tensor, root=root).wait()

    # Load object from serialized data.
    return _tensor_to_object(tensor)
