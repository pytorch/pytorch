import torch
from torch.utils._python_dispatch import TorchDispatchMode

def embedding_check(op_name, args):
    invalid_index_mask = args[0].shape[0] <= args[1]
    if torch.any(invalid_index_mask):
        msg = (f"{op_name}: Received invalid indices for embedding matrix of shape : {args[0].shape}."
                f" Invalid indices are {torch.masked_select(args[1], invalid_index_mask)}")
        raise RuntimeError(msg)

class DebugMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(func)
        if func == torch.ops.aten.embedding.default:
            embedding_check("embedding", args)
        if func == torch.ops.aten._embedding_bag.default:
            embedding_check("embedding_bag", args)

        return func(*args, **kwargs)
