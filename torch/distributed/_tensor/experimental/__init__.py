from contextlib import contextmanager

from torch.distributed._tensor.api import DTensor


@contextmanager
def implicit_replication():
    try:
        DTensor._op_dispatcher._allow_implicit_replication = True
        yield
    finally:
        DTensor._op_dispatcher._allow_implicit_replication = False
