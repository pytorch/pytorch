# mypy: allow-untyped-defs
from torch.fx.proxy import Proxy
from ._compatibility import compatibility

@compatibility(is_backward_compatible=False)
def annotate(val, type):
    """
    Annotates a Proxy object with a given type.

    This function annotates a val with a given type if a type of the val is a torch.fx.Proxy object
    Args:
        val (object): An object to be annotated if its type is torch.fx.Proxy.
        type (object): A type to be assigned to a given proxy object as val.
    Returns:
        The given val.
    Raises:
        RuntimeError: If a val already has a type in its node.
    """
    if isinstance(val, Proxy):
        if val.node.type:
            raise RuntimeError(f"Tried to annotate a value that already had a type on it!"
                               f" Existing type is {val.node.type} "
                               f"and new type is {type}. "
                               f"This could happen if you tried to annotate a function parameter "
                               f"value (in which case you should use the type slot "
                               f"on the function signature) or you called "
                               f"annotate on the same value twice")
        else:
            val.node.type = type
        return val
    else:
        return val
