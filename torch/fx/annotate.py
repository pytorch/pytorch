from torch.fx.proxy import Proxy
from ._compatibility import compatibility

@compatibility(is_backward_compatible=False)
def annotate(val, type):
    # val could be either a regular value (not tracing)
    # or fx.Proxy (tracing)
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
