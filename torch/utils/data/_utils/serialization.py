import pickle

from enum import Enum

try:
    import dill

    # XXX: By default, dill writes the Pickler dispatch table to inject its
    # own logic there. This globally affects the behavior of the standard library
    # pickler for any user who transitively depends on this module!
    # Undo this extension to avoid altering the behavior of the pickler globally.
    dill.extend(use_dill=False)
    DILL_AVAILABLE = True
except ImportError:
    DILL_AVAILABLE = False


class SerializationType(Enum):
    PICKLE = "pickle"
    DILL = "dill"


def serialize_fn(fn):
    """
    If `fn` is a lambda function, use `dill` if `pickle` fails and DILL_AVAILABLE.
    Returns a tuple of serialized function and SerializationType indicating the serialization method.
    """
    if callable(fn) and fn.__name__ == "<lambda>" and DILL_AVAILABLE:
        return dill.dumps(fn), SerializationType("dill")
    else:
        return fn, SerializationType("pickle")


def deserialize_fn(serialized_fn_with_method):
    """
    Given a tuple of function and SerializationType, deserializes the function based on the given SerializationType.
    """
    serialized_fn, method = serialized_fn_with_method
    if method == SerializationType("pickle"):
        return serialized_fn
    elif method == SerializationType("dill"):
        if DILL_AVAILABLE:
            return dill.loads(serialized_fn)
        else:
            raise RuntimeError("`dill` is not available but it is needed to deserialize the function.")
    else:
        raise TypeError(f"Expect valid SerializationType in deserialize_fn, got {method} instead.")
