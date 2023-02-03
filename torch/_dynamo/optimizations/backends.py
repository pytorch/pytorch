import functools
import logging
import tempfile

from ..backends.registry import register_backend

from .subgraph import SubGraph

log = logging.getLogger(__name__)


def create_backend(fn):
    """
    WARNING: We do not recommend using this for new backends.  This is
    primarily used to support legacy TorchScript-based backends.
    """

    @functools.wraps(fn)
    def inner(model, example_inputs=None, **kwargs):
        if model is None:
            return None

        if not isinstance(model, SubGraph):
            with tempfile.TemporaryDirectory() as tmp:
                return inner(SubGraph(model, example_inputs, tmp), **kwargs)
        else:
            assert example_inputs is None

        try:
            return fn(model, **kwargs)
        except KeyboardInterrupt:
            raise

    return register_backend(inner)
