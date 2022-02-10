from functools import partial
from .multipledispatch import dispatch  # type: ignore[import]

namespace = dict()  # type: ignore[var-annotated]

dispatch = partial(dispatch, namespace=namespace)
