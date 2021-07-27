from multipledispatch import dispatch  # type: ignore[import]
from functools import partial

namespace = dict()  # type: ignore[var-annotated]

dispatch = partial(dispatch, namespace=namespace)
