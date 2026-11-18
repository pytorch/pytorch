from functools import partial

from .multipledispatch import dispatch as _dispatch  # type: ignore[import]


namespace = {}  # type: ignore[var-annotated]

dispatch = partial(_dispatch, namespace=namespace)
