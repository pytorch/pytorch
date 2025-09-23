from functools import partial

from .multipledispatch import dispatch as _dispatch


namespace = {}  # type: ignore[var-annotated]

dispatch = partial(_dispatch, namespace=namespace)
