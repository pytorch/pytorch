from functools import partial

from .multipledispatch import dispatch


namespace = {}  # type: ignore[var-annotated]

dispatch = partial(dispatch, namespace=namespace)
