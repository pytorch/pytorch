# type: ignore

from multipledispatch import dispatch
from functools import partial

namespace = dict()

dispatch = partial(dispatch, namespace=namespace)