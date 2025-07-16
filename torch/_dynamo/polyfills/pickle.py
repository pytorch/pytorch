import pickle
from typing import Any, Optional

from ..decorators import substitute_in_graph


__all__ = [
    "loads",
    "dumps",
]


@substitute_in_graph(
    pickle.dumps,
    skip_signature_check=True,
    can_constant_fold_through=True,
    graph_break_if_cannot_constant_fold=True,
)
def dumps(obj: Any, protocol: Optional[int] = None, **kwargs) -> Any:  # type: ignore[no-untyped-def]
    return pickle.dumps(obj, protocol=protocol, **kwargs)


@substitute_in_graph(
    pickle.loads,
    skip_signature_check=True,
    can_constant_fold_through=True,
    graph_break_if_cannot_constant_fold=True,
)
def loads(data, **kwargs):  # type:ignore[no-untyped-def]
    return pickle.loads(data, **kwargs)
