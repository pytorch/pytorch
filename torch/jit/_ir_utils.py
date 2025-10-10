from types import TracebackType
from typing import Optional, Union

import torch


class _InsertPoint:
    def __init__(
        self,
        insert_point_graph: torch._C.Graph,
        insert_point: Union[torch._C.Node, torch._C.Block],
    ):
        self.insert_point = insert_point
        self.g = insert_point_graph
        self.guard = None

    def __enter__(self) -> None:
        self.prev_insert_point = self.g.insertPoint()
        self.g.setInsertPoint(self.insert_point)

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.g.setInsertPoint(self.prev_insert_point)


def insert_point_guard(
    self: torch._C.Graph, insert_point: Union[torch._C.Node, torch._C.Block]
) -> _InsertPoint:
    return _InsertPoint(self, insert_point)
