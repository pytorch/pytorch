from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Optional
from typing_extensions import TypeAlias

import torch


class CustomPartitionerFn(ABC):
    """
    Implement this interface for custom partitioning functions:

    1) The __call__() method contains the implementation of the custom partitioning function.

    2) The uuid() method enables inductor to cache compiled graphs when your custom
    partitioning function is applied. For more details, see a similar customization idea
    in torch/_inductor/custom_graph_pass.py.

    EXAMPLE:

    from torch._inductor.custom_graph_pass import get_hash_for_files

    class MyCustomPartitionerFn(CustomPartitionerFn):
        def __call__(
            self,
            gm: torch.fx.GraphModule,
            joint_inputs: Sequence[object],
            **kwargs: object
        ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
            # my custom partitioning implementation
            #     ...

        def uuid(self) -> Optional[Any]:
            return get_hash_for_files((__file__,))

    """

    @abstractmethod
    def __call__(
        self, gm: torch.fx.GraphModule, joint_inputs: Sequence[object], **kwargs: object
    ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        """
        Implementation of the custom partitioning function.
        """

    @abstractmethod
    def uuid(self) -> Optional[Any]:
        """
        Return an ID to uniquely identify your custom partitioning function implementation.
        Return None to skip inductor code caching entirely.
        """


CustomPartitionerFnType: TypeAlias = Optional[CustomPartitionerFn]
