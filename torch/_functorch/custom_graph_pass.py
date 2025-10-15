import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Callable, Optional, Union
from typing_extensions import TypeAlias

import torch.fx

class JointCustomPass(ABC):
    """
    Implement this interface for custom Joint Graph passes:

    1) The __call__() method contains the implementation of the custom pass.

    2) The uuid() method enables inductor to cache compiled graphs when your custom
    passes are applied. This method can return any identifier as long as it uniquely
    identifies your implementation (and can be pickled). The caching logic includes this
    identifier in its key calculation, i.e., any new value will effectively invalidate
    existing entries. We expect custom passes would typically depend purely on the
    textual representation of the implementation. In that case, we recommend using the
    'get_hash_for_files' helper below to compute a unique hash from the contents of a
    static list of source files, i.e., the source(s) containing the custom pass
    implementation. That approach ensures that any change to the implementation will
    mean a new uuid.
    """

    @abstractmethod
    def __call__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: tuple[list[torch.Tensor], list[torch.Tensor]],
    ) -> torch.fx.GraphModule:
        """
        Implementation of the custom pass for joint graph.
        """

    @abstractmethod
    def uuid(self) -> Optional[Any]:
        """
        Return an ID to uniquely identify your custom pass implementation. Return None
        to skip inductor code caching entirely.
        """


JointCustomPassType: TypeAlias = Optional[
    Union[
        JointCustomPass,
        Callable[
            [torch.fx.GraphModule, tuple[list[torch.Tensor], list[torch.Tensor]]],
            torch.fx.GraphModule,
        ],
    ]
]
