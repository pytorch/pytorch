import hashlib
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Tuple, Union

import torch.fx.graph


class PostGradCustomPass(ABC):
    """
    Implement this interface for post grad custom passes:

    1) The __call__() method contains the implementation of the custom pass.

    2) The uuid() method is necessary to support inductor caching compiled graphs when
    your custom passes are applied. This method can return any unique identifier, as
    long as it uniquely identifies your custom pass implementation. The caching logic
    includes the ID in its key calculation, i.e., a new value will effectively
    invalidate any existing entries. Most custom passes should depend purely on the
    textual reprensentation of the code. In that case, we recommend using the
    'get_hash_for_files` helper to compute a unique string based on the contents of a
    list of source files, i.e., the source containing the custom pass
    implementation. That ensures that any update to the code will map to new cache
    entries. If your custom pass behaves differently based on some external state, then
    you'll need to implement something more complicated (or disable caching).
    """

    @abstractmethod
    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        """
        Implementation of the custom pass.
        """

    @abstractmethod
    def uuid(self) -> Union[bytes, str]:
        """
        Return an ID to uniquely identify your custom pass implementation. Return None
        to bypass inductor code caching.
        """


@lru_cache(1)
def get_hash_for_files(paths: Tuple[str], extra: str = "") -> bytes:
    """
    Helper to compute a unique string by hashing the contents of a list of files.
    """
    hasher = hashlib.sha256()
    hasher.update(extra.encode("utf-8"))
    for path in paths:
        with open(path, "rb") as f:
            hasher.update(path.encode("utf-8"))
            hasher.update(f.read())
    return hasher.digest()
