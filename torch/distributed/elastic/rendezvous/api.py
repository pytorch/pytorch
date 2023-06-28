# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

from torch.distributed import Store


class RendezvousError(Exception):
    """Represents the base type for rendezvous errors."""


class RendezvousClosedError(RendezvousError):
    """Raised when a rendezvous is closed."""


class RendezvousTimeoutError(RendezvousError):
    """Raised when a rendezvous did not complete on time."""


class RendezvousConnectionError(RendezvousError):
    """Raised when the connection to a rendezvous backend has failed."""


class RendezvousStateError(RendezvousError):
    """Raised when the state of a rendezvous is corrupt."""


class RendezvousHandler(ABC):
    """Main rendezvous interface.

    Note:
        Distributed Torch users normally **do not** need to implement their own
        ``RendezvousHandler``. An implementation based on C10d Store is already
        provided, and is recommended for most users.
    """

    @abstractmethod
    def get_backend(self) -> str:
        """Returns the name of the rendezvous backend."""

    @abstractmethod
    def next_rendezvous(
        self,
    ) -> Tuple[Store, int, int]:
        """Main entry-point into the rendezvous barrier.

        Blocks until the rendezvous is complete and the current process is
        included in the formed worker group, or a timeout occurs, or the
        rendezvous was marked closed.

        Returns:
            A tuple of :py:class:`torch.distributed.Store`, ``rank``, and
            ``world size``.

        Raises:
            RendezvousClosedError:
                The rendezvous is closed.
            RendezvousConnectionError:
                The connection to the rendezvous backend has failed.
            RendezvousStateError:
                The rendezvous state is corrupt.
            RendezvousTimeoutError:
                The rendezvous did not complete on time.
        """

    @abstractmethod
    def is_closed(self) -> bool:
        """Checks whether the rendezvous has been closed.

        A closed rendezvous means all future attempts to re-rendezvous within
        same job will fail.

        ``is_closed()`` and :py:meth:`set_closed` have semantics of eventual
        propagation and should not be used for synchronization. The intention is
        that if at least one node decides the job is finished, it will close the
        rendezvous, and other nodes will soon observe this and stop running as
        well.
        """

    @abstractmethod
    def set_closed(self):
        """Marks the rendezvous as closed."""

    @abstractmethod
    def num_nodes_waiting(self) -> int:
        """Returns the number of nodes who arrived late at the rendezvous
        barrier, hence were not included in the current worker group.

        Callers should periodically call this method to check whether new
        nodes are waiting to join the job and if so admit them by calling
        :py:meth:`next_rendezvous()` (re-rendezvous).
        """

    @abstractmethod
    def get_run_id(self) -> str:
        """Returns the run id of the rendezvous.

        The run id is a user-defined id that uniquely identifies an instance of
        a distributed application. It typically maps to a job id and is used to
        allow nodes to join the correct distributed application.
        """

    @abstractmethod
    def shutdown(self) -> bool:
        """Closes all resources that were open for the rendezvous.

        Example::

            rdzv_handler = ...
            try:
                store, rank, world_size = rdzv_handler.next_rendezvous()
            finally:
                rdzv_handler.shutdown()
        """


class RendezvousParameters:
    """Holds the parameters to construct a :py:class:`RendezvousHandler`.

    Args:
        backend:
            The name of the backend to use to handle the rendezvous.
        endpoint:
            The endpoint of the rendezvous, usually in form <hostname>[:<port>].
        run_id:
            The id of the rendezvous.
        min_nodes:
            The minimum number of nodes to admit to the rendezvous.
        max_nodes:
            The maximum number of nodes to admit to the rendezvous.
        local_addr:
            The address of the local node.
        **kwargs:
            Additional parameters for the specified backend.
    """

    def __init__(
        self,
        backend: str,
        endpoint: str,
        run_id: str,
        min_nodes: int,
        max_nodes: int,
        local_addr: Optional[str] = None,
        **kwargs,
    ):
        if not backend:
            raise ValueError("The rendezvous backend name must be a non-empty string.")

        if min_nodes < 1:
            raise ValueError(
                f"The minimum number of rendezvous nodes ({min_nodes}) must be greater than zero."
            )
        if max_nodes < min_nodes:
            raise ValueError(
                f"The maximum number of rendezvous nodes ({max_nodes}) must be greater than or "
                f"equal to the minimum number of rendezvous nodes ({min_nodes})."
            )

        self.backend = backend
        self.endpoint = endpoint
        self.run_id = run_id
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.config = kwargs
        self.local_addr = local_addr

    def get(self, key: str, default: Any = None) -> Any:
        """Returns the value for ``key`` if ``key`` exists, else ``default``."""
        return self.config.get(key, default)

    def get_as_bool(self, key: str, default: Optional[bool] = None) -> Optional[bool]:
        """Returns the value for ``key`` as a ``bool``."""
        value = self.get(key, default)
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            if value == 1:
                return True
            if value == 0:
                return False
        elif isinstance(value, str):
            if value.lower() in ["1", "true", "t", "yes", "y"]:
                return True
            if value.lower() in ["0", "false", "f", "no", "n"]:
                return False
        raise ValueError(
            f"The rendezvous configuration option '{key}' does not represent a valid boolean value."
        )

    def get_as_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """Returns the value for ``key`` as an ``int``."""
        value = self.get(key, default)
        if value is None:
            return value
        try:
            return int(value)
        except ValueError as e:
            raise ValueError(
                f"The rendezvous configuration option '{key}' does not represent a valid integer "
                "value."
            ) from e


RendezvousHandlerCreator = Callable[[RendezvousParameters], RendezvousHandler]


class RendezvousHandlerRegistry:
    """Represents a registry of :py:class:`RendezvousHandler` backends."""

    _registry: Dict[str, RendezvousHandlerCreator]

    def __init__(self) -> None:
        self._registry = {}

    def register(self, backend: str, creator: RendezvousHandlerCreator) -> None:
        """Registers a new rendezvous backend.

        Args:
            backend:
                The name of the backend.
            creator:
                The callback to invoke to construct the
                :py:class:`RendezvousHandler`.
        """
        if not backend:
            raise ValueError("The rendezvous backend name must be a non-empty string.")

        current_creator: Optional[RendezvousHandlerCreator]
        try:
            current_creator = self._registry[backend]
        except KeyError:
            current_creator = None

        if current_creator is not None and current_creator != creator:
            raise ValueError(
                f"The rendezvous backend '{backend}' cannot be registered with '{creator}' as it "
                f"is already registered with '{current_creator}'."
            )

        self._registry[backend] = creator

    def create_handler(self, params: RendezvousParameters) -> RendezvousHandler:
        """Creates a new :py:class:`RendezvousHandler`."""
        try:
            creator = self._registry[params.backend]
        except KeyError as e:
            raise ValueError(
                f"The rendezvous backend '{params.backend}' is not registered. Did you forget "
                f"to call `{self.register.__name__}`?"
            ) from e

        handler = creator(params)

        # Do some sanity check.
        if handler.get_backend() != params.backend:
            raise RuntimeError(
                f"The rendezvous backend '{handler.get_backend()}' does not match the requested "
                f"backend '{params.backend}'."
            )

        return handler


# The default global registry instance used by launcher scripts to instantiate
# rendezvous handlers.
rendezvous_handler_registry = RendezvousHandlerRegistry()
