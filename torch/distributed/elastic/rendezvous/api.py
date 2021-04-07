# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
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


class RendezvousHandler(abc.ABC):
    """
    Main rendezvous interface.

    .. note:: torchelastic users normally **do not** need to implement their
              own ``RendezvousHandler``. An implementation based on
              `etcd <https://etcd.io/>`__ is already provided, and is recommended
              for most users, provided they can deploy it in their environment.

    .. warning:: torchelastic is currently considered experimental,
                 so the APIs may change!
    """

    @abc.abstractmethod
    def get_backend(self) -> str:
        """
        Return the string representation of the rendezvous handler.
        """
        pass

    @abc.abstractmethod
    def next_rendezvous(
        self,
    ) -> Tuple[Store, int, int]:
        """
        Main entry-point into the rendezvous barrier.
        Blocks until the rendezvous is complete (and the current
        process is included in the formed worker group), or a timeout occurs, or
        rendezvous was marked closed.

        Returns: a tuple of (``c10d Store``, ``rank``, ``world size``)

        Raises:
            RendezvousClosedError - if rendezvous for the current job is closed.
            RendezvousTimeoutError - on timeout
        """
        pass

    @abc.abstractmethod
    def is_closed(self) -> bool:
        """
        Checks whether rendezvous for current job has been closed,
        which means all future attempts to re-rendezvous (within same job) will
        fail.

        .. note:: ``is_closed`` and ``set_closed`` have semantics of eventual
                  propagation, and should not be used for synchronization.
                  The intention here is that if at least one worker decides
                  the job is finished, it will close the rendezvous, and
                  other workers will soon observe this and stop
                  training/rendezvous-ing as well.
        """
        pass

    @abc.abstractmethod
    def set_closed(self):
        """
        Used to mark the rendezvous (for current job) as closed.
        """
        pass

    @abc.abstractmethod
    def num_nodes_waiting(self) -> int:
        """
        Returns number of workers who *arrived late* at
        the rendezvous barrier, hence weren't included in the current worker
        group.

        Callers should periodically call this method to check whether
        new members are waiting to join the job and if so admit them by
        calling ``next_rendezvous()`` (re-rendezvous).
        """
        pass

    @abc.abstractmethod
    def get_run_id(self) -> str:
        """
        Returns the run_id of this rendezvous handler. The run_id is a user-defined
        id that uniquely identifies an instance of a distributed application.
        It typically maps to a job id and is used to allow workers to join the
        correct distributed application.
        """
        pass

    def shutdown(self) -> bool:
        """
        Closes all resources that were open for rendezvous run.

        Usage:

        ::

         def main():
             rdzv_handler = ...
             try:
               rank, world_size, store = rdzv_handler.next_rendezvous()
             finally:
               rdzv_handler.shutdown()
        """
        pass


class RendezvousParameters:
    """Holds the parameters to construct a `RendezvousHandler`.

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

    def get(self, key: str, default: Any = None) -> Any:
        """Returns the value for `key` if `key` exists, else `default`."""
        return self.config.get(key, default)

    def get_as_bool(self, key: str, default: Optional[bool] = None) -> Optional[bool]:
        """Returns the value for `key` as a `bool`."""
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
        """Returns the value for `key` as an `int`."""
        value = self.get(key, default)
        if value is None:
            return value
        try:
            return int(value)
        except ValueError:
            raise ValueError(
                f"The rendezvous configuration option '{key}' does not represent a valid integer "
                "value."
            )


RendezvousHandlerCreator = Callable[[RendezvousParameters], RendezvousHandler]


class RendezvousHandlerFactory:
    """
    Creates ``RendezvousHandler`` instances for supported rendezvous backends.
    """

    def __init__(self):
        self._registry: Dict[str, RendezvousHandlerCreator] = {}

    def register(self, backend: str, creator: RendezvousHandlerCreator):
        """
        Registers a new rendezvous backend.
        """
        try:
            current_creator = self._registry[backend]
        except KeyError:
            current_creator = None  # type: ignore[assignment]

        if current_creator is not None:
            raise ValueError(
                f"The rendezvous backend '{backend}' cannot be registered with"
                f" '{creator.__module__}.{creator.__name__}' as it is already"
                f" registered with '{current_creator.__module__}.{current_creator.__name__}'."
            )

        self._registry[backend] = creator

    def create_handler(self, params: RendezvousParameters) -> RendezvousHandler:
        """
        Creates a new ``RendezvousHandler`` instance for the specified backend.
        """
        try:
            creator = self._registry[params.backend]
        except KeyError:
            raise ValueError(
                f"The rendezvous backend '{params.backend}' is not registered. Did you forget "
                f"to call {self.register.__name__}?"
            )

        handler = creator(params)

        # Do some sanity check.
        if handler.get_backend() != params.backend:
            raise RuntimeError(
                f"The rendezvous handler backend '{handler.get_backend()}' does not match the "
                f"requested backend '{params.backend}'."
            )

        return handler
