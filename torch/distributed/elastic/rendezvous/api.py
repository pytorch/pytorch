# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Any, Callable, Dict, Optional, Tuple

from torch.distributed import Store


class RendezvousException(Exception):
    """
    Represents the base type for rendezvous exceptions.
    """

    pass


class RendezvousClosedException(RendezvousException):
    """
    Raised when a rendezvous is closed.

    This is used to signal completion to nodes that arrive late.
    """

    pass


class RendezvousTimeoutException(RendezvousException):
    """
    Raised to signal that a rendezvous did not succeed within the allocated
    time.

    This is a non-retryable type of failure.
    """

    pass


class RendezvousNonRetryableError(RendezvousException):
    """
    Raised when a failure occured that should not be retried within the same
    worker process.
    """

    pass


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
            RendezvousClosedException - if rendezvous for the current
               job is closed.
            RendezvousTimeoutException - on timeout
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
        the rendezvous barrier, hence weren’t included in the current worker
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
    """
    The data object holding parameters to construct a ``RendezvousHandler``.
    """

    # Default timeout for the rendezvous.
    _DEFAULT_TIMEOUT: int = 600  # 10 minutes

    # Additional waiting time after reaching the minimum number of nodes
    # in case the rendezvous is elastic (min != max).
    _DEFAULT_LAST_CALL_TIMEOUT: int = 30  # 30 seconds

    def __init__(
        self,
        backend: str,
        endpoint: str,
        run_id: str,
        min_nodes: int,
        max_nodes: int,
        **kwargs,
    ):
        """
        Args:
            backend: The backend that is used to register the rendezvous.
            endpoint: The endpoint of the rendezvous. Usually it is a string in the format
                <hostname>:<port>.
            run_id: The id of the rendezvous.
            min_nodes: The minimum number of nodes required to complete the rendezvous.
            max_nodes: The maximum number of nodes that are allowed to join the rendezvous.
            **kwargs: Additional parameters for the specified backend.
        """
        if backend is None:
            raise ValueError("The backend cannot be None.")

        if min_nodes < 1:
            raise ValueError("The minimum number of nodes must be greater than zero.")
        if max_nodes < min_nodes:
            raise ValueError(
                "The maximum number of nodes must be greater than"
                " or equal to the minimum number of nodes."
            )

        self.backend = backend
        self.endpoint = endpoint
        self.run_id = run_id
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.config = kwargs

    @property
    def timeout(self):
        """
        Gets the timeout for the rendezvous.
        """
        return self.get_as_int("timeout", self._DEFAULT_TIMEOUT)

    @property
    def last_call_timeout(self):
        """
        Gets additional waiting time after reaching the minimum number of nodes
        in case the rendezvous is elastic (min != max).
        """
        return self.get_as_int("last_call_timeout", self._DEFAULT_LAST_CALL_TIMEOUT)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Returns the value for ``key`` if ``key`` exists, else ``default``.
        """
        return self.config.get(key, default)

    def get_as_bool(self, key: str, default: Optional[bool] = None) -> Optional[bool]:
        """
        Returns the value for ``key`` as a ``bool`` if ``key`` exists.
        """
        val = self.get(key, default)
        if val is None:
            return val
        if isinstance(val, int) or isinstance(val, bool):
            return True if val else False
        if isinstance(val, str):
            return val.lower() in ["1", "true", "t", "yes", "y"]
        raise ValueError(
            f"The '{key}' rendezvous config does not represent a valid boolean value."
        )

    def get_as_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """
        Returns the value for ``key`` as an ``int`` if ``key`` exists.
        """
        val = self.get(key, default)
        if val is None:
            return val
        try:
            return int(val)
        except ValueError:
            raise ValueError(
                f"The '{key}' rendezvous config does not represent a valid integer value."
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
                f"The rendezvous backend '{params.backend}' is not registered. Did you forget to call {self.register.__name__}?"
            )

        handler = creator(params)

        # Do some sanity check.
        if handler.get_backend() != params.backend:
            raise RuntimeError(
                f"The rendezvous handler backend '{handler.get_backend()}' does not match the requested backend '{params.backend}'."
            )

        return handler
