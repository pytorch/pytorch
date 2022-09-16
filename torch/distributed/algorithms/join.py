import warnings
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, List, NamedTuple, Optional, Type

import torch
import torch.distributed as dist

__all__ = ['JoinHook', 'Joinable', 'Join']

class JoinHook():
    r"""
    This defines a join hook, which provides two entry points in the join
    context manager: a main hook, which is called repeatedly while there exists
    a non-joined process, and a post-hook, which is called once all processes
    have joined.

    To implement a join hook for the generic join context manager, define a
    class that inherits from :class:`JoinHook` and override ``main_hook()`` and
    ``post_hook()`` as appropriate.
    """
    def main_hook(self) -> None:
        r"""
        This hook is called repeatedly while there exists a non-joined process
        to shadow collective communications in one training iteration (i.e. in
        one forward pass, backward pass, and optimizer step).
        """
        ...

    def post_hook(self, is_last_joiner: bool) -> None:
        r"""
        This hook is called after all processes have joined. It is passed an
        additional ``bool`` argument ``is_last_joiner``, which indicates if the
        rank is one of the last to join.

        Arguments:
            is_last_joiner (bool): ``True`` if the rank is one of the last to
                join; ``False`` otherwise.
        """
        ...


class Joinable(ABC):
    r"""
    This defines an abstract base class for joinable classes. A joinable class
    (inheriting from :class:`Joinable`) should implement :meth:`join_hook`,
    which returns a :class:`JoinHook` instance, in addition to
    :meth:`join_device` and :meth:`join_process_group` that return device and
    process group information, respectively.
    """
    @abstractmethod
    def __init__(self):
        super(Joinable, self).__init__()
        self._join_config = _JoinConfig.construct_disabled_join_config()

    @abstractmethod
    def join_hook(self, **kwargs) -> JoinHook:
        r"""
        Returns a :class:`JoinHook` instance for the given :class:`Joinable`.

        Arguments:
            kwargs (dict): a :class:`dict` containing any keyword arguments
                to modify the behavior of the join hook at run time; all
                :class:`Joinable` instances sharing the same join context
                manager are forwarded the same value for ``kwargs``.
        """
        ...

    @property
    @abstractmethod
    def join_device(self) -> torch.device:
        r"""
        Returns the device from which to perform collective communications
        needed by the join context manager implementation itself.
        """
        ...

    @property
    @abstractmethod
    def join_process_group(self) -> Any:
        r"""
        Returns the process group for the collective communications needed by
        the join context manager itself.
        """
        ...


class _JoinConfig(NamedTuple):
    r"""
    This includes all fields needed from a :class:`Joinable` instance for the
    join context manager side.
    """
    enable: bool
    throw_on_early_termination: bool
    is_first_joinable: bool

    @staticmethod
    def construct_disabled_join_config():
        r"""
        Returns a :class:`_JoinConfig` instance indicating that join-related
        logic should be disabled, e.g. if the caller is not in a join context
        manager.
        """
        return _JoinConfig(
            enable=False,
            throw_on_early_termination=False,
            is_first_joinable=False
        )



class Join():
    r"""
    This class defines the generic join context manager, which allows custom
    hooks to be called after a process joins. These hooks should shadow the
    collective communications of non-joined processes to prevent hanging and
    erroring and to ensure algorithmic correctness. Refer to :class:`JoinHook`
    for details about the hook definition.

    .. warning::
        The context manager requires each participating :class:`Joinable` to
        call the method :meth:`notify_join_context()` before its own per-
        iteration collective communications to ensure correctness.

    .. warning::
        The context manager requires that all ``process_group`` attributes in
        the :class:`JoinHook` objects are the same. If there are multiple
        :class:`JoinHook` objects, then the ``device`` of the first is used.
        The process group and device information is used for checking for non-
        joined processes and for notifying processes to throw an exception if
        ``throw_on_early_termination`` is enabled, both of which using an all-
        reduce.

    Arguments:
        joinables (List[Joinable]): a list of the participating
            :class:`Joinable` s; their hooks are iterated over in the given
            order.

        enable (bool): a flag enabling uneven input detection; setting to
            ``False`` disables the context manager's functionality and should
            only be set when the user knows the inputs will not be uneven
            (default: ``True``).

        throw_on_early_termination (bool): a flag controlling whether to throw an
            exception upon detecting uneven inputs (default: ``False``).

    Example::

        >>> import os
        >>> import torch
        >>> import torch.distributed as dist
        >>> import torch.multiprocessing as mp
        >>> # xdoctest: +SKIP
        >>> import torch.nn.parallel.DistributedDataParallel as DDP
        >>> import torch.distributed.optim.ZeroRedundancyOptimizer as ZeRO
        >>> from torch.distributed.algorithms.join import Join
        >>>
        >>> # On each spawned worker
        >>> def worker(rank):
        >>>     dist.init_process_group("nccl", rank=rank, world_size=2)
        >>>     model = DDP(torch.nn.Linear(1, 1).to(rank), device_ids=[rank])
        >>>     optim = ZeRO(model.parameters(), torch.optim.Adam, lr=0.01)
        >>>     # Rank 1 gets one more input than rank 0
        >>>     inputs = [torch.tensor([1.]).to(rank) for _ in range(10 + rank)]
        >>>     with Join([model, optim]):
        >>>         for input in inputs:
        >>>             loss = model(input).sum()
        >>>             loss.backward()
        >>>             optim.step()
        >>>     # All ranks reach here without hanging/erroring
    """
    def __init__(
        self,
        joinables: List[Joinable],
        enable: bool = True,
        throw_on_early_termination: bool = False,
        **kwargs,
    ):
        if len(joinables) == 0:
            raise ValueError("The join context manager requires at least one joinable")
        self._joinables = joinables
        self._join_hooks = [joinable.join_hook(**kwargs) for joinable in self._joinables]
        self._enable = enable
        self._throw_on_early_termination = throw_on_early_termination
        self._set_joinable_configs()
        self._extract_dist_info()

    def _set_joinable_configs(self) -> None:
        r"""
        Sets the :class:`_JoinConfig` of each participating :class:`Joinable`.
        """
        assert len(self._joinables) > 0
        is_first_joinable = True
        for joinable in self._joinables:
            joinable._join_config = _JoinConfig(
                enable=self._enable,
                throw_on_early_termination=self._throw_on_early_termination,
                is_first_joinable=is_first_joinable
            )
            is_first_joinable = False

    def _extract_dist_info(self) -> None:
        r"""
        Extracts the process group and device information from the joinables.
        If there are multiple joinables, then the context manager uses the
        first specified device.

        Preconditions:
            ``self._joinables`` is not ``None`` and is non-empty.

        Raises:
            ValueError
                If there are multiple conflicting ``process_group`` attributes
                among the ``Joinable`` objects.
        """
        process_group = None
        device = None
        for joinable in self._joinables:
            if process_group is None:
                process_group = joinable.join_process_group
            elif process_group != joinable.join_process_group:
                raise ValueError("Using join context manager with multiple process groups")
            if device is None:
                device = joinable.join_device
        self._process_group = process_group
        self._rank = dist.get_rank(self._process_group)
        self._device = device

    def __enter__(self):
        ...

    def __exit__(
        self,
        type: Optional[Type[BaseException]],
        value: Optional[BaseException],
        traceback: Optional[TracebackType]
    ):
        r"""
        Repeatedly runs the main hooks until all processes join; then, runs
        the post-hooks.

        Raises:
            RuntimeError
                If ``throw_on_early_termination=True``.
        """
        if not self._enable or type:
            return  # propagate the exception directly if one was raised

        all_procs_joined = False
        is_last_joiner = True

        i = 0
        WARN_THRESHOLD = 1000
        warnings.simplefilter("once")

        while not all_procs_joined:
            if i > WARN_THRESHOLD:
                warnings.warn(
                    "Detected uneven input skew of greater than "
                    f"{WARN_THRESHOLD}. This means that rank "
                    f"{self._rank} has at least {WARN_THRESHOLD} "
                    f"fewer inputs than other currently-active ranks. "
                    "This level of skew could lead to performance "
                    "degradation during training."
                )
            # Shadow the all-reduce in non-joined processes
            num_nonjoined_procs = self._get_num_nonjoined_procs()
            if num_nonjoined_procs == 0:
                all_procs_joined = True
            else:
                if self._throw_on_early_termination:
                    self._notify_procs_to_terminate()

                # Run main hooks
                for join_hook in self._join_hooks:
                    join_hook.main_hook()

                is_last_joiner = False
                i += 1

        # Run post-hooks
        for join_hook in self._join_hooks:
            join_hook.post_hook(is_last_joiner)

    def _get_num_nonjoined_procs(self):
        r"""
        Returns the number of non-joined processes by shadowing an all-reduce
        in the non-joined processes.
        """
        num_nonjoined_procs = torch.zeros(1, device=self._device)
        dist.all_reduce(num_nonjoined_procs, group=self._process_group)
        return num_nonjoined_procs.item()

    def _notify_procs_to_terminate(self):
        r"""
        Schedules an all-reduce to notify non-joined processes to terminate
        and raises a ``RuntimeError`` indicating that the current process has
        exhausted its inputs.
        """
        ones = torch.ones(1, device=self._device)
        dist.all_reduce(ones, group=self._process_group)
        raise RuntimeError(f"Rank {self._rank} exhausted all inputs.")

    @staticmethod
    def notify_join_context(joinable: Joinable):
        r"""
        Notifies the join context manager that the calling process has not yet
        joined; then, if ``throw_on_early_termination=True``, checks if uneven
        inputs have been detected (i.e. if one process has already joined) and
        throws an exception if so.

        This method should be called from a :class:`Joinable` object before
        its per-iteration collective communications. For example, this should
        be called at the beginning of the forward pass in
        :class:`DistributedDataParallel`.

        Only the first :class:`Joinable` object passed into the context
        manager performs the collective communications in this method, and
        for the others, this method is vacuous.

        Arguments:
            joinable (Joinable): the :class:`Joinable` object calling this
                method.

        Returns:
            An async work handle for the all-reduce meant to notify the context
            manager that the process has not yet joined if ``joinable`` is the
            first one passed into the context manager; ``None`` otherwise.
        """
        assert hasattr(joinable, "_join_config"), \
            f"Check that the {type(joinable)} constructor calls the " \
            "``Joinable`` constructor"

        join_config = joinable._join_config
        # First joinable is responsible for the collective communications
        if not join_config.is_first_joinable or not join_config.enable:
            return None

        device = joinable.join_device
        process_group = joinable.join_process_group

        # Schedule an all-reduce to indicate that the caller has not yet joined
        ones = torch.ones(1, device=device)
        work = dist.all_reduce(ones, group=process_group, async_op=True)

        if join_config.throw_on_early_termination:
            # Check if uneven inputs have been detected
            zeros = torch.zeros(1, device=device)
            dist.all_reduce(zeros, group=process_group)
            should_throw = zeros.item()
            if should_throw:
                raise RuntimeError(
                    "Detected at least one rank that exhausted inputs. "
                    "Throwing across all ranks."
                )
        return work
