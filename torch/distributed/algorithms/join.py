import warnings
from abc import ABC, abstractmethod
from typing import Any, List, NamedTuple

import torch
import torch.distributed as dist


class _JoinHook(ABC):
    r"""
    This defines a join hook, which provides two entry points in the join
    context manager: a main hook, which is called repeatedly while there exists
    a non-joined process, and a post-hook, which is called once all processes
    have joined.

    To implement a join hook for the generic join context manager, define a
    class that inherits from :class:`_JoinHook`, override ``main_hook()`` and
    ``post_hook()`` as appropriate, and override ``device()`` and
    ``process_group()`` to provide the device and process group information,
    respectively, which are needed for the join context manager implementation.
    """
    def main_hook(self):
        r"""
        This hook is called repeatedly while there exists a non-joined process
        to shadow collective communications in the forward pass, backward pass,
        and optimizer.
        """
        ...

    def post_hook(self, is_last_joiner: bool):
        r"""
        This hook is called after all processes have joined. It is passed an
        additional ``bool`` argument ``is_last_joiner``, which indicates if the
        rank is one of the last to join.

        Arguments:
            is_last_joiner (bool): ``True`` if the rank is one of the last to
                join; ``False`` otherwise.
        """
        ...


class _Joinable(ABC):
    r"""
    This defines an abstract base class for joinable classes. A joinable class
    (inheriting from :class:`_Joinable`) should implement a private
    ``_join_hook()`` method that returns a :class:`_JoinHook` instance.
    """
    @abstractmethod
    def __init__(self):
        super(_Joinable, self).__init__()
        self._join_config = _JoinConfig.construct_disabled_join_config()

    @abstractmethod
    def _join_hook(self, **kwargs) -> _JoinHook:
        r"""
        Returns a :class:`_JoinHook` instance for the given :class:`_Joinable`.

        Arguments:
            kwargs (dict): a :class:`dict` containing any keyword arguments
                to modify the behavior of the join hook at run time; all
                :class:`_Joinable` instances sharing the same join context
                manager are forwarded the same value for ``kwargs``.
        """
        ...

    @property
    @abstractmethod
    def _join_device(self) -> torch.device:
        r"""
        Returns the device from which to perform collective communications
        needed for the join context manager implementation itself.
        """
        ...

    @property
    @abstractmethod
    def _join_process_group(self) -> Any:
        r"""
        Returns the process group for join-related collective communications.
        """
        ...


class _JoinConfig(NamedTuple):
    r"""
    This includes all fields needed from a :class:`_Joinable` instance for the
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



class _Join():
    r"""
    This class defines the generic join context manager, which allows custom
    hooks to be called after a process joins. These hooks should shadow the
    collective communications of non-joined processes to prevent hanging and
    erroring and to ensure algorithmic correctness. Refer to :class:`_JoinHook`
    for details about the hook definition.

    .. warning::
        The context manager requires each participating :class:`_Joinable` to
        call the method `notify_join_context()` before its own per-iteration
        collective communications to ensure correctness.

    .. warning::
        The context manager requires that all ``process_group`` attributes in
        the ``_JoinHook`` objects are the same. If there are multiple
        ``_JoinHook`` objects, then the ``device`` of the first is used. The
        process group and device information is used for checking for non-
        joined processes and for notifying processes to terminate if
        ``throw_on_early_termination`` is enabled, both of which using an all-
        reduce.

    Arguments:
        joinables (List[_Joinable]): a list of the participating
            :class:`_Joinable` s; their hooks are iterated over in the given
            order.

        enable (bool): a flag enabling uneven input detection; setting to
            ``False`` disables the context manager's functionality and should
            only be set when the user knows the inputs will not be uneven
            (default: ``True``).

        throw_on_early_termination (bool): a flag controlling whether to throw an
            exception upon detecting uneven inputs (default: ``False``).

    """
    def __init__(
        self,
        joinables: List[_Joinable],
        enable: bool = True,
        throw_on_early_termination: bool = False,
        **kwargs,
    ):
        if len(joinables) == 0:
            raise ValueError("The join context manager requires at least one joinable")
        self._joinables = joinables
        self._join_hooks = [joinable._join_hook(**kwargs) for joinable in self._joinables]
        self._enable = enable
        self._throw_on_early_termination = throw_on_early_termination
        self._set_joinable_configs()
        self._extract_dist_info()

    def _set_joinable_configs(self):
        r"""
        Sets the :class:`_JoinConfig` of each participating :class:`_Joinable`.
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

    def _extract_dist_info(self):
        r"""
        Extracts the process group and device information from the join hooks.

        Preconditions:
            ``self._join_hooks`` is not ``None`` and is non-empty.

        Raises:
            ValueError
                If there are multiple conflicting ``process_group`` attributes
                among the ``_JoinHook`` objects.

        NOTE: The context manager uses the first specified device.
        """
        process_group = None
        device = None
        for joinable in self._joinables:
            if process_group is None:
                process_group = joinable._join_process_group
            elif process_group != joinable._join_process_group:
                raise ValueError("Using join context manager with multiple process groups")
            if device is None:
                device = joinable._join_device
        self._process_group = process_group
        self._rank = dist.get_rank(self._process_group)
        self._device = device

    def __enter__(self):
        ...

    def __exit__(self, type, value, traceback):
        r"""
        Repeatedly runs the main hooks until all processes join; then, runs
        the post-hooks.

        Raises:
            RuntimeError
                If ``throw_on_early_termination`` is enabled.
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
                    "degradataion during training."
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
        # NOTE: Raising `StopIteration` does not throw an error in Python 3.6
        # and throws a `RuntimeError` in Python 3.7+ (PEP 479), so we just
        # raise a `RuntimeError` here
        raise RuntimeError(f"Rank {self._rank} exhausted all inputs.")

    @staticmethod
    def notify_join_context(joinable: _Joinable):
        r"""
        Notifies the join context manager that the calling process has not yet
        joined; then, if ``throw_on_early_termination=True``, checks if uneven
        inputs have been detected (i.e. if one process has already joined) and
        throws an exception if so.

        This method should be called from a :class:`_Joinable` object before
        its per-iteration collective communications. For example, this should
        be called at the beginning of the forward pass in
        :class:`DistributedDataParallel`.

        Only the first :class:`_Joinable` object passed into the context
        manager performs the collective communications in this method, and
        for the others, this method is vacuous.

        Arguments:
            joinable (_Joinable): the :class:`_Joinable` object calling this
                method.

        Returns:
            An async work handle for the all-reduce meant to notify the context
            manager that the process has not yet joined if ``joinable`` is the
            first one passed into the context manager; ``None`` otherwise.
        """
        assert hasattr(joinable, "_join_config"), \
            f"Check that the {type(joinable)} constructor calls the " \
            "``_Joinable`` constructor"

        join_config = joinable._join_config
        # First joinable is responsible for the collective communications
        if not join_config.is_first_joinable or not join_config.enable:
            return None

        device = joinable._join_device
        process_group = joinable._join_process_group

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
