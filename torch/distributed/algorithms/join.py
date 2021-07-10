import warnings
from abc import ABC, abstractmethod
from typing import List

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

    @property
    @abstractmethod
    def device(self):
        r"""
        Returns the device from which to perform collective communications
        needed for the join context manager implementation itself.
        """
        ...

    @property
    @abstractmethod
    def process_group(self):
        r"""
        Returns the process group for join-related collective communications.
        """
        ...

class _Join():
    r"""
    This class defines the generic join context manager, which allows custom
    hooks to be called after a process joins. These hooks should shadow the
    collective communications of non-joined processes to prevent hanging and
    erroring and to ensure algorithmic correctness. Refer to :class:`_JoinHook`
    for details about the hook definition.

    .. warning::
        The context manager requires a ``dist.all_reduce(torch.ones(1))`` to be
        called on every non-joined process each time before it performs its
        collective communications in order to indicate that the process has not
        yet joined. For example, this can occur at the beginning of the forward
        pass.

    .. warning::
        If ``throw_on_early_termination`` is enabled, then the context manager
        additionally requires every non-joined process to participate in an
        all-reduce before it performs its collective communications in order to
        check if it should terminate due to detecting uneven inputs. This all-
        reduce should be of the form ``dist.all_reduce(torch.zeros(1))``; if
        the result is positive, then the process should terminate.

    .. warning::
        The context manager requires that all ``process_group`` attributes in
        the ``_JoinHook`` objects are the same. If there are multiple
        ``_JoinHook`` objects, then the ``device`` of the first is used. The
        process group and device information is used for checking for non-
        joined processes and for notifying processes to terminate if
        ``throw_on_early_termination`` is eanbled, both of which using an all-
        reduce.

    Arguments:
        join_hooks (List[_JoinHook]): a list of the :class:`_JoinHook` s to
            use; the hooks are iterated over in the given order.

        enable (bool): a flag enabling uneven input detection; setting to
            ``False`` disables the context manager's functionality and should
            only be set when the user knows the inputs will not be uneven
            (default: ``True``).

        throw_on_early_termination (bool): a flag controlling whether to raise
            an exception upon detecting uneven inputs (default: ``False``).

    """
    def __init__(
        self,
        join_hooks: List[_JoinHook],
        enable: bool = True,
        throw_on_early_termination: bool = False,
    ):
        if len(join_hooks) == 0:
            raise ValueError("The join context manager requires at least one join hook")
        self._join_hooks = join_hooks
        self._enable = enable
        self._throw_on_early_termination = throw_on_early_termination
        self._extract_dist_info()

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
        for join_hook in self._join_hooks:
            if process_group is None:
                process_group = join_hook.process_group
            elif process_group != join_hook.process_group:
                raise ValueError("Using join context manager with multiple process groups")
            if device is None:
                device = join_hook.device
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
