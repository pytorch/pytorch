import warnings
from typing import NamedTuple, Any, Callable, List

import torch
import torch.distributed as dist

class _JoinHook(NamedTuple):
    r"""
    This encapsulates the data defining a join hook, which provides three entry
    points for hooks in the join context manager:
        ``object`` (Any): This is the object to be passed into the hooks.
        ``pre_hook`` (object -> None): This hook is called upon entering the
            join context.
        ``main_hook`` (object -> None): This hook is called repeatedly while
            there exists a non-joined process to shadow collective
            communications in the forward pass, backward pass, and optimizer
            step.
        ``post_hook`` (object * bool -> None): This hook is called after all
            processes have joined. It is passed an additional ``bool`` argument
            ``is_last_joiner``, which indicates if the rank is one of the last
            to join.
    """
    object: Any
    pre_hook: Callable
    main_hook: Callable
    post_hook: Callable

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
        For the ``join_hook.object`` s, the context manager requires that at
        least one has its ``process_group`` attribute set, that at least one
        has its ``device`` attribute set, and that all specified process groups
        are the same. This process group and device information is used for
        checking for non-joined processes and for notifying processes to
        terminate if ``throw_on_early_termination`` is enabled, both of which
        using an all-reduce.

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
                If none of the ``join_hook.object`` s have ``process_group``
                set; if none of the ``join_hook.object`` s have ``device``
                set; or if there are multiple conflicting ``process_group`` s.

        NOTE: If there are multiple devices specified, then the context manager
        uses the first.
        """
        process_group = None
        device = None
        for join_hook in self._join_hooks:
            if hasattr(join_hook.object, "process_group"):
                if process_group is None:
                    process_group = join_hook.object.process_group
                elif process_group != join_hook.object.process_group:
                    raise ValueError("Using join context manager with multiple process groups")
            if hasattr(join_hook.object, "device"):
                if device is None:
                    device = join_hook.object.device
        if process_group is None:
            raise ValueError(
                "Using the join context manager without specifying a process "
                "group; make sure that at least one ``join_hook.object`` has "
                "its ``process_group`` attribute set"
            )
        if device is None:
            raise ValueError(
                "Using the join context manager without specifying a device; "
                "make sure that at least one ``join_hook.object`` has its "
                "``device`` attribute set"
            )
        self._process_group = process_group
        self._rank = dist.get_rank(self._process_group)
        self._device = device

    def __enter__(self):
        """Runs the pre-hooks."""
        for join_hook in self._join_hooks:
            if join_hook.pre_hook is not None:
                join_hook.pre_hook(join_hook.object)

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
                    if join_hook.main_hook is not None:
                        join_hook.main_hook(join_hook.object)

                is_last_joiner = False
                i += 1

        # Run post-hooks
        for join_hook in self._join_hooks:
            if join_hook.post_hook is not None:
                join_hook.post_hook(join_hook.object, is_last_joiner)

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
