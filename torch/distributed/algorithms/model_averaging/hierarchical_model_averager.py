# Copyright 2022 Cruise LLC
import logging
import warnings
from collections import OrderedDict
from typing import Union, Iterable, Dict

import torch
import torch.distributed as dist
import torch.distributed.algorithms.model_averaging.averagers as averagers
import torch.distributed.algorithms.model_averaging.utils as utils

logger = logging.getLogger(__name__)


class HierarchicalModelAverager(averagers.ModelAverager):
    r"""
    Runs hierarchical model averaging (`hierarchical SGD <https://arxiv.org/pdf/2010.12998.pdf>`_).
    Process groups of different sizes are organized in a hierarhicy, and they average parameters
    by using different periods concurrently after the warm-up stage.
    This is an extension of :class:`~torch.distributed.algorithms.model_averaging.averagers.PeriodicModelAverager`
    that supports `post-local SGD <https://arxiv.org/abs/1808.07217>`_, which essentially only supports
    a two-level hierarchy: the intra-machine level and the global level, where the intra-machine
    level is usually embedded in :meth:`~torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook`.
    Similarly, the process groups within this class do not have such an intra-machine process
    subgroup, which should be embedded by the post-local SGD communication hook instead.

    Args:
        period_group_size_dict: An ordered dict mapping keys of model averaging period to
                                process group size, used for initializing process groups of
                                different sizes in a hierarchy to average parameters concurrently.
                                Particularly, at each iteration, there will be at most a single
                                process group that runs averaging -- the period of such group should
                                have the largest period which the current step can be divided by.
                                For example, if the dict has three keys: 2, 4, and 8,
                                then this means totally three process groups will be created to
                                average parameters every 2, 4, and 8 iterations, respectively.
                                At the 4th iteration, only the second process group will run
                                averaging, because the first process group should be a
                                subset of the second process group, and no need to execute the first
                                process group redundantly.
                                On the other hand, the third process group can only be triggered
                                every 8 iterations, so it will not be triggered at the 4th iteration.
        warmup_steps (int): The number of warm-up steps. During this stage, model averaging is skipped.
        process_group (ProcessGroup, optional): The overall process group containing all the processes that runs model averaging.
                                                If ``None``, the default process group, which is created
                                                by :func:`torch.distributed.init_process_group`, will be used.
                                                (default: ``None``)

    Example::
        >>> # xdoctest: +SKIP('undefined rank')
        >>> from collections import OrderedDict
        >>> import torch
        >>> import torch.distributed as dist
        >>> from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import (
        >>>     PostLocalSGDState,
        >>>     post_localSGD_hook,
        >>> )
        >>> import torch.distributed.algorithms.model_averaging.hierarchical_model_averager as hierarchicalSGD
        >>> import torch.nn as nn
        >>>
        >>> dist.init_process_group("nccl", rank=rank, world_size=16)
        >>> torch.cuda.set_device(rank)
        >>> module = nn.Linear(1, 1, bias=False).to(rank)
        >>> model = nn.parallel.DistributedDataParallel(
        >>>    module, device_ids=[rank], output_device=rank
        >>> )
        >>> # Register a post-localSGD communication hook.
        >>> # Assume that each machine has 4 GPUs, then each intra-machine subgroup has a size of 4.
        >>> subgroup, _ = dist.new_subgroups()
        >>> state = PostLocalSGDState(process_group=None, subgroup=subgroup, start_localSGD_iter=100)
        >>> model.register_comm_hook(state, post_localSGD_hook)
        >>>
        >>> # Average parameters among each group of 8 processes every 4 iterations, and among all
        >>> # the 16 processes every 16 iterations.
        >>> averager = hierarchicalSGD.HierarchicalModelAverager(
        >>>     period_group_size_dict=OrderedDict([(4, 8), (16, 16)]), warmup_steps=100)
        >>> # Note that ``warmup_steps`` must be the same as ``start_localSGD_iter`` used in ``PostLocalSGDState``.
        >>> # In the first 100 steps, run global gradient averaging like normal DDP at every step.
        >>> # After 100 steps, run model averaging at two levels.
        >>> for step in range(0, 200):
        >>>    optimizer.zero_grad()
        >>>    loss = loss_fn(output, labels)
        >>>    loss.backward()
        >>>    optimizer.step()
        >>>    # Average parameters after ``optimizer.step()``.
        >>>    # Thus, the inter-node communication only occurs periodically after ``warmup_steps``.
        >>>    averager.average_parameters(model.parameters())

    .. warning ::
        The last group size in the dict must be the size of the provided ``process_group``,
        which indicates model averaging at the highest level of the hierarchy.
        If ``process_group`` is not provided, then the last group size should be equal to the world size.

    .. warning ::
        `HierarchicalModelAverager` is experimental and subject to change.
    """

    def __init__(self, period_group_size_dict=None, warmup_steps=0, process_group=None):
        super().__init__(process_group)
        if not period_group_size_dict:
            raise ValueError("Arg ``period_group_size_dict`` must not be empty.")
        self._periods = list(period_group_size_dict.keys())
        if self._periods[0] <= 0:
            raise ValueError("The minimum period in arg ``period_group_size_dict`` must be a positive value.")
        elif self._periods[-1] == 1:
            warnings.warn(
                "When the maximum period in arg ``period_group_size_dict`` is 1, "
                "no need to use model averaging because the communication cost "
                "of all-reducing parameters will be no less than the cost of all-reducing gradients "
                "by DistributedDataParallel in the backward pass. Therefore, only "
                "DistributedDataParallel should be used for this case."
            )
        overall_group_size = dist.get_world_size(group=self.process_group)
        if list(period_group_size_dict.values())[-1] != overall_group_size:
            raise ValueError(
                f"The last value in arg ``period_process_group_dict`` {list(period_group_size_dict.values())[-1]} "
                f"must be equal to the size of arg ``process_group`` {overall_group_size}."
            )

        self.period_process_group_dict = OrderedDict()
        logger.info("Model averaging hierarchy:")
        for period, group_size in period_group_size_dict.items():
            logger.info(
                f"\tEach group that has {group_size} processes average parameters every {period} iterations, "
                "if no higher-level averaging.")
            if group_size != overall_group_size:
                self.period_process_group_dict[period], _ = dist.new_subgroups(
                    group_size=group_size, group=self.process_group)
            else:
                self.period_process_group_dict[period] = self.process_group

        if warmup_steps < 0:
            raise ValueError("Arg ``warmup_steps`` must be a non-negative number.")
        self.warmup_steps = warmup_steps

    def _find_process_group(self):
        """
        Returns a process group as the value of an ``period_process_group_dict`` entry,
        if ``step`` can be divided by a period in the keys of ``period_process_group_dict``.
        If ``step`` can be divided by multiple periods in the keys of ``period_process_group_dict``,
        then the returned process group is the one corresponding to the largest period,
        since this process group will be used for averaging parameters at this ``step``.
        Returns ``None`` if not found.
        """
        for period in reversed(self._periods):
            if self.step % period == 0:
                return self.period_process_group_dict[period]
        return None

    def average_parameters(self, params: Union[Iterable[torch.nn.Parameter], Iterable[Dict[str, torch.nn.Parameter]]]):
        """
        Averages parameters or parameter groups of an optimizer if ``step`` is no less than ``warmup_steps``
        and it can be divided by a period in the keys of ``period_process_group_dict``,
        where ``step`` is increased by 1 at each iteration in the training loop.
        If ``step`` can be divided by multiple periods in the keys of ``period_process_group_dict``,
        only the largest period is used, and the corresponding process group is used for averaging parameters.
        Args:
            params: The parameters of a model or parameter groups of an optimizer.
        """
        if self.step >= self.warmup_steps:
            group = self._find_process_group()
            if group is not None:
                utils.average_parameters_or_parameter_groups(params, group)
        self.step += 1
