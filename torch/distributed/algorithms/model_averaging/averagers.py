import warnings
from abc import ABC, abstractmethod

import torch.distributed as dist
import torch.distributed.algorithms.model_averaging.utils as utils


class ModelAverager(ABC):
    r"""Base class for all model averagers.

    Args:
        process_group: The process group to be used for all-reduce.
                       If ``None``, the default process group, which
                       is created by :func:`torch.distributed.init_process_group`,
                       will be used. (default: ``None``)
    """

    def __init__(self, process_group=None):
        self.process_group = (
            process_group if process_group is not None else dist.group.WORLD
        )
        self.step = 0

    @abstractmethod
    def average_parameters(self, params):
        raise NotImplementedError


class PeriodicModelAverager(ModelAverager):
    r"""
    Averages parameters periodically after the warm-up stage.

    This can be used for running `post-local SGD <https://arxiv.org/abs/1808.07217>`_,
    by running :class:`~torch.nn.DistributedDataParallel` (DDP)
    using the subgroups created by :meth:`~torch.distributed.new_subgroups`.

    Args:
        period (int): The number of steps per model averaging.
                      Usually the period should be greater than ``1`` to reduce the communication cost.
                      Otherwise, only DDP needs to be used.
        warmup_steps (int): The number of warm-up steps. During this stage,
                            model averaging is skipped.
        process_group: The process group to be used for all-reduce.
                       If ``None``, the default process group, which
                       is created by :func:`torch.distributed.init_process_group`,
                       will be used. (default: ``None``)

    Example::

        >>>  import torch
        >>>  import torch.distributed as dist
        >>>  import torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook as post_localSGD
        >>>  import torch.distributed.algorithms.model_averaging.averagers as averagers
        >>>  import torch.nn as nn
        >>>
        >>>  dist.init_process_group("nccl", rank=rank, world_size=16)
        >>>  torch.cuda.set_device(rank)
        >>>  module = nn.Linear(1, 1, bias=False).to(rank)
        >>>  model = nn.parallel.DistributedDataParallel(
        >>>     module, device_ids=[rank], output_device=rank
        >>>  )
        >>>  # Register a post-localSGD communication hook.
        >>>  subgroup, subgroups = dist.new_subgroups()
        >>>  state = PostLocalSGDState(subgroup=subgroup, start_localSGD_iter=100)
        >>>  model.register_comm_hook(state, post_localSGD_hook)
        >>>
        >>>  # In the first 100 steps, run global gradient averaging like normal DDP at every step.
        >>>  # After 100 steps, run model averaging every 4 steps.
        >>>  # Note that ``warmup_steps`` must be the same as ``start_localSGD_iter`` used in ``PostLocalSGDState``.
        >>>  averager = averagers.PeriodicModelAverager(period=4, warmup_steps=100)
        >>>  for step in range(0, 20):
        >>>     optimizer.zero_grad()
        >>>     loss = loss_fn(output, labels)
        >>>     loss.backward()
        >>>     optimizer.step()
        >>>     # Average parameters globally after ``optimizer.step()``.
        >>>     # Thus, the inter-node communication only occurs periodically after ``warmup_steps``.
        >>>     averager.average_parameters(model.parameters())

    .. warning ::
        `PeriodicModelAverager` is experimental and subject to change.
    """

    def __init__(
        self,
        period,
        warmup_steps=0,
        process_group=None,
    ):
        super().__init__(process_group)
        if warmup_steps < 0:
            raise ValueError("Arg ``warmup_steps`` must be a non-negative number.")
        self.warmup_steps = warmup_steps
        if period < 1:
            raise ValueError("Arg ``period`` must be a positive value.")
        elif period == 1:
            warnings.warn(
                "When period is 1, no need to use model averaging because the communication cost "
                "of all-reducing parameters will be no less than the cost of all-reducing gradients "
                "by DistributedDataParall in the backward pass. Therefore, only "
                "DistributedDataParallel should be used for this case."
            )
        self.period = period

    def average_parameters(self, params):
        r"""
        Averages parameters if ``step`` is no less than ``warmup_steps``
        and it can be divided by ``period``, where ``step`` is increased by 1
        at each iteration in the training loop.
        """
        if (
            self.step >= self.warmup_steps
            and (self.step - self.warmup_steps) % self.period == 0
        ):
            utils.average_parameters(iter(params), self.process_group)
        self.step += 1
