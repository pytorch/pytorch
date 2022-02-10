import torch
import torch.distributed.algorithms.model_averaging.averagers as averagers


class PostLocalSGDOptimizer(torch.optim.Optimizer):
    r"""
    Wraps an arbitrary :class:`torch.optim.Optimizer` and runs `post-local SGD <https://arxiv.org/abs/1808.07217>`_,
    This optimizer runs local optimizer at every step.
    After the warm-up stage, it averages parameters periodically afer the local optimizer is applied.

    Args:
        optim: The local optimizer.
        averager: A model averager instance to run post-localSGD algorithm.

    Example::

        >>>  import torch
        >>>  import torch.distributed as dist
        >>>  import torch.distributed.algorithms.model_averaging.averagers as averagers
        >>>  import torch.nn as nn
        >>>  from torch.distributed.optim import PostLocalSGDOptimizer
        >>> from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import post_localSGD_hook
        >>>
        >>>  model = nn.parallel.DistributedDataParallel(
        >>>     module, device_ids=[rank], output_device=rank
        >>>  )
        >>>
        >>>  # Register a post-localSGD communication hook.
        >>>  subgroup, subgroups = dist.new_subgroups()
        >>>  state = PostLocalSGDState(subgroup=subgroup, start_localSGD_iter=100)
        >>>  model.register_comm_hook(state, post_localSGD_hook)
        >>>
        >>>  # Create a post-localSGD optimizer that wraps a local optimizer.
        >>>  # Note that ``warmup_steps`` used in ``PostLocalSGDOptimizer`` must be the same as
        >>>  # ``start_localSGD_iter`` used in ``PostLocalSGDState``.
        >>>  local_optim = torch.optim.SGD(params=model.parameters(), lr=0.01)
        >>>  opt = PostLocalSGDOptimizer(
        >>>      optim=local_optim,
        >>>      averager=averagers.PeriodicModelAverager(period=4, warmup_steps=100)
        >>>  )
        >>>
        >>>  # In the first 100 steps, DDP runs global gradient averaging at every step.
        >>>  # After 100 steps, DDP runs gradient averaging within each subgroup (intra-node by default),
        >>>  # and post-localSGD optimizer runs global model averaging every 4 steps after applying the local optimizer.
        >>>  for step in range(0, 200):
        >>>     opt.zero_grad()
        >>>     loss = loss_fn(output, labels)
        >>>     loss.backward()
        >>>     opt.step()
    """

    def __init__(
        self,
        optim: torch.optim.Optimizer,
        averager: averagers.ModelAverager
    ):
        self.optim = optim
        self.param_groups = self.optim.param_groups
        self.averager = averager

    @property
    def state(self):
        return self.optim.state

    def __repr__(self):
        return self.optim.__repr__()

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def step(self):
        r"""
        Performs a single optimization step (parameter update).
        """
        self.optim.step()
        for param_group in self.param_groups:
            for params in param_group["params"]:
                if params.grad is None:
                    continue
                self.averager.average_parameters(iter(params))

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)
