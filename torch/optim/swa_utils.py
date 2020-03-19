import torch
from torch.nn import Module
from copy import deepcopy
from torch.optim.lr_scheduler import _LRScheduler


class AveragedModel(Module):
    r"""Implements averaged model for Stochastic Weight Averaging (SWA).
    
	Stochastic Weight Averaging was proposed in `Averaging Weights Leads to
    Wider Optima and Better Generalization`_ by Pavel Izmailov, Dmitrii
    Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson
    (UAI 2018).

	AveragedModel class creates a copy of the provided module :attr:`model`
	on the device :attr:`device` and allows to compute running averages of the 
	parameters of the :attr:`model`.

	Arguments:
		model (torch.nn.Module): model to use with SWA
		device (torch.device, optional): if provided, the averaged model will be
			stored on the :attr:`device` 
        avg_fn (function, optional):

	Example:
		>>> loader, optimizer, model = ...
		>>> swa_model = torch.optim.swa_utils.AveragedModel(model, 
		>>>										avg_function=<equal averaging>)
		>>> scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
		>>>										T_max=300)
		>>> swa_start = 160
		>>> swa_scheduler = SWALR(optimizer, start_epoch=swa_start, swa_lr=0.05)
		>>>
		>>> for i in range(300):
		>>>      for input, target in loader:
		>>>          optimizer.zero_grad()
		>>>          loss_fn(model(input), target).backward()
		>>>          optimizer.step()
		>>>          scheduler.step()
		>>>          swa_scheduler.step()
		>>> 
		>>>      if i > swa_start:
		>>>          swa_model.update_parameters(model)
		>>>
		>>> # Update bn statistics for the swa_model at the end
		>>> torch.optim.swa_utils.update_bn(loader, swa_model) 

	.. note::
        When using SWA with models containing Batch Normalization you may 
		need to update the activation statistics for Batch Normalization.
        You can do so by using :meth:`torch.optim.swa_utils.update_bn` utility.
	
	.. note::
        :attr:`avg_fun` is not saved in the :meth:`state_dict` of the model.
	
	.. note::
        When :meth:`update_parameters` is called for the first time (i.e. 
        :attr:`n_averaged` is `0`) the parameters of `model` are copied
        to the parameters of :class:`AveragedModel`. For every subsequent
        call of :meth:`update_parameters` the function `avg_fun` is used
        to update the parameters.

    .. _Averaging Weights Leads to Wider Optima and Better Generalization:
        https://arxiv.org/abs/1803.05407
    .. _There Are Many Consistent Explanations of Unlabeled Data: Why You Should
        Average:
        https://arxiv.org/abs/1806.05594
    .. _SWALP: Stochastic Weight Averaging in Low-Precision Training:
        https://arxiv.org/abs/1904.11943
    .. _Stochastic Weight Averaging in Parallel: Large-Batch Training That 
        Generalizes Well:
        https://arxiv.org/abs/2001.02312
    """
    def __init__(self, model, device=None, avg_fun=None):
        super(AveragedModel, self).__init__()
        self.module = deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer('n_averaged', torch.tensor(0, dtype=torch.long))
        if avg_fun is None:
            avg_fun = lambda p_avg, p, n_avg: p_avg + (p - p_avg) / (n_avg +1)
        self.avg_fun = avg_fun
        
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def update_parameters(self, model):
        for p_swa, p_model in zip(self.parameters(), model.parameters()):
            device = p_swa.device
            p_model_ = p_model.data.to(device)
            if self.n_averaged == 0:
                p_swa.data.copy_(p_model_)
            else:
                p_swa.data.copy_(self.avg_fun(p_swa.data, p_model_,
                                              self.n_averaged))
        self.n_averaged += 1

    
def update_bn(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be trasferred to
            :attr:`device` before being passed into :attr:`model`.
    """
    momenta = {}
    has_bn = False
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum
            has_bn = True

    if not has_bn:
        return
    
    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        if device is not None:
            input = input.to(device)

        model(input)

    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momenta[module]
    model.train(was_training)


class SWALR(_LRScheduler):
    """
    ToDo
    """
    def __init__(self, optimizer, swa_lr, start_epoch=None, last_epoch=-1):
        self.swa_lr = swa_lr
        self.start_epoch = start_epoch
        super(SWALR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        
        if self.start_epoch and self._step_count > self.start_epoch:
            return [self.swa_lr for group in self.optimizer.param_groups]
        return [max(group['lr'], self.swa_lr) for group in self.optimizer.param_groups]
