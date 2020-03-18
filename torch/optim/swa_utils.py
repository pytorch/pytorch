import torch
from torch.nn import Module
from copy import deepcopy
from torch.optim.lr_scheduler import _LRScheduler


class AveragedModel(Module):
    r"""Implements averaged model for Stochastic Weight Averaging (SWA).
    
    ToDo
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
    def __init__(self, model, device=None):
        super(AveragedModel, self).__init__()
        self.module = deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer('n_averaged', torch.tensor(0, dtype=torch.long))
        
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def update_parameters(self, model):
        for p_swa, p_model in zip(self.parameters(), model.parameters()):
            device = p_swa.device
            virtual_decay = 1 / float(self.n_averaged + 1)
            diff = (p_model.data.to(device) - p_swa.data) * virtual_decay
            p_swa.data.add_(diff)
        self.n_averaged += 1

    
def bn_update(loader, model, device=None):
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
    if not _check_bn(model):
        return
    was_training = model.training
    model.train()
    momenta = {}
    model.apply(_reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        b = input.size(0)

        momentum = b / float(n + b)
        for module in momenta.keys():
            module.momentum = momentum

        if device is not None:
            input = input.to(device)

        model(input)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))
    model.train(was_training)


# BatchNorm utils
def _check_bn_apply(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def _check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn_apply(module, flag))
    return flag[0]


def _reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


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
