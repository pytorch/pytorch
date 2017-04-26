import numpy as np
import warnings
from bisect import bisect_right
from torch.optim.optimizer import Optimizer


class LambdaLR(object):
    def __init__(self, optimizer, base_lr, lr_lambda):
        self.optimizer = optimizer
        self.lr_lambda = lr_lambda
        self.base_lr = base_lr

    def step(self, epoch):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr * self.lr_lambda(epoch)


class GroupLambdaLR(object):
    def __init__(self, optimizer, base_lrs, lr_lambdas):
        self.zip = zip(optimizer.param_groups, base_lrs, lr_lambdas)

    def step(self, epoch):
        for param_group, base_lr, lr_lambda in self.zip:
            param_group['lr'] = base_lr * lr_lambda(epoch)


class StepLR(LambdaLR):
    """Set the learning rate to the base_lr decayed by gamma 
    every step_size epochs.
    
    
    Example:
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, base_lr=0.05, gamma=0.1, step_size=30)
        >>> for epoch in range(100):
        >>>     scheduler.step(epoch)
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, optimizer, base_lr=0.1, gamma=0.1, step_size=30):
        super(StepLR, self).__init__(optimizer, base_lr,
                                     lambda epoch: gamma ** (epoch // step_size))


class MultiStepLR(LambdaLR):
    """Set the learning rate to the base_lr decayed by gamma 
    once the number of epoch reaches one of the milestones.
    
    
    Example:
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >=80
        >>> scheduler = MultiStepLR(optimizer, base_lr=0.05, gamma=0.1, milestones=[30,80])
        >>> for epoch in range(100):
        >>>     scheduler.step(epoch)
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, optimizer, base_lr=0.1, gamma=0.1, milestones=(10, 20, 30)):
        milestones = sorted(milestones)
        super(MultiStepLR, self).__init__(optimizer, base_lr,
                                          lambda epoch: gamma ** bisect_right(milestones, epoch))


class ExponentialLR(LambdaLR):
    """Set the learning rate to the initial LR decayed by gamma in
    every epoch."""

    def __init__(self, optimizer, base_lr, gamma):
        super(ExponentialLR, self).__init__(optimizer, base_lr,
                                            lambda epoch: gamma ** epoch)


class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.
    
    Args:
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing.
        epsilon: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.
        
        
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # different from LambdaLR, step should be called after validate()
        >>>     scheduler.step(epoch, val_loss) 
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=0, epsilon=1e-4, cooldown=0, min_lr=0):

        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.monitor_op = None
        self.wait = 0
        self.best = 0
        self.mode = mode
        assert isinstance(optimizer, Optimizer)
        self.optimizer = optimizer
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['min', 'max']:
            raise RuntimeError('Learning Rate Plateau Reducing mode %s is unknown!')
        if self.mode == 'min':
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0
        self.lr_epsilon = self.min_lr * 1e-4

    def reset(self):
        self._reset()

    def step(self, epoch, metrics):
        current = metrics
        if current is None:
            warnings.warn('Learning Rate Plateau Reducing requires metrics available!', RuntimeWarning)
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    for param_group in self.optimizer.param_groups:
                        old_lr = float(param_group['lr'])
                        if old_lr > self.min_lr + self.lr_epsilon:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            param_group['lr'] = new_lr
                            if self.verbose > 0:
                                print('\nEpoch %05d: reducing learning rate to %s.' % (epoch, new_lr))
                            self.cooldown_counter = self.cooldown
                            self.wait = 0
                self.wait += 1

    def in_cooldown(self):
        return self.cooldown_counter > 0
