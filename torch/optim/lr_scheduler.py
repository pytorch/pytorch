from bisect import bisect_right
from torch.optim.optimizer import Optimizer


class LambdaLR(object):
    def __init__(self, optimizer, base_lr, lr_lambda):
        self.optimizer = optimizer
        self.base_lrs = _make_lrs_for_groups(optimizer, base_lr)
        self.lr_lambda = lr_lambda

    def step(self, epoch):
        for inx, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[inx] * self.lr_lambda(epoch)


class GroupLambdaLR(object):
    def __init__(self, optimizer, base_lrs, lr_lambdas):
        self.optimizer = optimizer
        self.base_lrs = base_lrs
        self.lr_lambdas = lr_lambdas

    def step(self, epoch):
        for param_group, base_lr, lr_lambda in zip(
                self.optimizer.param_groups,
                self.base_lrs, self.lr_lambdas):
            param_group['lr'] = base_lr * lr_lambda(epoch)


class StepLR(LambdaLR):
    """Set the learning rate to the base_lr decayed by gamma
    every step_size epochs.

    Args:
        base_lr: a scalar or a list of scalars. The learning rate
            at epoch 0 for all param groups or for each group
            respectively.
        gamma: the multiplicative factor of learning rate decay.
        step_size: period of learning rate decay.

    Example:
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, base_lr=0.05, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     scheduler.step(epoch)
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, optimizer, base_lr, step_size, gamma=0.1):
        super(StepLR, self).__init__(optimizer, base_lr,
                                     lambda epoch: gamma ** (epoch // step_size))


class MultiStepLR(LambdaLR):
    """Set the learning rate to the base_lr decayed by gamma
    once the number of epoch reaches one of the milestones.

    Args:
        base_lr: a scalar or a list of scalars. The learning rate
            at epoch 0 for all param groups or for each group
            respectively.
        gamma: the multiplicative factor of learning rate decay.
        milestones: a list of epoch indices. Must be increasing.

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

    def __init__(self, optimizer, base_lr, milestones, gamma=0.1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        super(MultiStepLR, self).__init__(optimizer, base_lr,
                                          lambda epoch: gamma ** bisect_right(milestones, epoch))


class ExponentialLR(LambdaLR):
    """Set the learning rate to the initial LR decayed by gamma in
    every epoch.

    Args:
        base_lr: a scalar or a list of scalars. The learning rate
            at epoch 0 for all param groups or for each group
            respectively.
        gamma: the multiplicative factor of learning rate decay.
    """

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
        threshold: threshold for measuring the new optimum,
            to only focus on significant changes.
        threshold_mode: one of {rel, abs}. In 'rel' mode,
            dynamic_threshold = best * ( 1 + threshold )
            in 'max' mode or best * ( 1 - threshold ) in
            'min' mode. In 'abs' mode, dynamic_threshold =
            best + threshold in 'max' mode or
            best - threshold in 'min' mode.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: a scalar or a list of scalars. A lower bound
            on the learning rate of all param groups or each
            group respectively.


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
                 verbose=True, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor
        self.min_lrs = _make_lrs_for_groups(optimizer, min_lr)
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self._make_is_better_op(mode=mode, threshold=threshold,
                                threshold_mode=threshold_mode)
        self.wait = 0
        self.best = 0
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.wait = 0
        self.lr_epsilons = [min_lr * 1e-4 for min_lr in self.min_lrs]

    def step(self, epoch, metrics):
        current = metrics

        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0

        if self.is_better(current, self.best):
            self.best = current
            self.wait = 0
        elif not self.in_cooldown():
            if self.wait >= self.patience:
                self._reduce_lr(epoch)
                self.cooldown_counter = self.cooldown
                self.wait = 0
            self.wait += 1

    def _reduce_lr(self, epoch):
        for inx_group, param_group in enumerate(self.optimizer.param_groups, 0):
            old_lr = float(param_group['lr'])
            if old_lr > self.min_lrs[inx_group] + self.lr_epsilons[inx_group]:
                new_lr = old_lr * self.factor
                new_lr = max(new_lr, self.min_lrs[inx_group])
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch %05d: reducing learning rate'
                          ' of group %d to %s.' % (epoch, inx_group, new_lr))

    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _make_is_better_op(self, mode, threshold, threshold_mode):
        if mode not in ['min', 'max']:
            raise RuntimeError('Learning Rate Plateau Reducing mode %s is unknown!')
        if threshold_mode not in ['rel', 'abs']:
            raise RuntimeError('Learning Rate Plateau Reducing'
                               ' threshold mode %s is unknown!')
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            self.is_better = lambda a, best: a < best * rel_epsilon
            self.mode_worse = float('Inf')
        elif mode == 'min' and threshold_mode == 'abs':
            self.is_better = lambda a, best: a < best - threshold
            self.mode_worse = float('Inf')
        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            self.is_better = lambda a, best: a > best * rel_epsilon
            self.mode_worse = -float('Inf')
        else:  # mode == 'max' and epsilon_mode == 'abs':
            self.is_better = lambda a, best: a > best + threshold
            self.mode_worse = -float('Inf')


def _make_lrs_for_groups(optimizer, lr):
    if isinstance(lr, list) or isinstance(lr, tuple):
        if len(lr) != len(optimizer.param_groups):
            raise ValueError('len(lr)={} does not match'
                             ' len(param_groups)={}'.format(
                                len(lr), len(optimizer.param_groups)))
    else:
        lr = [lr for _ in range(len(optimizer.param_groups))]
    return lr
