import warnings

from .base_scheduler import BaseScheduler

class DeltaSL(BaseScheduler):
    """Sets the sparsity level of each parameter group to the final sl
    times a given function. When last_epoch=-1, sets initial sl as zero.
    This is different from the `LambdaSL`, in that the function is applied
    to the sparsity level as follows:

    .. code-block::

        S = Sf + (Si - Sf) * f(epoch)
        # S -> Current sparsity level
        # Sf -> Target sparsity level
        # Si - > Initial sparsity level
        # f -> function applied to the current epoch
        # epoch -> epoch

    Args:
        sparsifier (BaseSparsifier): Wrapped sparsifier.
        sl_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in sparsifier.param_groups.
        initial_sparsity: Initial value (or values) for the sparsity
        delta_t: Number of steps before the mask is updated
        last_epoch (int): The index of last epoch. Default: -1.
        starting_epoch: The first epoch, when the sparsification starts
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    Example:
        >>> # Assuming sparsifier has two groups.
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95 ** epoch
        >>> scheduler = DeltaSL(sparsifier, sl_lambda=[lambda1, lambda2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, sparsifier, sl_lambda, last_epoch=-1,
                 initial_sparsity=0, delta_t=1,
                 starting_epoch=0, verbose=False):
        self.sparsifier = sparsifier
        self.delta_t = delta_t
        self.starting_epoch = starting_epoch

        if not isinstance(sl_lambda, (list, tuple)):
            self.sl_lambdas = [sl_lambda] * len(sparsifier.module_groups)
        else:
            if len(sl_lambda) != len(sparsifier.module_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(sparsifier.module_groups), len(sl_lambda)))
            self.sl_lambdas = list(sl_lambda)

        if not isinstance(initial_sparsity, (list, tuple)):
            self.initial_sparsity = [initial_sparsity] * len(sparsifier.module_groups)
        else:
            if len(initial_sparsity) != len(sparsifier.module_groups):
                raise ValueError("Expected {} initial_sparsities, but got {}".format(
                    len(sparsifier.module_groups), len(initial_sparsity)))
            self.initial_sparsity = list(initial_sparsity)
        super(DeltaSL, self).__init__(sparsifier, last_epoch, verbose)

    def get_sl(self):
        if not self._get_sl_called_within_step:
            warnings.warn(
                "To get the last sparsity level computed by the scheduler, "
                "please use `get_last_sl()`.")
        if self.last_epoch <= self.starting_epoch:
            return self.initial_sparsity
        return [base_sl + (init_sl - base_sl) * lmbda(self.last_epoch)
                for lmbda, base_sl, init_sl in zip(self.sl_lambdas, self.base_sl, self.initial_sparsity)]
