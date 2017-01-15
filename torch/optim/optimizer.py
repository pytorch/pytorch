from collections import defaultdict

required = object()


class Optimizer(object):
    """Base class for all optimizers.

    Arguments:
        params (iterable): an iterable of :class:`Variable` s or
            :class:`dict` s. Specifies what Variables should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """

    def __init__(self, params, defaults):
        self.state = defaultdict(dict)
        self.param_groups = list(params)
        if not isinstance(self.param_groups[0], dict):
            self.param_groups = [{'params': self.param_groups}]

        param_set = set()
        for group in self.param_groups:
            group['params'] = list(group['params'])
            group_set = set(group['params'])
            if not param_set.isdisjoint(group_set):
                raise ValueError("some parameters appear in more than one "
                        "parameter group")
            param_set.update(group_set)

        for name, default in defaults.items():
            for i, group in enumerate(self.param_groups):
                if default is required and name not in group:
                    raise ValueError("parameter group " + str(i) + " didn't "
                        "specify a value of required optimization parameter "
                        + name)
                else:
                    group.setdefault(name, default)

        for group in self.param_groups:
            for param in group['params']:
                if not param.requires_grad:
                    raise ValueError("optimizing a parameter that doesn't "
                        "require gradients")

    def __getstate__(self):
        return {
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def state_dict(self):
        """Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a dict containig all parameter groups
        """
        return self.__getstate__()

    def zero_grad(self):
        """Clears the gradients of all optimized :class:`Variable` s."""
        for group in self.param_groups:
            for param in group['params']:
                param.grad.zero_()

    def step(self, closure):
        """Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        raise NotImplementedError
