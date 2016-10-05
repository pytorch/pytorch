from copy import copy
from collections import defaultdict

required = object()

class Optimizer(object):

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

    def __getstate__(self):
        return {
            'state': self.state,
            'parameters': self.parameters,
        }

    def state_dict(self):
        return self.__getstate__()

    def _forward_backward(self, forward_closure):
        for group in self.param_groups:
            for p in group['params']:
                p.grad.zero_()
        loss = forward_closure()
        loss.backward()
        return loss

    def step(self, forward_closure):
        raise NotImplementedError

