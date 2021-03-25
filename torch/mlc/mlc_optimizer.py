import torch
from torch.optim.optimizer import Optimizer
from typing import List


class MLCOptimizer(Optimizer):
    r"""
    Implements and MLCOptimizer base class.
    """
    def __init__(self, params, defaults):
        super(MLCOptimizer, self).__init__(params, defaults)
        self.mlcopt = None
        self.torchopt = None

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # One pass to update all the MLC-updateable parameters
        assert(len(self.param_groups) == 1)

        # We make a copy of param_groups
        # We want to make a deep copy of just the list and dictionary
        # structures, but we do not want to actually copy the tensors
        param_groups_copy = []
        for group in self.param_groups:
            group_copy = group.copy()
            group_copy['params'] = group_copy['params'].copy()
            param_groups_copy.append(group_copy)

        updateable = []
        grads = []
        weightids = []
        for group in param_groups_copy:
            lr = group['lr']
            # We can't change the parameters
            if self._same_parameters(group):
                # Therefore, we need to partition the set of gradients
                for i in range(len(group['params'])):
                    p = group['params'][i]
                    # For that to work, we have the
                    # requirements that:
                    # 1) p.grad is not None (we have a gradient)
                    # 2) p.grad is a gradient node
                    # 3) p is a ValueNode
                    if not (p.grad is not None and
                            str(p.grad.device.type) == 'mlc' and
                            torch.ops.mlc.get_type(p.grad) == "MLCGradientNode" and
                            torch.ops.mlc.get_type(p) == "MLCValueNode"):
                        continue
                    updateable.append(p)
                    grads.append(p.grad)
                    weightids.append(id(p))

        modified_weights = self.mlcopt.step(updateable, grads, weightids, lr)  # type: ignore[union-attr]
        modified_weights = set(modified_weights)

        # One more pass through the parameter set to clear out
        # the set of weights we modified. Then we let the torch
        # built-in optimizer take care of the rest
        for group in param_groups_copy:
            if self._same_parameters(group):
                for i in range(len(group['params'])):
                    p = group['params'][i]
                    if str(p.device.type) == 'mlc' and torch.ops.mlc.get_id(p) in modified_weights:
                        group['params'][i] = torch.Tensor()
                    else:
                        if self.defaults.get("max_gradient_clipping") and p.grad is not None:
                            max_clip = self.defaults.get("max_gradient_clipping", float("inf"))
                            min_clip = self.defaults.get("min_gradient_clipping", float("-inf"))
                            p.grad, = torch.clamp(p.grad, min_clip, max_clip)

        self.torchopt.param_groups = param_groups_copy  # type: ignore[union-attr]
        self.torchopt.step()  # type: ignore[union-attr]

        # We need to ensure to materialize the ValueNode prior to sending it to MLC.
        # In the case that it we need to perform a `+=`, this means that the
        # node becomes a BinaryArithmetic op and we need to materialize it.
        # Hence. we just materialize any node that is not a MLCValueNode.
        for weight in updateable:
            if torch.ops.mlc.get_type(weight) != "MLCValueNode":
                weight.data.copy_(weight.data.cpu().to("mlc"))

        return loss

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad = None
        self.torchopt.zero_grad()  # type: ignore[union-attr]

    def state_dict(self):
        # shift everything into self.state
        # self.state is a dictionary of Tensor -> stuff for things like momentum
        # or sometimes just str -> stuff for other things like optimizer parameters
        # first fill with torchopt state
        # then fill with our state
        self.state = {k: {'mlc': 0, 'contents': v}
                      for k, v in self.torchopt.state.items()}  # type: ignore[union-attr]
        mlcdata = self.mlcopt.get_optimizer_data()  # type: ignore[union-attr]

        for group in self.param_groups:
            for i in range(len(group['params'])):
                p = group['params'][i]
                if id(p) in mlcdata:
                    self.state[p] = {'mlc': 1, 'contents': mlcdata[id(p)]}
        self.state['step_count'] = self.mlcopt.get_step_count()  # type: ignore[union-attr]

        ret = super().state_dict()
        self.state = None  # type: ignore[union-attr, assignment]
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # ok. rebuild from self.state
        if 'step_count' in self.state:
            self.mlcopt.set_step_count(self.state['step_count'])  # type: ignore[union-attr]

        del self.state['step_count']
        # split the rest of the parameters.
        # some will go to the torchopt, some will not
        torchstate = {}
        mlcstate = {}
        for k, v in self.state.items():
            if isinstance(v, dict) and 'mlc' in v:
                if v['mlc']:
                    mlcstate[id(k)] = v['contents']
                else:
                    torchstate[k] = v['contents']

        self.torchopt.state = torchstate  # type: ignore[union-attr]
        self.mlcopt.set_optimizer_data(mlcstate)  # type: ignore[union-attr]
        self.state = None  # type: ignore[union-attr, assignment]

    def _same_parameters(self, group):
        # Dummy same parameters
        keys: List[str] = []
        return all(self.defaults[k] == group[k] for k in keys)
