import torch
from typing import List, Dict
# from collections import defaultdict

@torch.jit.script
class AdagradJit(object):
    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0):
        # type: (List[Tensor], float, float, float, float) -> None
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))

        self.defaults = {'lr': lr, 'lr_decay': lr_decay, 'weight_decay': weight_decay,
                    'initial_accumulator_value': initial_accumulator_value}
        self.param_groups = torch.jit.annotate(List[Dict[str, List[torch.Tensor]]], [])
        self.state = torch.jit.annotate(Dict[torch.Tensor, Dict[str, torch.Tensor]], {})

        if len(params) == 0:
            raise ValueError("optimizer got an empty parameter list")

        param_groups = [{'params': params}]

        for param_group in param_groups:
            self.add_param_group(param_group)

        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = torch.jit.annotate(Dict[str, torch.Tensor], {})
                state = self.state[p]
                # TODO: no union or any types in TorchScript, make step a tensor instead
                state['step'] = torch.tensor(0)
                state['sum'] = torch.full_like(p, initial_accumulator_value)

    def add_param_group(self, param_group):
        # type: (Dict[str, List[Tensor]]) -> None
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Arguments:
            param_group (dict): Specifies what Tensors should be optimized along with group
            specific optimization options.
        """
        # check if each tensor in list of tensor is a valid tensor
        # for param in param_group['params']:
        #     if not param.is_leaf:
        #         raise ValueError("can't optimize a non-leaf Tensor")

        # for all the default hyper params, check if need to specified it in param group
        # specify the param group with the coresponding optimization options (hyper params)

        # TODO: TorchScript could not make two different value types into a single map, so we don't 
        # set the hyperparams per group, we use the global default for now. This need to be changed
        # to aligned with the general Adagrad algorithm if TorchScript support either Any or Union types

        # for name, default in self.defaults.items():
            # param_group.setdefault(name, default)

        # TODO: TorchScript does not have good coverage on Set, so we skipped the checking here and
        # delegate the checks to user. 

        # param_set = set()
        # for group in self.param_groups:
        #     param_set.update(set(group['params']))

        # if not param_set.isdisjoint(set(param_group['params'])):
        #     raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)


    def zero_grad(self):
        # type: () -> None
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
                
    # def make_sparse(grad, values):
    #     constructor = grad.new
    #     if grad_indices.dim() == 0 or values.dim() == 0:
    #         return constructor().resize_as_(grad)
    #     return constructor(grad_indices, values, size)

    def step(self):
        # type: () -> None
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                state['step'] += 1

                # if group['weight_decay'] != 0:
                if self.defaults['weight_decay'] != 0:
                    # if p.grad.is_sparse():
                    #     raise RuntimeError("weight_decay option is not compatible with sparse gradients")
                    grad = grad.add(self.defaults['weight_decay'], p)

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

                # if grad.is_sparse:
                #     grad = grad.coalesce()  # the update is non-linear so indices must be unique
                #     grad_indices = grad._indices()
                #     grad_values = grad._values()
                #     size = grad.size()

                #     state['sum'].add_(make_sparse(grad_values.pow(2)))
                #     std = state['sum'].sparse_mask(grad)
                #     std_values = std._values().sqrt_().add_(1e-10)
                #     p.data.add_(-clr, make_sparse(grad_values / std_values))
                # else:
                state['sum'].addcmul_(1, grad, grad)
                std = state['sum'].sqrt().add_(1e-10)
                p.addcdiv_(-clr, grad, std)



# def adagrad_step(params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0):
#     """Performs a single optimization step."""
#     for p in params:
#         if p.grad is not None:
#             grad = p.grad
#             state = self.state[p]

#             state['step'] += 1

#             if group['weight_decay'] != 0:
#                 if p.grad.data.is_sparse:
#                     raise RuntimeError("weight_decay option is not compatible with sparse gradients")
#                 grad = grad.add(group['weight_decay'], p.data)

#             clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

#             if grad.is_sparse:
#                 grad = grad.coalesce()  # the update is non-linear so indices must be unique
#                 grad_indices = grad._indices()
#                 grad_values = grad._values()
#                 size = grad.size()

#                 state['sum'].add_(make_sparse(grad_values.pow(2)))
#                 std = state['sum'].sparse_mask(grad)
#                 std_values = std._values().sqrt_().add_(1e-10)
#                 p.data.add_(-clr, make_sparse(grad_values / std_values))
#             else:
#                 state['sum'].addcmul_(1, grad, grad)
#                 std = state['sum'].sqrt().add_(1e-10)
#                 p.data.addcdiv_(-clr, grad, std)


# import torch.jit as jit
# import torch.nn as nn
# from collections import defaultdict
# from .optimizer import required

# class OptimizerJit(nn.Module):
#     def __init__(self, params, defaults):
#         super(OptimizerJit, self).__init__()
#         self.defaults = defaults

#         if isinstance(params, torch.Tensor):
#             raise TypeError("params argument given to the optimizer should be "
#                             "an iterable of Tensors or dicts, but got " +
#                             torch.typename(params))

#         self.state = defaultdict(dict)
#         self.param_groups = []

#         param_groups = list(params)
#         if len(param_groups) == 0:
#             raise ValueError("optimizer got an empty parameter list")
#         if not isinstance(param_groups[0], dict):
#             param_groups = [{'params': param_groups}]

#         for param_group in param_groups:
#             self.add_param_group(param_group)

#     def add_param_group(self, param_group):
#         r"""Add a param group to the :class:`Optimizer` s `param_groups`.

#         This can be useful when fine tuning a pre-trained network as frozen layers can be made
#         trainable and added to the :class:`Optimizer` as training progresses.

#         Arguments:
#             param_group (dict): Specifies what Tensors should be optimized along with group
#             specific optimization options.
#         """
#         assert isinstance(param_group, dict), "param group must be a dict"

#         # making params a list of tensors
#         params = param_group['params']
#         if isinstance(params, torch.Tensor):
#             param_group['params'] = [params]
#         elif isinstance(params, set):
#             raise TypeError('optimizer parameters need to be organized in ordered collections, but '
#                             'the ordering of tensors in sets will change between runs. Please use a list instead.')
#         else:
#             param_group['params'] = list(params)

#         # check if each tensor in list of tensor is a valid tensor
#         for param in param_group['params']:
#             if not isinstance(param, torch.Tensor):
#                 raise TypeError("optimizer can only optimize Tensors, "
#                                 "but one of the params is " + torch.typename(param))
#             if not param.is_leaf:
#                 raise ValueError("can't optimize a non-leaf Tensor")

#         # for all the default hyper params, check if need to specified it in param group
#         for name, default in self.defaults.items():
#             if default is required and name not in param_group:
#                 raise ValueError("parameter group didn't specify a value of required optimization parameter " +
#                                  name)
#             else:
#                 # specify the param group with the coresponding optimization options (hyper params)
#                 param_group.setdefault(name, default)

#         param_set = set()
#         for group in self.param_groups:
#             param_set.update(set(group['params']))

#         if not param_set.isdisjoint(set(param_group['params'])):
#             raise ValueError("some parameters appear in more than one parameter group")

#         self.param_groups.append(param_group)


# class AdagradJit(OptimizerJit):
#     def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0):
#         if not 0.0 <= lr:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if not 0.0 <= lr_decay:
#             raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
#         if not 0.0 <= weight_decay:
#             raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
#         if not 0.0 <= initial_accumulator_value:
#             raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))

#         defaults = dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
#                         initial_accumulator_value=initial_accumulator_value)
#         super(AdagradJit, self).__init__(params, defaults)

#         for group in self.param_groups:
#             for p in group['params']:
#                 state = self.state[p]
#                 state['step'] = 0
#                 state['sum'] = torch.full_like(p.data, initial_accumulator_value)

#     def share_memory(self):
#         for group in self.param_groups:
#             for p in group['params']:
#                 state = self.state[p]
#                 state['sum'].share_memory_()

#     # @jit.script_method
#     # def make_sparse(grad, values):
#     #     constructor = grad.new
#     #     if grad_indices.dim() == 0 or values.dim() == 0:
#     #         return constructor().resize_as_(grad)
#     #     return constructor(grad_indices, values, size)

#     def forward(self):
#         self.step()

#     def step(self):
#         """Performs a single optimization step.

#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue

#                 grad = p.grad.data
#                 state = self.state[p]

#                 state['step'] += 1

#                 if group['weight_decay'] != 0:
#                     if p.grad.data.is_sparse:
#                         raise RuntimeError("weight_decay option is not compatible with sparse gradients")
#                     grad = grad.add(group['weight_decay'], p.data)

#                 clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

#                 # if grad.is_sparse:
#                 #     grad = grad.coalesce()  # the update is non-linear so indices must be unique
#                 #     grad_indices = grad._indices()
#                 #     grad_values = grad._values()
#                 #     size = grad.size()

#                 #     state['sum'].add_(make_sparse(grad_values.pow(2)))
#                 #     std = state['sum'].sparse_mask(grad)
#                 #     std_values = std._values().sqrt_().add_(1e-10)
#                 #     p.data.add_(-clr, make_sparse(grad_values / std_values))
#                 # else:
#                 state['sum'].addcmul_(1, grad, grad)
#                 std = state['sum'].sqrt().add_(1e-10)
#                 p.data.addcdiv_(-clr, grad, std)



