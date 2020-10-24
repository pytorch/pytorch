import torch
import numpy

class AdaBound(torch.optim.Optimizer):
    def __init__(self, parameters, initial_lr=1e-5, betas=(0.9, 0.99), final_lr=1e-2, gamma=1e-3,
                 eps=1e-7, weight_decay=0.000009):
        """
        This is an implementation of the paper "Adaptive Gradient Methods With dynamic Bound of Learning Rate,
        https://openreview.net/pdf?id=Bkg3g2R9FX "

        :param parameters: Model parameters
        :param initial_lr: The initial step size for Adam (As described in the paper, The optim essentially starts as
        Adam and goes towards SGD. )
        :param betas: Betas.
        :param final_lr: The learning rate for SGD.
        :param gamma: Convergence rate for adam.
        :param eps: epsilon
        :param weight_decay: L2 penalty, defaults to 0.000009
        """
        optim_parameters = dict(initial_lr=initial_lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                                weight_decay=weight_decay)
        super(AdaBound, self).__init__(parameters, optim_parameters)
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.betas = betas
        self.gamma = gamma
        self.eps = eps
        self.weight_decay = weight_decay
        self.lrs = list(map(lambda group: group['initial_lr'], self.param_groups))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for param, lr in zip(self.param_groups, self.lrs):
            for p in param['params']:
                if p.grad is None:
                    continue
                gradient = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['avg'] = torch.zeros_like(p.data) # Exponential Moving averages
                    state['avg_squared'] = torch.zeros_like(p.data)

                average, average_squared = state['avg'],  state['avg_squared']
                b1, b2 = param['betas']
                state['step'] += 1
                if param['weight_decay'] != 0:
                    gradient = gradient.add_(param['weight_decay'], p.data)

                average.mul_(b1).add_(1-b1, gradient)
                average_squared.mul_(b2).addcmul_(1-b2, gradient, gradient)
                div = average_squared.sqrt().add_(param['eps'])
                bias1 = 1 - b1**state['step']
                bias2 = 1 - b2**state['step']
                stepSize = param['initial_lr']*numpy.sqrt(bias2)/bias1

                learning_rate = param['final_lr']*param['initial_lr']/lr
                lower_bound = learning_rate*(1 - 1/(param['gamma']*state['step']+1))
                upper_bound = learning_rate*(1 + 1/(param['gamma']*state['step']+1))
                stepSize = torch.full_like(div, stepSize)
                stepSize.div_(div).clamp_(lower_bound, upper_bound).mul_(average)
                p.data.add_(-stepSize)

        return loss