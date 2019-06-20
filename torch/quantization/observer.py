from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn as nn
import torch

class Observer(nn.Module):
    def __init__(self, config):
        super(Observer, self).__init__()
        self.config = config
        self.stats = torch.Tensor([-1, 1])
        # Symmetric range for initialization
        self.avg_constant = 0.9

    def forward(self, x):
        self.stats = (1 - self.avg_constant) * self.stats + self.avg_constant * torch.tensor([torch.min(x), torch.max(x)])

    def calculate_q_params(self):
        # TODO: Once q_scheme lands update this line
        if self.config.q_scheme == 'per_tensor_symmetric':
            if self.config.q_dtype == torch.qint8:
                qparams = torch.zeros(2)
                nLevels = 255.0
                qparams[0] = 2 * torch.max(self.stats[1], -self.stats[0]) / nLevels
                qparams[1] = 0
            else:
                print('quint8')
                qparams = torch.zeros(2)
                nLevels = 255.0
                qparams[0] = 2 * torch.max(self.stats[1], -self.stats[0]) / nLevels
                qparams[1] = 128
        else:
            print('q_scheme not supported', self.config.q_scheme)

        return qparams

class WeightObserver(Observer):
    r""" Forward hook function inserted for wrap and quant modules.
     Observer updates the stats parameter of wrap/quant modules with statistics
     for use later.
    """

    def __init__(self, config):
        super(WeightObserver, self).__init__(config)
        self.stats = torch.Tensor([-2, 2])

    def forward(self, x):
        self.stats = torch.tensor([torch.min(x), torch.max(x)])
        return x
