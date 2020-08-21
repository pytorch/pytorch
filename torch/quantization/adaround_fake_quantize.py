import torch
import torch.nn.init as init
import math
from torch.quantization.fake_quantize import FakeQuantize
from torch.quantization.observer import MovingAverageMinMaxObserver
from torch.quantization.qconfig import *
from torch.quantization.fake_quantize import *


class AdaRoundFakeQuantize(FakeQuantize):
    ''' This class is an extension of the FakeQuantize module and is used similarly.
    The difference between the two is that you can use the continous_V attribute to alter the
    default rounding scheme during quantization. The defualt is to round to the nearest number,
    but applying an optimizer on this attribute can yield a better rounding scheme and improve
    quantization results.

    Attributes:
        continous_V:
        beta_high:
        beta_low:
        norm_scaling:
        regulatizaion_scaling:
    '''

    def __init__(self, observer=MovingAverageMinMaxObserver, quant_min=-128, quant_max=127,
                 dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False,
                 beta_high=8, beta_low=2, norm_scaling=10, regularization_scaling=.1, continous_V=None):
        self.enable_adaround = False
        self.continous_V = continous_V
        self.beta_high = beta_high
        self.beta_low = beta_low
        self.norm_scaling = norm_scaling
        self.regulatizaion_scaling = regularization_scaling
        super(AdaRoundFakeQuantize, self).__init__(observer=MovingAverageMinMaxObserver,
                                                   quant_min=-128, quant_max=127, dtype=torch.qint8,
                                                   qscheme=torch.per_tensor_symmetric, reduce_range=False)

    def forward(self, X):
        if self.continous_V is None:
            # need some value already assigned to continuous_V for kaiming to work
            self.continous_V = torch.nn.Parameter(torch.zeros(X.size()))
            init.kaiming_uniform_(self.continous_V, a=math.sqrt(5))

        if self.enable_adaround:
            X = self.adaround_rounding(X)

        return super().forward(X)

    @staticmethod
    def clipped_sigmoid(continous_V):
        ''' Function to create a non-vanishing gradient for V

        Paper Reference: https://arxiv.org/pdf/2004.10568.pdf Eq. 23
        '''
        sigmoid_of_V = torch.sigmoid(continous_V)
        scale_and_add = (sigmoid_of_V * 1.2) - 0.1
        return torch.clamp(scale_and_add, 0, 1)

    def adaround_rounding(self, x):
        ''' Using the scale and continous_V parameters of the module, the given tensor x is
        rounded using adaround

        Paper Reference: https://arxiv.org/pdf/2004.10568.pdf Eq. 22
        '''
        weights_divided_by_scale = torch.div(x, self.scale)
        weights_divided_by_scale = torch.floor(weights_divided_by_scale)
        weights_clipped = weights_divided_by_scale + self.clipped_sigmoid(self.continous_V)

        weights_w_adaround_rounding = self.scale * torch.clamp(weights_clipped, self.quant_min, self.quant_max)
        return weights_w_adaround_rounding

    def layer_loss_function(self, rate, float_weight):
        ''' Calculates the loss function for a submodule

        Paper Reference: https://arxiv.org/pdf/2004.10568.pdf Eq. 25
        Note: Table 6 of the paper suggested similar results using the norm of the weights
            as opposed to the norm of the outputs and is what is used here.
        '''
        # rate=0 -> beta=beta_low, rate=1 -> beta=beta_high
        beta = rate * (self.beta_high - self.beta_low) + self.beta_low

        clipped_weight = self.adaround_rounding(float_weight)
        quantized_weight = torch.fake_quantize_per_tensor_affine(clipped_weight, float(self.scale),
                                                                 int(self.zero_point), self.quant_min,
                                                                 self.quant_max)
        Frobenius_norm = torch.norm(float_weight - quantized_weight)


        clip_V = self.clipped_sigmoid(self.continous_V)
        spreading_range = torch.abs((2 * clip_V) - 1)
        one_minus_beta = 1 - (spreading_range ** beta)
        regulization = torch.sum(one_minus_beta)

        return Frobenius_norm * self.norm_scaling + self.regularization_scaling * regulization

default_araround_fake_quant = AdaRoundFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                             quant_min=-128, quant_max=127, dtype=torch.qint8,
                                                             qscheme=torch.per_tensor_symmetric, reduce_range=False,
                                                             beta_high=8, beta_low=2, norm_scaling=10,
                                                             regularization_scaling=.1, continous_V=None)

adaround_qconfig = QConfig(activation=default_fake_quant,
                           weight=default_araround_fake_quant)
