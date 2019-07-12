from __future__ import absolute_import, division, print_function, unicode_literals
from .quantize import *  # noqa: F401
from .observer import *  # noqa: F401
from .QConfig import *  # noqa: F401
from .fake_quantize import *  # noqa: F401
import torch

def default_eval_fn(model, calib_data):
    r"""
    Default evaluation function takes a torch.utils.data.Dataset or a list of
    input Tensors and run the model on the dataset
    """
    for data in calib_data:
        model(data)

default_loss_fn = torch.nn.MSELoss(reduction='sum')
def default_train_fn(model, train_data, loss_fn=default_loss_fn, optimizer=None):
    r"""
    Default train function takes a torch.utils.data.Dataset and train the model
    on the dataset
    """
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for i in range(10):
        for data in train_data:
            input, target = data
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

_all__ = [
    'QuantWrapper', 'QuantStub', 'DeQuantStub', 'DEFAULT_MODULE_MAPPING',
    # Top level API for quantizing a float model
    'quantize',
    # Sub functions called by quantize
    'prepare', 'convert',
    # Sub functions for `prepare` and `swap_module`
    'propagate_qconfig', 'add_quant_dequant', 'add_observer', 'swap_module',
    'default_eval_fn',
    # Observers
    'Observer', 'WeightObserver', 'observer', 'default_observer',
    'default_weight_observer',
    # QConfig
    'QConfig', 'default_qconfig',
    # QAT utilities
    'default_qat_qconfig', 'default_train_fn', 'default_loss_fn'
]
