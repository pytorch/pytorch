from __future__ import absolute_import, division, print_function, unicode_literals
from .convert_modules import *  # noqa: F401
from .observer import *  # noqa: F401
from .QConfig import *  # noqa: F401

def default_eval_fn(model, calib_data):
    r"""
    Default evaluation function takes a torch.utils.data.Dataset or a list of
    input Tensors and run the model on the dataset
    """
    for data in calib_data:
        model(data)

_all__ = [

]
