from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import torch
import torch.nn as nn
import torch.nn.intrinsic as nni
import torch.nn.mobile as mobile

def freeze(model, inplace=False):
    if not inplace:
        model = copy.deepcopy(model)

    # Transformations are applied in passes.  Order matters!!
    mappings = [
        { nni.ConvReLU2d : mobile.Conv2dReLU },
        { nni.LinearReLU : mobile.LinearReLU },
        { nn.Conv2d : mobile.Conv2d },
        { nn.Linear : mobile.modules.Linear }
    ]

    def apply(model, mapping):
        def set(model, name, module):
            path = name.split('.')
            nodes = path[:-1]
            current = model
            for node in nodes:
                current = getattr(current, node)

            setattr(current, path[-1], module)

        for name, module in model.named_modules():
            if type(module) in mapping:
                set(model, name, mapping[type(module)](module))

    for mapping in mappings:
        apply(model, mapping)

    return model.eval()
