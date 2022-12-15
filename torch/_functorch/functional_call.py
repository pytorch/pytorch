# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Union, Any, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def functional_call(
    module: 'torch.nn.Module',
    parameter_and_buffer_dicts: Union[Dict[str, Tensor], Tuple[Dict[str, Tensor]]],
    args: Union[Any, Tuple],
    kwargs: Dict[str, Any] = None,
):
    parameters_and_buffers = parameter_and_buffer_dicts
    if isinstance(parameter_and_buffer_dicts, tuple):
        keys = [parameter_and_buffer.keys() for parameter_and_buffer in parameter_and_buffer_dicts]
        for key in keys:
            if keys.count(key) > 1:
                raise ValueError(f"{key} appeared in multiple dictionaries; behavior of functional call is ambiguous")

        parameters_and_buffers = {k: v for d in parameter_and_buffer_dicts for k, v in d.items()}

    return nn.utils.stateless.functional_call(module, parameters_and_buffers, args, kwargs)
