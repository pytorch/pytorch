# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
__all__ = ["make_fx", "dispatch_trace", "PythonKeyTracer", "pythonkey_decompose"]
from torch.fx.experimental.proxy_tensor import (
    decompose,
    dispatch_trace,
    make_fx,
    PythonKeyTracer,
)

pythonkey_decompose = decompose
