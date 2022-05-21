"""
This script generates a CSV table with all ATen operators
supported by `torch.onnx.export`. The generated table is included by
docs/source/onnx_supported_aten_list.rst.
"""

import os
from torch.onnx import onnx_supported_ops

# Constants
BUILD_DIR = 'build'
AUTO_GEN_ATEN_OPS_CSV_FILE = 'auto_gen_aten_op_list.csv'

os.makedirs(BUILD_DIR, exist_ok=True)

aten_list = onnx_supported_ops.onnx_supported_ops()

with open(os.path.join(BUILD_DIR, AUTO_GEN_ATEN_OPS_CSV_FILE), 'w') as f:
    f.write('Operator,opset_version(s)\n')
    for name, opset_version in aten_list:
        f.write(f'"``{name}``","{opset_version}"\n')
