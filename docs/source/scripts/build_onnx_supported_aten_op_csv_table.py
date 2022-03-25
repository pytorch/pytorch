"""
This script will generate a CSV table with all ATen operators
currently supported by ONNX converter. The generated table is included by
docs/source/onnx_supported_aten_list.rst and rendered on a proper public format
"""

import os
from torch.onnx import onnx_supported_ops

# Constants
BUILD_DIR = 'build'
AUTO_GEN_ATEN_OPS_CSV_FILE = 'auto_gen_aten_op_list.csv'

print('Generating list of ATen operators supported by ONNX converter')
os.makedirs(BUILD_DIR, exist_ok=True)

# Retrieve list of supported ATen operators
aten_list = onnx_supported_ops.get_onnx_supported_ops()

# Write CSV file
with open(os.path.join(BUILD_DIR, AUTO_GEN_ATEN_OPS_CSV_FILE), 'w') as f:
    f.write('Operator,Opset(s)\n')
    for name, opset in aten_list:
        f.write(f'"``{name}``","{opset}"\n')
