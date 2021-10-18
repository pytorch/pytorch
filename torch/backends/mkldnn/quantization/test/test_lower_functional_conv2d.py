import torch
from torch import nn
import torch.nn.functional as F
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.backends.mkldnn.quantization.lowering import lower_to_mkldnn_backend

class M(torch.nn.Module):
    def forward(self, x, w, b):
        y = F.conv2d(x, w, b)
        return y;

class M_ReLU(torch.nn.Module):
    def forward(self, x, w, b):
        y = F.conv2d(x, w, b)
        y = F.relu(y)
        return y;

# conv2d
m = M().eval()
qconfig_dict = {"": torch.quantization.default_qconfig}
m = prepare_fx(m, qconfig_dict)
m = convert_fx(m, is_reference=True)
print('Reference Model:\n', m)

m = lower_to_mkldnn_backend(m)
print('Lowered Model:\n', m)

# conv2d_relu
m = M_ReLU().eval()
qconfig_dict = {"": torch.quantization.default_qconfig}
m = prepare_fx(m, qconfig_dict)
m = convert_fx(m, is_reference=True)
print('Reference Model:\n', m)

m = lower_to_mkldnn_backend(m)
print('Lowered Model:\n', m)
