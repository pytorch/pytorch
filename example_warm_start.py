import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import torch._inductor.config
torch.set_default_device('cuda')
import time

encoder_layers = [nn.TransformerEncoderLayer(d_model=512, nhead=8) for _ in range(64)]

encoder_layers = [torch.compile(layer, options={"aot_cache": "transformer.py"}) for layer in encoder_layers]
# encoder_layers = [torch.compile(layer) for layer in encoder_layers]

def all_layers(x):
    for layer in encoder_layers:
        x = layer(x)
    return x

with torch.no_grad():
    print(torch.randn(4))
    torch.cuda.synchronize()
    begin = time.time()
    all_layers(torch.rand(10, 32, 512))
    torch.cuda.synchronize()
    print(time.time()-begin)
