import torch
import torchvision
import copy
import torch._dynamo
from torch.fx import symbolic_trace
from torch._dynamo import config
#from torch._dynamo.optimizations.log_args import conv_args_analysis
#from torchdynamo.optimizations import backends

from typing import Type, Dict, Any, Tuple, Iterable, Optional, List, cast, Callable

from torch.fx.passes.shape_prop import ShapeProp
import torch.fx as fx
from torch._inductor import config
config.debug = True
torch._dynamo.config.verbose=True
#config.cpp.simdlen = 8
#config.dynamic_shapes = True
#config.normalize_ir = True
import numpy as np

import math
#config.debug = True

class MockModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.temperature = np.power(128, 0.5)
        #self.temperature = 5.0

        self.pool = torch.nn.LeakyReLU()


    def forward(self, x):
        x = self.pool(x)
        return x
        #return x + self.temperature


mod = MockModule().eval()


#x = torch.randn(128, 3, 7, 7)
#.transpose(2, 3)
x = torch.randn(1, 64, 8, 9)
#.to(memory_format=torch.channels_last)
#x = torch.randn(1, 3, 224, 224).to(memory_format=torch.channels_last)

#mod = torchvision.models.resnet50().eval()

#other = torch.randn(128, 3, 7, 7).to(memory_format=torch.channels_last)
#mod = mod.to(memory_format=torch.channels_last)
#other2 = torch.randn(1, 3, 16, 16).to(memory_format=torch.channels_last)


#y = torch.cat([x, other], dim=1)
import pdb
#pdb.set_trace()
#print(y.stride())



with torch.no_grad():
    #traced = torch.jit.trace(mod, x)
    #print(traced.graph_for(x))
    opt_model = torch._dynamo.optimize('inductor')(mod)
    print("2222222222222222222222222222222222222")
    #opt_model = mod
    out = opt_model(x)
    print("333333333333333333")
    out = opt_model(x)
    out = opt_model(x)

#print(ref - out)
#print(torch.equal(ref, out))
#out.sum()
