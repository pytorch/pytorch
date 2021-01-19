import torch
from torchvision import models

import sys
import argparse

torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)

with torch.no_grad():
    input_shape = (1, 3, 224, 224)
    torch.manual_seed(0)
    model = models.__dict__['alexnet'](num_classes=50)
    scripted_model = torch.jit.script(model)
    scripted_model.eval()
    x = torch.rand(input_shape)
    py_output = model(x)
    scripted_model(x)
    opt_output = scripted_model(x)
    scripted_model(x)
    torch.allclose(py_output, opt_output)
    print("Done!")
