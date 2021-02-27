import torch
import torchvision

import numpy as np
from torch import nn

import onnxruntime as ort
import onnx
import copy

old_call = torch._C.ScriptMethod.__call__

def prof_meth_call(*args, **kwargs):
    return old_call(*args, **kwargs)

torch._C.ScriptMethod.__call__ = prof_meth_call

def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()

############################## Preparation ##########################################################
class SmartModule(nn.Module):
    def __init__(self, model):
        super(SmartModule, self).__init__()
        self.inner_model = model

    def forward(self, input, *args):
        self.inner_model.eval()

        module_1st = torch.jit.trace(self.inner_model, input)
        module_2nd = torch.jit.trace(self.inner_model, input)

        torch._C._jit_nezha_update_graph(module_1st._c, module_2nd._c)

        all_modules = [module_1st, module_2nd]

        outputs = input
        for m in all_modules:
            outputs = m.forward(outputs, *args)
        
        return outputs

        # outputs_1st = module_1st.forward(input, *args)
        # outputs_2nd = module_2nd.forward(outputs_1st, *args)
        # return outputs_2nd

        # return self.inner_model(input, *args)

class NeuralNet_All(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet_All, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.relu(out)

        return out

#####################################################################################################

total_input_size=5
total_hidden_size=4
total_num_classes=10
dummy_input = torch.randn(32, 5)

with torch.no_grad():
    my_model = NeuralNet_All(total_input_size, total_hidden_size, total_num_classes)
    my_model.eval()
    my_results = my_model(dummy_input)

    new_model = SmartModule(my_model)
    new_outputs = new_model(dummy_input)

    [np.testing.assert_allclose(my_results, new_outputs, rtol=1e-03, atol=1e-05)]

print('End')
