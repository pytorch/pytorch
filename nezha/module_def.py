import torch
import torchvision

import numpy as np
from torch import nn

import onnxruntime as ort
import onnx
import copy
import io

import nezha_helper

old_call = torch._C.ScriptMethod.__call__

def py_meth_call(*args, **kwargs):
    return old_call(*args, **kwargs)

torch._C.ScriptMethod.__call__ = py_meth_call

def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()

init_is_not_done = True
temp_results = torch.zeros(1)
f = io.BytesIO()
ort_sess = None
label_name = 'Unknown'
input_name = 'Unknown'
my_inputs = None
all_modules = []

class SmartModule(nn.Module):
    def __init__(self, model):
        super(SmartModule, self).__init__()
        self.inner_model = model

    def inference_by_ort(self, m, export_input):
        global init_is_not_done
        global temp_results
        global f
        global ort_sess
        global label_name
        global input_name
        global my_inputs

        m.eval()
        if (init_is_not_done):
            init_is_not_done = False
            temp_results = m.forward(export_input)
            temp_results = torch.randn_like(temp_results)
            torch.onnx.export(m, (export_input, ), f, example_outputs=temp_results)
        
            ort_sess = ort.InferenceSession(f.getvalue())
            input_name = ort_sess.get_inputs()[0].name
            label_name = ort_sess.get_outputs()[0].name

            my_input, _ = torch.jit._flatten(export_input)
            my_inputs = [to_numpy(inp) for inp in my_input]

        ort_outs = ort_sess.run([label_name], {input_name: my_inputs[0]})
        return ort_outs[0]

    def forward(self, input, *args):
        global all_modules
        self.inner_model.eval()

        if (init_is_not_done):
            module_1st = torch.jit.trace(self.inner_model, input)
            module_2nd = copy.deepcopy(module_1st)

            all_C_modules = nezha_helper.split_modules(module_1st._c)
            # all_C_modules = torch._C._jit_nezha_split_modules(module_1st._c)
            
            all_modules = [module_1st, module_2nd]
            module_length = len(all_modules)
            for i in range(module_length):
                all_modules[i]._c = all_C_modules[i]
        
        outputs = input
        use_ort = True
        for m in all_modules:
            if use_ort:
                use_ort = False;
                outputs = torch.from_numpy(self.inference_by_ort(m, outputs))
            else:
                outputs = m.forward(outputs, *args)

        return outputs

        # outputs_1st = module_1st.forward(input, *args)
        # outputs_2nd = module_2nd.forward(outputs_1st, *args)
        # return outputs_2nd

        # return self.inner_model(input, *args)
