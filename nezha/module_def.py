import torch
import torchvision

import numpy as np
from torch import nn

import onnxruntime as ort
import onnx
import copy

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


class SmartModule(nn.Module):
    def __init__(self, model):
        super(SmartModule, self).__init__()
        self.inner_model = model

    def inference_by_ort(self, m, export_input):
        m.eval()
        temp_results = m.forward(export_input)
        torch.onnx.export(m, (export_input, ), 'test_model_01.onnx', example_outputs=temp_results)
        ort_sess = ort.InferenceSession('test_model_01.onnx')
        input_name = ort_sess.get_inputs()[0].name
        label_name = ort_sess.get_outputs()[0].name

        my_input, _ = torch.jit._flatten(export_input)
        my_inputs = [to_numpy(inp) for inp in my_input]

        ort_outs = ort_sess.run([label_name], {input_name: my_inputs[0]})
        return ort_outs[0]

    def forward(self, input, *args):
        self.inner_model.eval()

        module_1st = torch.jit.trace(self.inner_model, input)
        module_2nd = torch.jit.trace(self.inner_model, input)

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
