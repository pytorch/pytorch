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

        # torch._C._jit_nezha_update_graph(module_1st._c, module_2nd._c)
        all_C_modules = torch._C._jit_nezha_split_modules(module_1st._c)
        
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

def measure_perf(model_pytorch, model_nezha, input):
    print("========= Perf Measurement =========")
    @contextmanager
    def track_infer_time(title: str):
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        print("======== {} took {} ms ========".format(title, (end - start) * 10))

    # Inference time
    with track_infer_time("Nezha Model"):
        for i in range(50):
            ort_outputs = model_nezha(input)
    with track_infer_time("PyTorch Model"):
        for i in range(50):
            outputs = model_pytorch(input)

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
    pytorch_model = copy.deepcopy(my_model)

    my_results = my_model(dummy_input)

    new_model = SmartModule(my_model)
    new_outputs = new_model(dummy_input)

    [np.testing.assert_allclose(my_results, new_outputs, rtol=1e-03, atol=1e-05)]
    print('===================== Results are expected ===========================')
print('End')
