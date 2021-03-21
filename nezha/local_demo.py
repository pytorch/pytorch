import io
import torch
import numpy as np
from torch import nn

from module_def import SmartModule
from contextlib import contextmanager

import time
import copy

import nezha_helper

torch.ops.load_library("/home/jay/repos/fatcat-z/pytorch/jay_my_ops/build/libonnx_ops.so")

############################## Preparation ##########################################################

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
        out = torch.ops.onnx_ops.dummy_ops(out, out.shape[1])        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.relu(out)

        return out

class TestModule(torch.nn.Module):
    def __init__(self, file_name):
        super(TestModule, self).__init__()
        self.file_name = file_name

    def forward(self, x, y:str):
        new_y = '0' + y
        new_size = int(new_y)

        output = torch.ops.onnx_ops.dummy_ops(x, x.shape[1])
        return output

def export_c_module(m, inputs, outputs, file_name):
    local_module = torch.jit.trace(DummyModule(), torch.ones(1))
    local_module._c = m
    torch.onnx.export(local_module, inputs, file_name, example_outputs=outputs)

#####################################################################################################


# test_onnx_cls = torch.classes.nezha_classes.ONNXRuntimeClass()

total_input_size=5
total_hidden_size=4
total_num_classes=10
dummy_input = torch.randn(32, 5)

with torch.no_grad():

    # new_module = TestModule("my_onnx.onnx")
    # new_module.eval()
    # new_output = new_module(dummy_input, '5')

    # new_script_module = torch.jit.script(new_module)

    # x_module = torch._C._jit_nezha_update_ops(new_script_module._c)

    # print(x_module)

    # new_script_module._c = x_module
    # test_output = new_script_module(dummy_input, '5')

    # torch.onnx.export(new_module, (dummy_input, '5'), "test_example.onnx", verbose=True)

    # all_C_modules = nezha_helper.split_modules(new_script_module._c)


    my_model = NeuralNet_All(total_input_size, total_hidden_size, total_num_classes)
    my_model.eval()
    output = my_model(dummy_input)

    # script_module = torch.jit.trace(my_model, dummy_input)
    script_module = torch.jit.script(my_model)
    script_module.eval()

    script_module._c = torch._C._jit_nezha_update_ops(script_module._c)

    # export_c_module(script_module._c, dummy_input, output, "my_nezha_test.onnx")

    # torch.onnx.export(script_module, dummy_input, "good_example.onnx", example_outputs=output)

    # module_file_name = "test_nezha.pt"
    # script_module.save(module_file_name)
    # torch.jit.save(script_module, module_file_name)

    # print(test_onnx_cls.inference(module_file_name, dummy_input, output))
    # print(nezha_helper.ort_inference(module_file_name, dummy_input, output))

    # torch.ops.nezha_ops.split_graph(script_module._)

    # pytorch_model = copy.deepcopy(my_model)

    # my_results = my_model(dummy_input)

    # my_model = SmartModule(my_model)

    # test_model = torch.jit.trace(my_model, torch.randn(1, 2))

    # final_outputs = my_model(dummy_input)

    # [np.testing.assert_allclose(my_results, final_outputs, rtol=1e-03, atol=1e-05)]
    # print('===================== Results are expected ===========================')

    # # measure performance
    # measure_perf(pytorch_model, my_model, dummy_input)

print('End')
