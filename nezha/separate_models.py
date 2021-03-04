import torch
import torchvision

import numpy as np
from torch import nn
from torch.onnx import utils

import onnxruntime as ort
import onnx
import io


old_call = torch._C.ScriptMethod.__call__

def prof_meth_call(*args, **kwargs):
    return old_call(*args, **kwargs)

torch._C.ScriptMethod.__call__ = prof_meth_call

import onnxruntime as ort
import onnx
import io


old_call = torch._C.ScriptMethod.__call__

def prof_meth_call(*args, **kwargs):
    return old_call(*args, **kwargs)

torch._C.ScriptMethod.__call__ = prof_meth_call

def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()

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

class NeuralNet_1st(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet_1st, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        return out

class NeuralNet_2nd(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(NeuralNet_2nd, self).__init__()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc2(x)
        out = self.relu(out)
        out = self.relu(out)        
        return out

class BadModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BadModule, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = torch.cumsum(out, dim=0)
        out1 = self.fc2(out)
        out2 = self.relu(out1)
        out2 = self.relu(out2)

        return out, out1, out2

total_input_size=5
total_hidden_size=4
total_num_classes=10

dummy_input = torch.randn(10, 5)
# x = dummy_input
x = torch.randn(32, 5)
y = torch.randn(15, 6)


print('Start normal model.')
with torch.no_grad():
    test_module = BadModule(total_input_size, total_hidden_size, total_num_classes)
    test_module.eval()

    temp_results = test_module(dummy_input)

    f = io.BytesIO()
    test_trace_module = torch.jit.trace(test_module, dummy_input)
    
    full_graph, unsupported_ops = utils._find_missing_ops_onnx_export(test_module, (dummy_input,), f,
                                                                     opset_version=9)
    m_all = NeuralNet_All(total_input_size, total_hidden_size, total_num_classes)
    m_all.eval()

    temp_results = m_all(dummy_input)
    test_trace_module = torch.jit.trace(m_all, dummy_input)
    
    full_graph, unsupported_ops = utils._find_missing_ops_onnx_export(m_all, (dummy_input,), f,
                                                                     opset_version=9)

    all_params = m_all.state_dict()
print('----End normal model.')

# print('----Start script model.')
# with torch.no_grad():
#     trace_m_all_01 = torch.jit.trace(NeuralNet_All(total_input_size, total_hidden_size, total_num_classes), x)
#     trace_m_all_01.load_state_dict(all_params, strict=False)
#     trace_m_all_01.eval()
#     trace_results_all_before = trace_m_all_01(dummy_input)

#     print('----Finish trace_m_all_01.')

#     # torch.onnx.export(trace_m_all_01, (x, ), 'test_model_before.onnx', example_outputs=trace_results_all_before)
#     # ort_sess = ort.InferenceSession('test_model_before.onnx')
#     # input_name = ort_sess.get_inputs()[0].name
#     # label_name = ort_sess.get_outputs()[0].name
#     # my_input, _ = torch.jit._flatten(x)
#     # my_inputs = [to_numpy(inp) for inp in my_input]

#     # ort_outs = ort_sess.run([label_name], {input_name: my_inputs[0]})
#     # [np.testing.assert_allclose(trace_results_all_before, ort_outs[0], rtol=1e-03, atol=1e-05)]

#     trace_m_all_02 = torch.jit.trace(NeuralNet_All(total_input_size, total_hidden_size, total_num_classes), x)
#     trace_m_all_02.load_state_dict(all_params, strict=False)
#     trace_m_all_02.eval()    
#     print('----Finish trace_m_all_02.')

#     # Split the overall module into 2 parts. This is an in-place operation.
#     torch._C._jit_nezha_update_graph(trace_m_all_01._c, trace_m_all_02._c)

#     trace_m_1 = torch.jit.trace(NeuralNet_1st(total_input_size, total_hidden_size), x)
#     trace_m_1.load_state_dict(all_params, strict=False)
#     trace_m_1.eval()
#     print('----Finish trace_m_1.')

#     # Compare results of different modules
#     trace_results_1 = trace_m_1(dummy_input)
#     # trace_results_new_01 = trace_m_all_01.forward(dummy_input)
#     # [np.testing.assert_allclose(trace_results_new_01, trace_results_1, rtol=1e-03, atol=1e-05)]
#     # print('----trace_m_1 results are same as first splitte module.')

#     # Save the first part into ONNX and get the result via ORT.
#     torch.onnx.export(trace_m_all_01, (dummy_input, ), 'test_model_01.onnx', example_outputs=trace_results_1)
#     ort_sess = ort.InferenceSession('test_model_01.onnx')
#     input_name = ort_sess.get_inputs()[0].name
#     label_name = ort_sess.get_outputs()[0].name

#     my_input, _ = torch.jit._flatten(dummy_input)
#     my_inputs = [to_numpy(inp) for inp in my_input]

#     ort_outs = ort_sess.run([label_name], {input_name: my_inputs[0]})
#     [np.testing.assert_allclose(trace_results_1, ort_outs[0], rtol=1e-03, atol=1e-05)]
#     print('----trace_m_1 results are same as ORT results of first splitte module.')

#     # # Construct the second part of the module
#     # trace_m_2 = torch.jit.trace(NeuralNet_2nd(total_hidden_size, total_num_classes), trace_results_1)
#     # trace_m_2.load_state_dict(all_params, strict=False)
#     # trace_m_2.eval()

#     trace_results_new_02 = trace_m_all_02(trace_results_1)
#     #trace_results_2 = trace_m_2(trace_results_1)
#     #[np.testing.assert_allclose(trace_results_new_02, trace_results_2, rtol=1e-03, atol=1e-05)]

#     [np.testing.assert_allclose(trace_results_all_before, trace_results_new_02, rtol=1e-03, atol=1e-05)]

    result_2 = m_2(result_1)
    [np.testing.assert_allclose(results_all, result_2, rtol=1e-03, atol=1e-05)]
    
print('----End normal model.')

print('----Start script model.')
with torch.no_grad():
    trace_m_all_01 = torch.jit.trace(NeuralNet_All(total_input_size, total_hidden_size, total_num_classes), x)
    trace_m_all_01.load_state_dict(all_params, strict=False)
    trace_m_all_01.eval()
    trace_results_all_before = trace_m_all_01(dummy_input)

    print('----Finish trace_m_all_01.')

    # torch.onnx.export(trace_m_all_01, (x, ), 'test_model_before.onnx', example_outputs=trace_results_all_before)
    # ort_sess = ort.InferenceSession('test_model_before.onnx')
    # input_name = ort_sess.get_inputs()[0].name
    # label_name = ort_sess.get_outputs()[0].name
    # my_input, _ = torch.jit._flatten(x)
    # my_inputs = [to_numpy(inp) for inp in my_input]

    # ort_outs = ort_sess.run([label_name], {input_name: my_inputs[0]})
    # [np.testing.assert_allclose(trace_results_all_before, ort_outs[0], rtol=1e-03, atol=1e-05)]

    trace_m_all_02 = torch.jit.trace(NeuralNet_All(total_input_size, total_hidden_size, total_num_classes), x)
    trace_m_all_02.load_state_dict(all_params, strict=False)
    trace_m_all_02.eval()    
    print('----Finish trace_m_all_02.')

    # Split the overall module into 2 parts. This is an in-place operation.
    torch._C._jit_nezha_update_graph(trace_m_all_01._c, trace_m_all_02._c)

    trace_m_1 = torch.jit.trace(NeuralNet_1st(total_input_size, total_hidden_size), x)
    trace_m_1.load_state_dict(all_params, strict=False)
    trace_m_1.eval()
    print('----Finish trace_m_1.')

    # Compare results of different modules
    trace_results_1 = trace_m_1(dummy_input)
    # trace_results_new_01 = trace_m_all_01.forward(dummy_input)
    # [np.testing.assert_allclose(trace_results_new_01, trace_results_1, rtol=1e-03, atol=1e-05)]
    # print('----trace_m_1 results are same as first splitte module.')

    # Save the first part into ONNX and get the result via ORT.
    torch.onnx.export(trace_m_all_01, (dummy_input, ), 'test_model_01.onnx', example_outputs=trace_results_1)
    ort_sess = ort.InferenceSession('test_model_01.onnx')
    input_name = ort_sess.get_inputs()[0].name
    label_name = ort_sess.get_outputs()[0].name

    my_input, _ = torch.jit._flatten(dummy_input)
    my_inputs = [to_numpy(inp) for inp in my_input]

print('End')
