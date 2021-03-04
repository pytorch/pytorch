import torch
import torchvision

import numpy as np
from torch import nn
from torch.onnx import utils

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
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc2(x)
        return out


total_input_size=5
total_hidden_size=4
total_num_classes=10

dummy_input = torch.randn(10, 5)
x = torch.randn(32, 5)


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

    # results_all = m_all(x)
    results_all = m_all(dummy_input)
    print("Finish model all inference.")

    m_1 = NeuralNet_1st(total_input_size, total_hidden_size)
    m_1.load_state_dict(all_params, strict=False)    
    m_1.eval()

    # result_1 = m_1(x)
    result_1 = m_1(dummy_input)

    m_2 = NeuralNet_2nd(total_hidden_size, total_num_classes)
    m_2.load_state_dict(all_params, strict=False)    
    m_2.eval()

    result_2 = m_2(result_1)

    [np.testing.assert_allclose(results_all, result_2, rtol=1e-03, atol=1e-05)]

print('End normal model.')

print('Start script model.')
with torch.no_grad():
    trace_m_all = torch.jit.trace(NeuralNet_All(total_input_size, total_hidden_size, total_num_classes), x)
    trace_m_1 = torch.jit.trace(NeuralNet_1st(total_input_size, total_hidden_size), x)
    
    trace_m_all.load_state_dict(all_params, strict=False)
    trace_m_all.eval()

    trace_m_1.load_state_dict(all_params, strict=False)
    trace_m_1.eval()
    trace_results_1 = trace_m_1(x)

    f = 'model_before.pt'
    torch.jit.save(trace_m_all, f)
    loaded_m_before = torch.jit.load(f)

    torch._C._jit_nezha_update_graph(trace_m_all._c, trace_m_1.graph)

    f = 'model_after.pt'
    torch.jit.save(trace_m_all, f)
    loaded_m_after = torch.jit.load(f)


    trace_m_2 = torch.jit.trace(NeuralNet_2nd(total_hidden_size, total_num_classes), trace_results_1)
    trace_m_2.load_state_dict(all_params, strict=False)
    trace_m_2.eval()

    trace_results_all = trace_m_all(dummy_input)

    trace_results_1 = trace_m_1(dummy_input)
    [np.testing.assert_allclose(result_1, trace_results_1, rtol=1e-03, atol=1e-05)]

    trace_results_2 = trace_m_2(trace_results_1)

    [np.testing.assert_allclose(trace_results_all, results_all, rtol=1e-03, atol=1e-05)]

    [np.testing.assert_allclose(trace_results_all, trace_results_2, rtol=1e-03, atol=1e-05)]

print('End')

# print("Start trace evaluation.")
# yyy = torch.jit.trace(m, x)

# print("------Code:")
# print(yyy.code)
# print("------Graph:")
# print(yyy.graph)

# print("End trace evaluation.")


#results = yyy(x)

#print("PyTorch Results are out.: {}".format(results))

#torch.onnx.export(m, (x,), 'model.onnx', verbose=True)

# import onnx
# om = onnx.load('model.onnx')
