import torch
import torchvision

import numpy as np
from torch import nn

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


total_input_size=3
total_hidden_size=4
total_num_classes=10

dummy_input = torch.randn(10, 3, 224, 224)

with torch.no_grad():
    m_all = NeuralNet_All(total_input_size, total_hidden_size, total_num_classes)
    m_all.eval()

    all_params = m_all.state_dict()

    x = torch.randn(32, 3)
    results_all = m_all(x)
    print("Finish model all inference.")

    m_1 = NeuralNet_1st(total_input_size, total_hidden_size)
    m_1.load_state_dict(all_params, strict=False)    
    m_1.eval()

    result_1 = m_1(x)

    m_2 = NeuralNet_2nd(total_hidden_size, total_num_classes)
    m_2.load_state_dict(all_params, strict=False)    
    m_2.eval()

    result_2 = m_2(result_1)

    [np.testing.assert_allclose(results_all, result_2, rtol=1e-03, atol=1e-05)]

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
