import time
import torch
from torch.jit.mobile import _load_for_lite_interpreter
from torch import nn, use_deterministic_algorithms
from torch.utils.mobile_optimizer import (LintCode,
                                          generate_mobile_module_lints,
                                          optimize_for_mobile)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

inputs = torch.randn(10, 28 * 28)
labels = torch.randn(10)

net = NeuralNetwork().to('cpu')
net.zero_grad()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
out = net(inputs)
criterion = nn.MSELoss()

loss = criterion(out, labels)
loss.backward()
print('loss is ', loss)
optimizer.step()

net.eval()
print('Net is:', net)

scripted = torch.jit.script(net)
mobiled = optimize_for_mobile(scripted)
print(mobiled.graph)

mobiled._save_for_lite_interpreter('qihan_model_false.pt', use_flatbuffer=False)
mobiled._save_for_lite_interpreter('qihan_model_true.pt', use_flatbuffer=True)


#model2 = torch.jit.load('/home/qihan/local/pytorchmodel_2_190000.pt')
#model2.eval()
#model2m = optimize_for_mobile(model2)
#model2m._save_for_lite_interpreter('model_large_true.pt', use_flatbuffer=True)
#model2m._save_for_lite_interpreter('model_large_false.pt', use_flatbuffer=False)
#
model3 = torch.jit.load('model_25.pt')
model3.eval()
# model3 = optimize_for_mobile(model3)
model3._save_for_lite_interpreter('model_25_false.pt', use_flatbuffer=False)
model3._save_for_lite_interpreter('model_25_true.pt', use_flatbuffer=True)

model4 = torch.jit.load('/home/qihan/mobilenet_v2.pt')
model4.eval()
# model3 = optimize_for_mobile(model3)
model4._save_for_lite_interpreter('mobilenet_v2_false.pt', use_flatbuffer=False)
model4._save_for_lite_interpreter('mobilenet_v2_true.pt', use_flatbuffer=True)

#model5 = torch.jit.load('pytorchmodel.pt')
#model5._save_for_lite_interpreter('milan_false.pt', use_flatbuffer=False)
#model5._save_for_lite_interpreter('milan_true.pt', use_flatbuffer=True)
