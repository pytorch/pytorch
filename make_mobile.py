import torch
from torch import nn
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
mobiled._save_for_lite_interpreter('qihan_model.pt', use_flatbuffer=True)


# ./build/bin/flatc --json --raw-binary torch/csrc/jit/serialization/mobile_bytecode.fbs -- flatbuffer.dat