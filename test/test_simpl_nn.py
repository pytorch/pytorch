import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x
    

torch._inductor.config.freezing = True
dtype = torch.float16

aModule = SimpleNN().to(dtype).xpu()

with torch.no_grad():
    compiled_module = torch.compile(aModule)

    rand_input = torch.randn(1, 3, 224, 224, dtype=dtype).xpu()
    compiled_module(rand_input)

    compiled_module(rand_input)
