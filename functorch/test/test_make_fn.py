import torch
import functorch

class MyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(3, 3, 3)
        self.bn = torch.nn.BatchNorm1d(3)
    
    def forward(self, x):
        return self.bn(self.conv1d(x))

net = MyNet()

x = torch.randn(10, 3, 3)
y = net(x).sum()

fn, params, buffers = functorch.make_functional_with_buffers(net)

def fn_model(params, buffers, x):
    return fn(params, buffers, x).sum()

y1 = fn_model(params, buffers, x)

torch.testing.assert_close(y, y1)

functorch.grad(fn_model)(params, buffers, x)
