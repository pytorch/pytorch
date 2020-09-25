import torch
import torch.nn as nn
import torch.nn.functional as F
from fx_grad import grad

class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(57600, 128)

        # TODO: I can't figure out how to create a constant inside
        # an FX graph so I've hardcoded this here. I need this 1. to seed
        # gradient computation.
        self.ones = torch.tensor(1.)

    def forward(self, x, target):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        prediction = self.fc1(x)

        # FIXME: Normally, the loss computation is OUTSIDE the module.
        # However, with FX, you can only symbolic trace over Modules,
        # so we've weirdly put the loss computation here.
        diff = torch.sub(prediction, target)
        losses = torch.mul(diff, diff)
        return torch.sum(losses)

x = torch.rand(5, 1, 32, 32)
targets = torch.ones(5, 128)

# Set up a SimpleConvNet() as the baseline
torch.manual_seed(0)
net = SimpleConvNet()
x.requires_grad_(True)
net(x, targets).backward()

# Set up grad_loss via fx.
x.requires_grad_(False)
torch.manual_seed(0)
net2 = SimpleConvNet()
torch.manual_seed(0)
grad_loss = grad(SimpleConvNet())
grads = grad_loss(x, targets)

# Check the grads
def check_grad(param):
    actual = grads[param]
    mod, param_ = param.split('.')
    expected = getattr(getattr(net, mod), param_).grad
    if torch.allclose(actual, expected, rtol=1e-03, atol=1e-07):
        return
    print(f'Mismatch for {param} of {(actual - expected).abs().max()}')
    assert False

# Prints the graph. Take note of all of the extra ops that compute grad.
# We probably want to run CSE / DCE on this.
print(grad_loss.code)

# Run some tests that check parity between the symbolic grad output and
# regular autograd.
check_grad('conv2.weight')
check_grad('conv2.bias')
check_grad('conv1.weight')
check_grad('conv1.bias')
check_grad('fc1.weight')
check_grad('fc1.bias')
print("All tests passed!")
