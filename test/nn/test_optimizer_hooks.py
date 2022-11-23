import torch
import torch.nn as nn
import torch.optim as optim

class Xor(nn.Module):
    def __init__(self):
        super(Xor, self).__init__()
        self.fc1 = nn.Linear(2, 3, True)
        self.fc2 = nn.Linear(3, 1, True)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x

def test_xor():
    xor = Xor()

    data = []
    def pre_hook():
        data.append(1)

    def post_hook():
        data.pop(0)

    inputs = torch.Tensor([[0,0],[0,1], [1,0], [1,1]])
    targets = torch.Tensor([0,1,1,0]).view(-1,1)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(xor.parameters(), lr=0.01)
    optimizer.register_step_post_hook(decrement)
    optimizer.register_step_pre_hook(increment)

    xor.train()

    EPOCHS_TO_TRAIN = 5
    for _ in range(0, EPOCHS_TO_TRAIN):
        for input, target in zip(inputs, targets):
            optimizer.zero_grad()   # zero the gradient buffers
            output = xor(input)
            loss = criterion(output, target)
            loss.backward()
            # pre_hook()
            optimizer.step()    # Does the update
            # post_hook()

    assert len(data) == 0
