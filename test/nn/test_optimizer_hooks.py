# Owner(s): ["module: nn"]
import pytest
import torch
import torch.nn as nn
from torch.optim import SGD

class XOR(nn.Module):
    def __init__(self):
        super(XOR, self).__init__()
        self.fc1 = nn.Linear(2, 3, True)
        self.fc2 = nn.Linear(3, 1, True)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x


data = []

def reset_data():
    while len(data) > 0:
        data.pop()

def pre_hook_list():
    data.append(1)

def post_hook_list():
    data.append(0)

@pytest.mark.skip
def test_pre_and_post_hook():
    reset_data()
    inputs = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = torch.Tensor([0, 1, 1, 0]).view(-1, 1)
    print(len(data))
    xor1 = XOR()

    optimizer1 = SGD(xor1.parameters(), lr=0.01)
    criterion1 = nn.MSELoss()

    post_hook_handles, pre_hook_handles = [], []
    post_hook_handle = optimizer1.register_step_post_hook(post_hook_list)
    post_hook_handles.append(post_hook_handle)

    pre_hook_handle = optimizer1.register_step_pre_hook(pre_hook_list)
    pre_hook_handles.append(pre_hook_handle)

    print(id(xor1), id(optimizer1))
    xor1.train()
    for _ in range(5):
        for input, target in zip(inputs, targets):
            optimizer1.zero_grad()   # zero the gradient buffers
            output = xor1(input)
            loss = criterion1(output, target)
            loss.backward()
            optimizer1.step()
    assert len(data) == 40

    while len(post_hook_handles) > 0:
        elt = post_hook_handles.pop()
        elt.remove()

    while len(pre_hook_handles) > 0:
        elt = pre_hook_handles.pop()
        elt.remove()

@pytest.mark.skip
def test_pre_hook():
    reset_data()
    inputs = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = torch.Tensor([0, 1, 1, 0]).view(-1, 1)

    print(len(data))
    xor2 = XOR()
    criterion2 = nn.MSELoss()
    optimizer2 = SGD(xor2.parameters(), lr=0.01)
    handle = optimizer2.register_step_pre_hook(pre_hook_list)
    print(id(xor2), id(optimizer2))

    xor2.train()
    for _ in range(2):
        for input, target in zip(inputs, targets):
            optimizer2.zero_grad()   # zero the gradient buffers
            output = xor2(input)
            loss = criterion2(output, target)
            loss.backward()
            optimizer2.step()

    assert len(data) == 8

head: int = 4
tail: int = 20

def post_hook_int(*args):
    global tail
    tail -= 4

def pre_hook_int(*args):
    global head
    head += 4

def test_update_int():
    xor3 = XOR()
    optimizer3 = SGD(xor3.parameters(), lr=0.01)

    optimizer3.step()
    assert head == 4
    assert tail == 20

    optimizer3.register_step_post_hook(post_hook_int)
    optimizer3.register_step_pre_hook(pre_hook_int)
    optimizer3.step()

    assert head == 8
    assert tail == 16


def test_update_int2():
    xor4 = XOR()
    optimizer4 = SGD(xor4.parameters(), lr=0.01)

    optimizer4.step()
    assert head == 12
    assert tail == 12

    optimizer4.register_step_post_hook(post_hook_int)
    optimizer4.step()

    assert head == 16
    assert tail == 8

#
#    def test_post_hook():
#        XOR, inputs, targets, criterion, optimizer = setup_XOR()
#        _ = optimizer.register_step_post_hook(post_hook)
#
#        XOR.train()
#        for _ in range(5):
#            for input, target in zip(inputs, targets):
#                optimizer.zero_grad()   # zero the gradient buffers
#                output = XOR(input)
#                loss = criterion(output, target)
#                loss.backward()
#                optimizer.step()
#
#        assert len(data) == 20
