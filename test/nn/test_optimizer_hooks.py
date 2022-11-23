from typing import Any, Tuple, List, Dict
from collections import defaultdict

from sklearn.datasets import make_blobs
import torch
import torch.nn as nn
from torch.optim import SGD


# Test: feed forward network
class FeedForward(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        torch.manual_seed(7)
        super(FeedForward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


def load_data() -> Tuple[Any, Any]:
    torch.manual_seed(11)
    x_train, y_train = make_blobs(n_samples=40, n_features=2, cluster_std=1.5, shuffle=True, random_state=10)
    x_train = torch.Tensor.float(x_train)
    y_train = torch.Tensor.float(blob_label(y_train, 0, [0]))
    y_train = torch.Tensor.float(blob_label(y_train, 1, [1, 2, 3]))

    data = (x_train, y_train)
    return data


def train_linear_model(model: FeedForward) -> Dict:
    x, y = 2, 16
    
    def increment():
        global x
        x += 4

    def decrement():
        global y
        y -= 3
    
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001)
    pre_hook_handle = optimizer.register_step_pre_hook(increment)
    post_hook_handle = optimizer.register_step_post_hook(decrement)

    x_train, y_train = load_data()
    num_epochs = 2
    model.train()
    
    assert x == 2
    assert y == 16 
    
    state = defaultdict(List)
    state[-1] = [2, 16]
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
   
        loss = criterion(y_pred.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        state[epoch] = [x,y]

    return state

def test_linear_model() -> None:
    model1 = FeedForward(2, 10)
    vanilla_model = train_linear_model(model1)

    assert vanilla_model[1][0] == vanilla_model[1][1]
