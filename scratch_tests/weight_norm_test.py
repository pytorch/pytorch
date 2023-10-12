import torch
import torch.nn as nn
from tqdm import tqdm


print(torch.__file__)

torch.manual_seed(0)

def loss_fn(output):
    return torch.sum(output**2)

def get_weights_copy(model):
    weights_path = 'weights_temp.pt'
    torch.save(model.state_dict(), weights_path)
    return torch.load(weights_path)

class Model(nn.Module):
    def __init__(self, conv=False):
        super().__init__()
        self.conv = conv
        if conv:
            self.conv = nn.Conv2d(2, 3, 5, padding=1)
            self.norm_conv = nn.utils.weight_norm(self.conv, dim=1)

            # weight norm currently doesn't work with dim 0 and 3 on conv2d

        self.lin1 = nn.Linear(192, 1000)
        self.lin2 = nn.Linear(1000, 5000)
        self.lin3 = nn.Linear(5000, 1000)
        self.lin4 = nn.Linear(1000, 5)
        self.norm1 = nn.utils.weight_norm(self.lin1, dim=1)
        self.norm2 = nn.utils.weight_norm(self.lin2, dim=0)
        self.norm1 = nn.utils.weight_norm(self.lin3, dim=1)
        self.norm2 = nn.utils.weight_norm(self.lin4, dim=0)

    def forward(self, x):
        if self.conv:
            #print(list(self.conv.parameters())[0].shape)
            x = self.conv(x)
            x = torch.flatten(x)
            #print(x.shape)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        x = self.lin4(x)
        return x

conv = True

cpu_model = Model(conv=conv)

mps_model = Model(conv=conv).to('mps')
mps_model.load_state_dict(get_weights_copy(cpu_model))

if conv:
    test_input_cpu = torch.rand((1, 2, 10, 10))
else:
    test_input_cpu = torch.rand((1, 192))

test_input_mps = test_input_cpu.detach().clone().to('mps')

# print(test_input_mps)
print(mps_model(test_input_mps))

print()

# print(test_input_cpu)
print(cpu_model(test_input_cpu))
print()

device = 'mps'

if device == 'cpu':
    model = cpu_model
else:
    model = mps_model

optimizer = torch.optim.Adam(model.parameters())

if conv:
    inputs = torch.rand((10000, 2, 10, 10)).to(device)
else:
    inputs = torch.rand((100, 192)).to(device)

losses = []

# 115 its/s with cpu fallback (dim 1 and 2 is the same)
# 415 its/s with just the two linear layers and no weight norm
# 150 its/s with mps weight norm forward and cpu fallback backward
for input in tqdm(inputs):
    input = input
    result = model(input)
    # model.zero_grad()
    # loss = loss_fn(result)
    # loss.backward()
    # optimizer.step()
    # losses.append(loss.item())
print(loss.item())
print(result)
