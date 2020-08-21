import torch
import time
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, StepLR

import time

H = 1
W = 1
R = 1
T = 2

weight = torch.ones(H, W, device='cuda', requires_grad=True)
bias = torch.ones(H, device='cuda', requires_grad=True)
bias1 = torch.ones(H, device='cuda', requires_grad=True)
input = torch.ones(W, device='cuda')

params0 = [weight, bias]
optimizer = optim.Adam(params0, lr=1e-3)
schedulers = [
    ExponentialLR(optimizer, gamma=0.9), 
    ReduceLROnPlateau(optimizer),
    StepLR(optimizer, gamma=0.9, step_size=10)
]

def fn():
    optimizer.zero_grad()
    y = weight.mv(input)
    if y.is_cuda and bias.is_cuda and y.get_device() != bias.get_device():
        y = y.cuda(bias.get_device())
    loss = (y + bias).pow(2).sum()
    loss.backward()
    return loss

start = time.time()
torch.cuda.synchronize()
    
initial_value = fn().item()
for _i in range(R):
    for scheduler in schedulers:
        if isinstance(scheduler, ReduceLROnPlateau):
            val_loss = fn()
            scheduler.step(val_loss)
        else:
            scheduler.step()
    optimizer.step(fn)
    
torch.cuda.synchronize()
#    t += time.monotonic() - t0
 #   count += 1

end = time.time()

print("\n\n\n\n")
print("--------------------------------------------")
print("time: ", end - start)
print("--------------------------------------------")
print(initial_value, " ---> ", fn().item())
print("Test1: ", initial_value == 4.0) 
print("Test2: ", fn().item() == 3.9856133460998535)

#1-1-1 4.0  **** --->  3.9928035736083984
#1000-1000-100 **** 1002001024.0  --->  985084800.0

#
# 1000200011776.0  --->  983313350656.0
# 1000200011776.0  --->  988581134336.0
# 1000200011776.0  --->  988581134336.0