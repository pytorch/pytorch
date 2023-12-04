import torch
import torch.random
import time

high=5
low=1
size=[3,2,2]
torch.manual_seed(0)

t0=time.time()
for j in range(100):
    for i in range(1000000):
        x=torch.FloatTensor(size[0],size[1],size[2]).uniform_(low,high)
t1=time.time()
print("It took", (t1-t0)/100, "time for original method ")

t2=time.time()
for j in range(100):

    for i in range(1000000):
        y=torch.random.uniform(low,high,size[0],size[1],size[2])
    t3=time.time()
print("It took", (t3-t2)/100, "time for updated method")
