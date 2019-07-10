import torch.prototypes.nestedtensor as nestedtensor
import torch
import time

num_tensors = 100

def bench(num_tensors, trials=10, size=(16, 128, 128)):
    print("num_tensors: " + str(num_tensors))
    print("size: " + str(size))
    list_tensors = []
    
    for _ in range(num_tensors):
        list_tensors.append(torch.rand(*size))
    
    nt = torch.nestedtensor(list_tensors)

    tt = torch.rand(*((num_tensors,) + size))
    print(tt.size())
    
    t1 = time.time()
    for i in range(trials):
        for tensor in list_tensors:
            tensor.cos_()
    t2 = time.time() - t1
    t2 = t2 / trials
    
    print("1: t2: " + str(t2))

    t1 = time.time()
    for i in range(trials):
        nt.cos_()
    t2 = time.time() - t1
    t2 = t2 / trials
    
    print("2: t2: " + str(t2))

    t1 = time.time()
    for i in range(trials):
        tt.cos_()
    t2 = time.time() - t1
    t2 = t2 / trials
    
    print("3: t2: " + str(t2))
        
    
bench(100)
bench(200)
