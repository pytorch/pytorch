import torch.prototypes.nestedtensor as nestedtensor
import torch
import time

def gen_rand_tensor(size):
    return torch.rand(size).cuda()

def gen_float_tensors(num_tensors, size):
    print("size: " + str(size))
    print("num_tensors: " + str(num_tensors))
    list_tensors = []
    for _ in range(num_tensors):
        list_tensors.append(gen_rand_tensor(size))
    
    nt = torch.nestedtensor(list_tensors)
    tt = gen_rand_tensor((num_tensors,) + size)
    print(tt.size())
    return list_tensors, nt, tt

def bench(num_tensors, trials=10, size=(32, 3, 128, 128)):

    list_tensors, nt, tt = gen_float_tensors(num_tensors, size)
    
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
        
    
bench(100,    trials=20, size=(16, 128))
bench(1000,   trials=20, size=(16, 128))
bench(10000,  trials=20, size=(16, 128))
bench(100000, trials=5,  size=(16, 128))
