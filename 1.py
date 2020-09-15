import torch 

a = [torch.ones(2, 2) for _ in range(3)]
a2 = [torch.ones(2, 2) for _ in range(3)]

scalars = [1.1, 2, 3]
resA = torch._foreach_add(a, a2) # used to work until this change
resA = torch._foreach_add(a, scalars) # this works fine 
