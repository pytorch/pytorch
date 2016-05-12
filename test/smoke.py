import torch

a = torch.FloatTensor(4, 3)
b = torch.FloatTensor(3, 4)

a.add(b)

c = a.storage()

d = a.select(0, 1)

print(c)
print(a)
print(b)
print(d)


a.fill(0)

print(a[1])

print(a.ge(long(0)))
print(a.ge(0))

