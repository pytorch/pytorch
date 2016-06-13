import torch
import sys

PY3 = sys.version.startswith('3.')

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

if not PY3:
    print(a.ge(long(0)))
print(a.ge(0))

a = torch.ones(2, 2)
b = torch.DoubleTensor()
b.set(a)
assert b.isSetTo(a)
