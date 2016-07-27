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

print(a.ge(0).size()[0], a.ge(0).size()[1])
s = a.ge(0)
d = s.select(0, 1)
print(d.double())
print(a.ge(0))

if not PY3:
    s = a.ge(int(0))
    print(s)

a = torch.ones(2, 2)
b = torch.DoubleTensor()
b.set(a)
assert b.isSetTo(a)
