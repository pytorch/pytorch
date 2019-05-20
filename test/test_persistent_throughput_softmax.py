import torch
from torch.autograd import Variable

a = torch.rand([16384,512])

a1 = Variable(a, requires_grad=True)
a2 = Variable(a.half().cuda(), requires_grad=True)

m = torch.nn.Softmax(dim=-1)

b1 = m(a1)
b2 = m(a2)

print(b1)
print(b2)

b1.backward(a)
b2.backward(a.half().cuda())

print(a1.grad)
print(a2.grad)

c1 = Variable(a, requires_grad=True)
c2 = Variable(a.half().cuda(), requires_grad=True)

ml = torch.nn.LogSoftmax(dim=-1)

d1 = ml(c1)
d2 = ml(c2)

print(d1)
print(d2)

d1.backward(a)
d2.backward(a.half().cuda())

print(c1.grad)
print(c2.grad)


