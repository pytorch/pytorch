import torch
import torch.nn as nn
import torch.legacy.nn as legacy


class BilinearTests:

    m = nn.Bilinear(10, 10, 20)
    m2 = legacy.Bilinear(10, 10, 20)
    m2.weight.copy_(m.weight.data)
    m2.bias.copy_(m.bias.data)

    def test_forward(self):
        x1 = torch.randn(5, 10)
        x2 = torch.randn(5, 10)
        out = self.m(torch.autograd.Variable(x1), torch.autograd.Variable(x2))
        x = [x1, x2]
        out2 = self.m2.forward(x)
        return (out.data - out2)

    def test_backward(self):
        import operator
        inp = torch.randn(5, 10)
        inp1 = torch.randn(5, 10)
        inpu = [inp, inp1]
        inp_v = torch.autograd.Variable(inp, requires_grad=True)
        inp1_v = torch.autograd.Variable(inp1, requires_grad=True)
        out = self.m(inp_v, inp1_v)
        out2 = self.m2.forward(inpu)

        grad = torch.randn(*out.size())

        out.backward(grad)
        gi = inp_v.grad.data.clone()
        gi1 = inp1_v.grad.data.clone()
        gii = [gi, gi1]
        gi2 = self.m2.backward(inpu, grad)

        return list(map(operator.sub, gii, gi2)),
        (self.m.weight.data - self.m2.weight), (self.m.bias.data - self.m2.bias)

a = BilinearTests()
print a.test_forward()
print a.test_backward()
