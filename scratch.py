import torch.jit.te

@torch.jit.te.pointwise_operator
def mul(a, b):
    return a*b


def test(fn):
    torch.random.manual_seed(99)
    x = torch.randn(4, requires_grad=True)
    y = torch.randn(4, requires_grad=True)

    fn(x, fn(x, y)).sum().backward()

    print(x.grad, y.grad)

test(torch.mul)
test(torch.mul)
test(mul)
