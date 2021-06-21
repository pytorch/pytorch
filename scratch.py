import torch.jit.te

@torch.jit.te.pointwise_operator
def mul(a, b):
    return a+b


def test(fn):
    torch.random.manual_seed(99)
    x = torch.randn(8)
    y = torch.randn(8, requires_grad=True)

    fn(x, y).sum().backward()

    print(y.grad)

test(torch.mul)
test(torch.mul)
test(mul)
