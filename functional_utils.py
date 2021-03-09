import torch
from torch import vmap
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from make_functional import make_functional

# x = torch.ones(2, 3)
# y = torch.ones(2, 3)
# # result = vmap(torch.add)(x, y)
# result = vmap(vmap(torch.add))(x, y)

# assert torch.allclose(result, x + y)

def _create_differentiable(tensor_or_tuple_of_tensors):
    if isinstance(tensor_or_tuple_of_tensors, torch.Tensor):
        tensor = tensor_or_tuple_of_tensors
        # if tensor.requires_grad:
        #     return tensor
        assert not tensor.requires_grad
        return tensor.requires_grad_()
    if isinstance(tensor_or_tuple_of_tensors, tuple):
        return tuple(map(_create_differentiable, tensor_or_tuple_of_tensors))
    if isinstance(tensor_or_tuple_of_tensors, list):
        return tuple(map(_create_differentiable, tensor_or_tuple_of_tensors))
    assert False

def _undo_create_differentiable(tensor_or_tuple_of_tensors):
    if isinstance(tensor_or_tuple_of_tensors, torch.Tensor):
        tensor = tensor_or_tuple_of_tensors
        return tensor.requires_grad_(False)
    if isinstance(tensor_or_tuple_of_tensors, tuple):
        return tuple(map(_undo_create_differentiable, tensor_or_tuple_of_tensors))
    if isinstance(tensor_or_tuple_of_tensors, list):
        return tuple(map(_undo_create_differentiable, tensor_or_tuple_of_tensors))
    assert False

def _any_differentiable(tensor_or_tuple_of_tensors):
    if isinstance(tensor_or_tuple_of_tensors, torch.Tensor):
        tensor = tensor_or_tuple_of_tensors
        return tensor.requires_grad
    if isinstance(tensor_or_tuple_of_tensors, tuple):
        return any(tuple(map(_any_differentiable, tensor_or_tuple_of_tensors)))
    if isinstance(tensor_or_tuple_of_tensors, list):
        return any(tuple(map(_any_differentiable, tensor_or_tuple_of_tensors)))
    return False


def grad_with_value(f, diff_argnums=(0,), has_aux=False):
    def wrapper(*args):
        torch._C._grad_increment_nesting()
        output, aux = None, None
        try:
            args = [_create_differentiable(arg) if i in diff_argnums else arg
                    for i, arg in enumerate(args)]
            output = f(*args)
            if has_aux:
                output, aux = output
            assert output.dim() == 0
            diff_args = [args[i] for i in diff_argnums]
            single_diff_arg = isinstance(diff_args[0], torch.Tensor) and len(diff_args) == 1
            # TODO: quick hack...
            if len(diff_args) == 1 and isinstance(diff_args[0], tuple):
                diff_args = diff_args[0]
            # NB: need create_graph so that backward pass isn't run in no_grad mode
            grad_input = torch.autograd.grad(
                output, diff_args,
                create_graph=True, retain_graph=True)
            if single_diff_arg:
                grad_input = grad_input[0]
        finally:
            _undo_create_differentiable(args)
            torch._C._grad_decrement_nesting()
        if has_aux:
            return grad_input, output, aux
        return grad_input, output
    return wrapper

def grad(f, diff_argnums=(0,), has_aux=False):
    def wrapper(*args):
        results = grad_with_value(f, diff_argnums, has_aux=has_aux)(*args)
        if has_aux:
            return results[0], results[2]
        return results[0]
    return wrapper


if __name__ == '__main__':
    def f(x):
        y = x.sin()
        return y.sum()

    x = torch.tensor([0., 1., 2.])
    assert torch.allclose(grad(f)(x), x.cos())

    g = lambda x: x.sin()
    y = torch.tensor(0.3)
    neg_sin_y = grad(grad(g))(y)
    assert torch.allclose(neg_sin_y, -y.sin())

    x = torch.tensor([0., 1., 2.])
    result = vmap(grad(f))(x)
    assert torch.allclose(result, x.cos())

    def g(x):
        return (x ** 2).sum()
    result = vmap(grad(g))(x)
    assert torch.allclose(result, 2 * x)

    def compute_loss(weight, x, t):
        y = x @ weight
        return ((y - t) ** 2).sum()

    weight = torch.randn(16, 2)
    x = torch.randn(64, 16)
    t = torch.randn(64, 2)
    result = vmap(partial(grad(compute_loss), weight))(x, t)
    expected = [grad(compute_loss)(weight, x[i], t[i]) for i in range(64)]
    expected = torch.stack(expected)
    assert torch.allclose(result, expected)


    class SampleNet(nn.Module):
        def __init__(self, vocab_size: int):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, 16)
            self.fc1 = nn.Linear(16, 16)
            self.fc2 = nn.Linear(16, 2)

        def forward(self, x):
            x = self.emb(x)
            x = torch.transpose(x, -1, -2)
            x = torch.mean(x, -1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x

        def name(self):
            return "SampleNet"

    # Create our inputs...
    vocab_size = 1000
    batch_shape = [64]
    words_per_sentence = 5
    data = torch.randint(0, vocab_size, (*batch_shape, words_per_sentence))
    targets = torch.randint(0, 1, (*batch_shape,))

    # Construct our module
    net = SampleNet(vocab_size)
    criterion = nn.CrossEntropyLoss()

    params = dict(net.named_parameters())
    weights, net_func, _ = make_functional(net)

    def compute_loss(weights, data, target):
        output = net_func(weights, (data,))
        result = criterion(output, target)
        # import torchviz; import graphviz
        # graph = torchviz.make_dot(result)
        # graph.save("graph_single.dot")
        return result

    expected = [grad(compute_loss)(weights, data[i], targets[i]) for i in range(64)]
    expected = zip(*expected)
    expected = tuple(torch.stack(shards) for shards in expected)

    result = vmap(partial(grad(compute_loss), weights))(data, targets)
    for r, e in zip(result, expected):
        assert torch.allclose(r, e)

    # Can we use regular autograd with the grad transform?
    x = torch.randn([], requires_grad=True)
    cos_x = grad(torch.sin)(x)
    result, = torch.autograd.grad(cos_x, x)
    assert torch.allclose(result, -x.sin())

    # Test that views work
    x = torch.randn([], requires_grad=True)
    y = torch.randn([], requires_grad=True)

    def silly_sin(x):
        x = x.view([])
        x = x.sin()
        return x

    def foo(x, y):
        z1 = grad(silly_sin)(x)
        z2 = torch.cos(y)
        return z1 + z2

    result = foo(x, y)
    grads = torch.autograd.grad(result, [x, y])
    assert torch.allclose(grads[0], -x.sin())
    assert torch.allclose(grads[1], -y.sin())

    # Test in-place
    def foo(x):
        x = x.clone()
        x.sin_()
        return x

    result = grad(foo)(x)
    assert torch.allclose(result, x.cos())

    # Test simple view + in-place
    def foo(x):
        x = x.clone()
        x.view([]).sin_()
        return x

    result = grad(foo)(x)
    assert torch.allclose(result, x.cos())

    # Weird case
    x = torch.randn([], requires_grad=True)
    y = torch.randn(3, requires_grad=True)

    def silly_sin(x):
        x = x.view([])
        x = x.sin()
        return x

    def foo(x, y):
        z1 = grad(silly_sin)(x)
        z2 = torch.cos(y)
        return z1 + z2

    result = vmap(foo, (None, 0))(x, y)
    loss = result.sum()
    grads = torch.autograd.grad(loss, [x, y])
    assert torch.allclose(grads[0], 3 * -x.sin())
    assert torch.allclose(grads[1], -y.sin())

    # import torchviz; import graphviz
    # graph = torchviz.make_dot(loss)
    # graph.save("gvg.dot")
    # 
    # grads = torch.autograd.grad(result, [x, y], torch.ones_like(result))
