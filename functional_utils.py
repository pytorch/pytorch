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
        if tensor.requires_grad:
            return tensor
        return tensor.detach().requires_grad_()
    if isinstance(tensor_or_tuple_of_tensors, tuple):
        return tuple(map(_create_differentiable, tensor_or_tuple_of_tensors))
    if isinstance(tensor_or_tuple_of_tensors, list):
        return tuple(map(_create_differentiable, tensor_or_tuple_of_tensors))
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


def grad_with_value(f, diff_argnums=(0,)):
    def wrapper(*args):
        should_increment_nesting = not torch._C._grad_layer_at_top()
        if should_increment_nesting: 
            torch._C._grad_increment_nesting()
        output = None
        try:
            create_graph = _any_differentiable(args)
            args = [_create_differentiable(arg) if i in diff_argnums else arg.detach()
                    for i, arg in enumerate(args)]
            output = f(*args)
            assert output.dim() == 0
            diff_args = [args[i] for i in diff_argnums]
            # TODO: quick hack...
            if len(diff_args) == 1 and isinstance(diff_args[0], tuple):
                diff_args = diff_args[0]
            grad_input = torch.autograd.grad(output, diff_args, create_graph=create_graph)
        finally:
            if should_increment_nesting:
                torch._C._grad_decrement_nesting()
        return grad_input, output
    return wrapper

def grad(f, diff_argnums=(0,)):
    def wrapper(*args):
        result, _ = grad_with_value(f, diff_argnums)(*args)
        return result
    return wrapper

def f(x):
    y = x.sin()
    return y.sum()

x = torch.tensor([0., 1., 2.])
assert torch.allclose(grad(f)(x)[0], x.cos())

x = torch.tensor([0., 1., 2.])
result, = vmap(grad(f))(x)
assert torch.allclose(result, x.cos())

def g(x):
    return (x ** 2).sum()
result, = vmap(grad(g))(x)
assert torch.allclose(result, 2 * x)

def compute_loss(weight, x, t):
    y = x @ weight
    return ((y - t) ** 2).sum()

weight = torch.randn(16, 2)
x = torch.randn(64, 16)
t = torch.randn(64, 2)
result, = vmap(partial(grad(compute_loss), weight))(x, t)
expected = torch.stack([grad(compute_loss)(weight, x[i], t[i])[0] for i in range(64)])
assert torch.allclose(result, expected)

def compute_loss(weight, x):
    y = torch.matmul(x, weight)
    return y.sum()

weight = torch.randn(16, 16)
x = torch.randn(64, 16)
vmap(partial(grad(compute_loss), weight))(x)


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
weights, net_func = make_functional(net)

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

# NB: Much nicer when create_graph is True
x = torch.randn([]).requires_grad_()
result = grad(lambda x: grad(torch.sin)(x)[0])(x)[0]
assert torch.allclose(result, -torch.sin(x))
