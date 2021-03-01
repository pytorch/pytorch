import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx.experimental.shape_prop
import torch.fx as fx
from torch.fx import Node, Proxy, symbolic_trace, Graph, GraphModule
from typing import List, Dict, Tuple
from torch import Tensor
import copy

# Utilities to make nn.Module "functional"
# In particular the goal is to be able to provide a function that takes as input
# the parameters and evaluate the nn.Module using fixed inputs.
def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
    """
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])

def _set_nested_attr(obj: nn.Module, names: List[str], value: Tensor) -> None:
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)

def extract_weights(mod: nn.Module) -> Tuple[Tuple[Tensor, ...], List[str]]:
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        _del_nested_attr(mod, name.split("."))
        names.append(name)

    # Make params regular Tensors instead of nn.Parameter
    params = tuple(p.detach().requires_grad_() for p in orig_params)
    return params, names

def load_weights(mod: nn.Module, names: List[str], params: Tuple[Tensor, ...]) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    for name, p in zip(names, params):
        _set_nested_attr(mod, name.split("."), p)


# =========================== End to end test ==============================

# NB: Assumes the model takes ONE input (data) only
# This is because FX doesn't allow the * operator (e.g. *args) on a Proxy
def make_functional(model: nn.Module):
    weights, descriptors = extract_weights(model)

    def fun(weights, data):
        mutable_model = copy.deepcopy(model)
        load_weights(mutable_model, descriptors, weights)
        return mutable_model(data)

    return weights, fun


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

# Here's how we do program acquisition on an nn.Module in this file.
# We might not actually need step 1 but that might require some more modifications to FX.
# 1. First, we need to make the module functional via the "nn module functional api".
# 2. Next, we symbolic trace through the function version of the module. This was
#    made possible by modifying a few lines of FX code to ignore modules.

# Step 1: Extract the state (weights) from the network.
weights, func = make_functional(net)

def compute_loss(w0, w1, w2, w3, w4, data, target):
    output = func([w0, w1, w2, w3, w4], data)
    return criterion(output, target)

# Step 2: symbolically trace through the function.
graph_func = symbolic_trace(compute_loss)
print(graph_func.code)

# import torch
# def forward(self, w0, w1, w2, w3, w4, data, target):
#     embedding_1 = torch.embedding(w0, data, -1, False, False);  w0 = data = None
#     transpose_1 = torch.transpose(embedding_1, -1, -2);  embedding_1 = None
#     mean_1 = torch.mean(transpose_1, -1);  transpose_1 = None
#     linear_1 = torch.nn.functional.linear(mean_1, w1, bias = w2);  mean_1 = w1 = w2 = None
#     relu_1 = torch.nn.functional.relu(linear_1, inplace = False);  linear_1 = None
#     linear_2 = torch.nn.functional.linear(relu_1, w3, bias = w4);  relu_1 = w3 = w4 = None
#     log_softmax_1 = torch.nn.functional.log_softmax(linear_2, dim = -1, _stacklevel = 3, dtype = None);  linear_2 = None
#     nll_loss_1 = torch.nn.functional.nll_loss(log_softmax_1, target, weight = None, size_average = None, ignore_index = -100, reduce = None,
#  reduction = 'mean');  log_softmax_1 = target = None
#     return nll_loss_1

