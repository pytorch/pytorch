"""
==========================
Model ensembling
==========================
This example illustrates how to vectorize model ensembling using vmap.

What is model ensembling?
--------------------------------------------------------------------
Model ensembling combines the predictions from multiple models together.
Traditionally this is done by running each model on some inputs separately
and then combining the predictions. However, if you're running models with
the same architecture, then it may be possible to combine them together
using ``vmap``. ``vmap`` is a function transform that maps functions across
dimensions of the input tensors. One of its use cases is eliminating
for-loops and speeding them up through vectorization.

Let's demonstrate how to do this using an ensemble of simple CNNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(0)


# Here's a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        output = x
        return output


# Let's generate some dummy data. Pretend that we're working with an MNIST dataset
# where the images are 28 by 28.
# Furthermore, let's say we wish to combine the predictions from 10 different
# models.
device = "cuda"
num_models = 10
data = torch.randn(100, 64, 1, 28, 28, device=device)
targets = torch.randint(10, (6400,), device=device)
models = [SimpleCNN().to(device) for _ in range(num_models)]

# We have a couple of options for generating predictions. Maybe we want
# to give each model a different randomized minibatch of data, or maybe we
# want to run the same minibatch of data through each model (e.g. if we were
# testing the effect of different model initializations).

# Option 1: different minibatch for each model
minibatches = data[:num_models]
predictions1 = [model(minibatch) for model, minibatch in zip(models, minibatches)]

# Option 2: Same minibatch
minibatch = data[0]
predictions2 = [model(minibatch) for model in models]


######################################################################
# Using vmap to vectorize the ensemble
# --------------------------------------------------------------------
# Let's use ``vmap`` to speed up the for-loop. We must first prepare the models
# for use with ``vmap``.
#
# First, let's combine the states of the model together by stacking each parameter.
# For example, model[i].fc1.weight has shape [9216, 128]; we are going to stack the
# .fc1.weight of each of the 10 models to produce a big weight of shape [10, 9216, 128].
#
# functorch offers the following convenience function to do that. It returns a
# stateless version of the model (fmodel) and stacked parameters and buffers.
from functorch import combine_state_for_ensemble


fmodel, params, buffers = combine_state_for_ensemble(models)
[p.requires_grad_() for p in params]

# Option 1: get predictions using a different minibatch for each model.
# By default, vmap maps a function across the first dimension of all inputs to the
# passed-in function. After `combine_state_for_ensemble`, each of of ``params``,
# ``buffers`` have an additional dimension of size ``num_models`` at the front;
# and ``minibatches`` has a dimension of size ``num_models``.
print([p.size(0) for p in params])
assert minibatches.shape == (num_models, 64, 1, 28, 28)
from functorch import vmap


predictions1_vmap = vmap(fmodel)(params, buffers, minibatches)
assert torch.allclose(
    predictions1_vmap, torch.stack(predictions1), atol=1e-6, rtol=1e-6
)

# Option 2: get predictions using the same minibatch of data
# vmap has an in_dims arg that specify which dimensions to map over.
# Using ``None``, we tell vmap we want the same minibatch to apply for all of
# the 10 models.
predictions2_vmap = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, minibatch)
assert torch.allclose(
    predictions2_vmap, torch.stack(predictions2), atol=1e-6, rtol=1e-6
)

# A quick note: there are limitations around what types of functions can be
# transformed by vmap. The best functions to transform are ones that are
# pure functions: a function where the outputs are only determined by the inputs
# that have no side effects (e.g. mutation). vmap is unable to handle mutation of
# arbitrary Python data structures, but it is able to handle many in-place
# PyTorch operations.
