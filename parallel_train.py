import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import vmap
from functional_utils import make_functional, grad_with_value

# Adapted from http://willwhitney.com/parallel-training-jax.html
# GOAL: Demonstrate that it is possible to use eager-mode vmap
# to parallelize training over models.

# NB: this code runs off of a branch on zou3519/pytorch:dynlayer

DEVICE = 'cpu'

# Step 1: Make some spirals
def make_spirals(n_samples, noise_std=0., rotations=1.):
    ts = torch.linspace(0, 1, n_samples, device=DEVICE)
    rs = ts ** 0.5
    thetas = rs * rotations * 2 * math.pi
    signs = torch.randint(0, 2, (n_samples,), device=DEVICE) * 2 - 1
    labels = (signs > 0).to(torch.long)

    xs = rs * signs * torch.cos(thetas) + torch.randn(n_samples, device=DEVICE) * noise_std
    ys = rs * signs * torch.sin(thetas) + torch.randn(n_samples, device=DEVICE) * noise_std
    points = torch.stack([xs, ys], dim=1)
    return points, labels

points, labels = make_spirals(100, noise_std=0.05)


# Step 2: Define two-layer MLP and loss function
class MLPClassifier(nn.Module):
    def __init__(self, hidden_dim=32, n_classes=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.fc1 = nn.Linear(2, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.n_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, -1)
        return x

loss_fn = nn.NLLLoss()

# Step 3: Make the model functional(!!) and define a training function.
# NB: this mechanism doesn't exist in PyTorch today, but we want it to:
# https://github.com/pytorch/pytorch/issues/49171
weights, func_model, _ = make_functional(MLPClassifier().to(DEVICE))

def train_step_fn(weights, batch, targets, lr=0.2):
    def compute_loss(weights, batch, targets):
        output = func_model(weights, (batch,))
        loss = loss_fn(output, targets)
        return loss

    grad_weights, loss = grad_with_value(compute_loss)(weights, batch, targets)

    # NB: PyTorch is missing a "functional optimizer API" (possibly coming soon)
    # so we are going to re-implement SGD here.
    new_weights = []
    with torch.no_grad():
        for grad_weight, weight in zip(grad_weights, weights):
            new_weights.append(weight - grad_weight * lr)
    # NB: return looks weird because torch.vmap must return Tensors
    return (loss, *new_weights)


def unpack(train_result):
    return train_result[0], train_result[1:]

# Step 4: Let's verify this actually trains.
# We should see the loss decrease.
def step4():
    global weights
    for i in range(2000):
        loss, weights = unpack(train_step_fn(weights, points, labels))
        if i % 100 == 0:
            print(loss)

step4()

# Step 5: We're ready for multiple models. Let's define an init_fn
# that, given a number of models, returns to us all of the weights.
def init_fn(num_models):
    models = tuple(MLPClassifier() for _ in range(num_models))
    weights = tuple(make_functional(model)[0] for model in models)
    weights = tuple(zip(*weights))
    weights = tuple(torch.stack(shards).detach() for shards in weights)
    return weights

# Step 6: Now, can we try multiple models at the same time?
# The answer is: yes! `loss` is a 2-tuple, and we can see that the value keeps
# on decreasing
def step6():
    parallel_train_step_fn = vmap(train_step_fn, in_dims=(0, None, None))
    batched_weights = init_fn(num_models=2)
    for i in range(2000):
        loss, batched_weights = unpack(parallel_train_step_fn(batched_weights, points, labels))
        if i % 200 == 0:
            print(loss)

step6()

# Step 7: Now, the flaw with step 6 is that we were training on the same exact
# data. This can lead to all of the models in the ensemble overfitting in the 
# same way. The solution that http://willwhitney.com/parallel-training-jax.html
# applies is to randomly subset the data in a way that the models do not recieve
# exactly the same data in each training step!
# Because the goal of this doc is to show that we can use eager-mode vmap to
# achieve similar things as JAX, the rest of this is left as an exercise to the reader.

# In conclusion, to achieve what http://willwhitney.com/parallel-training-jax.html
# does, we used the following additional items that PyTorch does not have:
# 1. NN module functional API that turns a module into a (state, state_less_fn) pair
# 2. Functional optimizers
# 3. A "functional" grad API (that effectively wraps autograd.grad)
# 4. Composability between the functional grad API and torch.vmap.
