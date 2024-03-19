# Lazy Tensor Tutorial

## Introduction

Lazy Tensor is a brand-new tracing system in PyTorch. It includes a safety guarantee not provided by other tracing systems (jit.trace) in that it retraces and recompiles if properties about the input change or uses a cached computation otherwise. It's easier to use than jit.trace and **much** easier to use than jit.script! Lazy Tensor traces both forward and backward passes and removes many Python features present in jit scripted and traced graphs
that are difficult for hardware vendors to support.

Let's kick off our introduction to Lazy Tensor with an example that illustrates the safety guarantee, as it's one of the biggest usability issues of jit.trace. Suppose we'd like to jit trace the following function.

```python
import torch

def add_two_maybe(t: torch.Tensor, maybe: torch.Tensor):
    if maybe:
        return t + 2
    return t
```

You may have noticed that `add_two_maybe` contains an if statement that depends on `maybe` input.
Let's jit trace the function with the following inputs.

```python
t = torch.ones(1)
maybe_false = torch.BoolTensor([0])
good_inputs = (t, maybe_false)
jit = torch.jit.trace(add_two_maybe, good_inputs)
# let's check that the results match with eager
assert jit(*good_inputs) == add_two_maybe(*good_inputs)
```

So far, so good! We successfully traced `add_two_maybe` into `jit` and running it gives us the same result as the original function.

Our troubles start if we change the second input and re-run the traced function.

```python
maybe_true = torch.BoolTensor([1])
assert jit(t, maybe_true) == add_two_maybe(t, maybe_true)
```

```shell
Traceback (most recent call last):
  File "/home/villedepommes/github/pytorch4/test/test_tutorial.py", line 27, in <module>
    assert jit(t, maybe_true) == add_two_maybe(t, maybe_true)
AssertionError
```

Uh oh?! What really happened here? Let's print out the graph for `jit`:


```python

print(torch.jit.last_executed_optimized_graph())

# graph(%t : Tensor,
#       %maybe : Tensor):
#   %2 : Tensor = prim::profile[profiled_type=Float(1, strides=[1], requires_grad=0, device=cpu), seen_none=0](%t)
#    = prim::profile()
#   return (%2)
```

We could see that the if statement disappeared and jit trace only traced the `else` path. In fact, jit trace can trace **only** aten operations. It's completely oblivious to any control flow operations such as `if`, `for` or an exception.
If this sounds unsafe to you, that's because it is!

Let's now learn how we can solve this issue with Lazy Tensors.

The first step is to move the inputs to the Lazy device. The Lazy device isn't any real hardware device. Your code still runs either on CPU or on GPU if you set `LTC_TS_CUDA="1"`.

The lazy device is however very special: it makes PyTorch "remember" every aten operation (into a graph) the user calls rather than eagerly executing it. It's lazy that way ;) get it?

So, the lazy device is an API that users should use to trace their models with Lazy Tensor. It's also a PyTorch device which is a very convenient way for implementing tracing based on PyTorch dispatcher.

First of all, we need a little bit of setup. The Lazy Tensor needs a backend to actually run traced graphs. We implemented a TorchScript-based backend to give our users end-to-end experience running their models with Lazy Tensor. It also serves as an example for hardware vendors looking to integrate with Lazy Tensor.


```python
import torch._lazy
import torch._lazy.ts_backend
torch._lazy.ts_backend.init()
```

Now, we can run our example,

```python
dev = "lazy"
t_lazy = torch.ones(1).to(dev)
maybe_false_lazy = torch.BoolTensor([0]).to(dev)
lazy_result = add_two_maybe(t_lazy, maybe_false_lazy)
```

This is pretty cool! Eventually, however, we would still like to execute our computation and access the result, wouldn't we?

There are a few ways to do it. Typically, PyTorch transparently triggers the execution when the user tries to access the result e.g., print a tensor out, move the tensor to a non-lazy device, etc.

Let's give it a try:

```python
lazy_result = add_two_maybe(t_lazy, maybe_false_lazy)
print(lazy_result)
assert lazy_result.cpu() == add_two_maybe(t, maybe_false)
```

This works as expected! Let's try the case jit trace couldn't handle.

```python
maybe_true_lazy = torch.BoolTensor([1]).to(dev)
lazy_result = add_two_maybe(t_lazy, maybe_true_lazy)
assert lazy_result.cpu() == add_two_maybe(t, maybe_true)
```

Woo-hoo! This works too!
Unfortunately, this flexibility comes with a few downsides. Remember that backends need to translate aten ops into some much lower-level operations that an accelerator understands. The translation process may be time-consuming. Although, usually, it's well worth it!

However, if a non-trivial model is wildly dynamic and contains loops that always run different number of times or if statements one after another that explode into different traces every time you run the model, the backend will spend non-trivial amount of time compiling each trace even though the latter is used only for a few times.

Alright, at this point, you should have learned the main ideas behind Lazy Tensor, most common usage patterns and APIs.
Also, you are hopefully as inspired and motivated about Lazy Tensor as I am.

Let's see now how we can run a full training loop with an optimizer and backward pass! We will learn a few more important concepts and APIs.

## MNIST MLP

We will adapt the following example running MNIST_MLP from [pytorch/examples](https://github.com/pytorch/examples/blob/main/mnist/main.py)

Note, you can access the full version of the script [here](https://github.com/pytorch/pytorch/blob/master/torch/csrc/lazy/test_mnist.py)

First, we need to install one single dependency, `torchvision`

```
pip install torchvision
```

`torchvision` comes with MNIST dataset w/ images of handwritten digits, which we will be using for training.

Here's our model definition:

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
```

We are using a multi-level perceptron model with two convolutions, two linear layers and activations sandwiched in between.

Let's set up a loader that would feed the `MNIST` dataset in `train` to our model.
We are going to run the training loop for 14 epochs which is what the original MNIST example uses.
**Note, we had to move the model to the Lazy device, `Net().to(device)`. This is very similar to what we would have done had we been training this model on a GPU.**

The rest of the code is pretty standard boilerplate.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch._lazy
import torch._lazy.ts_backend
import torch._lazy.metrics
torch._lazy.ts_backend.init()

if __name__  == '__main__':
    bsz = 64
    device = 'lazy'
    epochs = 14
    log_interval = 10
    lr = 1
    gamma = 0.7
    train_kwargs = {'batch_size': bsz}
    # if we want to use CUDA
    if "LTC_TS_CUDA" in os.environ:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True,
                       'batch_size': bsz}
        train_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                        transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(log_interval, model, device, train_loader, optimizer, epoch)
        scheduler.step()
```

The training loop in `train` also has one addition. Namely, `torch._lazy.mark_step()` which deserves some elaboration on our part. `mark_step()` instructs Lazy Tensor to break up the current trace and start executing it asynchronously. The current trace encompasses both forward and backward passes and provides the backends with the whole model graph w/o any pythonisms.
If we don't stop the trace after `optimizer_step` it will include two or more iterations which is way more stuff for the backends to chew through without a whole lot of benefit.

Another important point is that after `mark_step()` we actually continue tracing the next iteration! And... start executing the previous one at the same time! Really, nothing stops us from tracing the next iteration ...and then the one after next until we hit `if batch_idx % log_interval == 0:` where
we actually need to wait for execution to catch up, so we can print out `loss`. Remember to avoid accessing intermediate results too often if you would like to extract the maximum benefit out of Lazy Tensor.

Since every iteration looks exactly like the one before it, the TS backend will be re-using the same TS compilation.

Alright, let's run it now!

```python
def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        torch._lazy.mark_step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
```


After the script downloads the dataset, the model will be trained on the Lazy device as
evidenced by the decreasing loss.

```shell
Train Epoch: 1 [0/60000 (0%)]   Loss: 2.343924
Train Epoch: 1 [640/60000 (1%)] Loss: 1.760821
Train Epoch: 1 [1280/60000 (2%)]        Loss: 0.802798
Train Epoch: 1 [1920/60000 (3%)]        Loss: 0.856164
Train Epoch: 1 [2560/60000 (4%)]        Loss: 0.568396
Train Epoch: 1 [3200/60000 (5%)]        Loss: 0.399044
Train Epoch: 1 [3840/60000 (6%)]        Loss: 0.457996
Train Epoch: 1 [4480/60000 (7%)]        Loss: 0.285104
Train Epoch: 1 [5120/60000 (9%)]        Loss: 0.193083
Train Epoch: 1 [5760/60000 (10%)]       Loss: 0.486165
Train Epoch: 1 [6400/60000 (11%)]       Loss: 0.163996
Train Epoch: 1 [7040/60000 (12%)]       Loss: 0.200323

```

Let's briefly mention a few more APIs before we wrap this up. Unfortunately, LT is still very early in its development which means it doesn't implement every single PyTorch op out of there.
In fact, we implement about a hundred most common ops. What happens if a model contains an op that LT does **not** implement. Lazy Tensor transparently (from a user) breaks up the current trace, waits until all inputs to the op are computed, computes the op on some different device, and finally moves the results onto the lazy device again and starts a new trace.
This big-little wrinkle means that *sometimes* LT can **not** give the backend a whole model graph which may have a negative impact on performance. You could get the list of the ops that LT could handle for your model by adding the following to your model:

```python
torch._lazy.metrics.reset()
train(...)
print(torch._lazy.metrics.counter_names())
```

If you are seeing any ops with the prefix: `aten::`

*Sometimes* you could replace such ops with similar that LT does support. More often than not, we will have to just live with it until LT matures.

Another handy API is `torch._lazy.wait_device_ops()`. Remember, we said that `mark_step()` breaks up the current trace and kicks off a computation asynchronously? If downstream there are no blocking operations such as `print`, `item()`, `to`, LT will happily continue tracing.
If you would like to time how much exactly time computation and tracing took for some model without including device transfers or printing, you could stick `torch._lazy.wait_device_ops()` and `time.perf_counter()` right after it. Don't forget another `time.perf_counter()` before the trace start!

This concludes our brief introduction to LT. Hopefully, you'll remember the main takeaways:

* Backends prefer bigger graphs that preferably include both forward and backward as there's ample opportunity for performance optimizations
* It's really tricky to produce such graphs without overburdening a user too much. Think, torch.jit.script, torch.jit.trace! Also, think ifs, fors, "Lions, and Tigers, and Bears, Oh My" We digressed.


Please give LT a try and tell us what you think on GitHub! We are **eager, not lazy** (haha!) to hear from you!
