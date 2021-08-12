# PyTorch on Accelerators

PyTorch runs on accelerators, like TPUs, with the
[lazy_tensor_core package](https://github.com/pytorch/ltc/). This document describes
how to run your models on these devices.

## Creating a Lazy Tensor

PyTorch/LTC adds a new `ltc` device type to PyTorch. This device type works just
like other PyTorch device types. For example, here's how to create and
print a lazy tensor:

```python
import torch
import lazy_tensor_core
import lazy_tensor_core.core.lazy_model as ltm

t = torch.randn(2, 2, device=ltm.lazy_device())
print(t.device)
print(t)
```

This code should look familiar. PyTorch/LTC uses the same interface as regular
PyTorch with a few additions. Importing `lazy_tensor_core` initializes PyTorch/LTC
and `ltm.lazy_device()` returns the current lazy device. This may be a CPU or an
accelerator, depending on your environment.

## Lazy Tensors Are PyTorch Tensors

PyTorch operations can be performed on lazy tensors just like CPU or CUDA tensors.

For example, lazy tensors can be added together:

```python
t0 = torch.randn(2, 2, device=ltm.lazy_device())
t1 = torch.randn(2, 2, device=ltm.lazy_device())
print(t0 + t1)
```

Or matrix multiplied:

```python
print(t0.mm(t1))
```

Or used with neural network modules:

```python
l_in = torch.randn(10, device=ltm.lazy_device())
linear = torch.nn.Linear(10, 20).to(ltm.lazy_device())
l_out = linear(l_in)
print(l_out)
```

Like other device types, lazy tensors only work with other lazy tensors on the
same device. So code like

```python
l_in = torch.randn(10, device=ltm.lazy_device())
linear = torch.nn.Linear(10, 20)
l_out = linear(l_in)
print(l_out)
# Input tensor is not a lazy tensor: torch.FloatTensor
```

will throw an error since the `torch.nn.Linear` module is on the CPU.

## Running Models on Lazy Devices

Building a new PyTorch network or converting an existing one to run on lazy
devices requires only a few lines of lazy-specific code. The following snippets
highlight these lines when running on a single device. For multiple devices, see
the documentation of the specific accelerator.

### Running on a Single Lazy Device

The following snippet shows a network training on a single lazy device:

```python
import lazy_tensor_core.core.lazy_model as ltm

device = ltm.lazy_device()
model = MNIST().train().to(device)
loss_fn = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

for data, target in train_loader:
  optimizer.zero_grad()
  data = data.to(device)
  target = target.to(device)
  output = model(data)
  loss = loss_fn(output, target)
  loss.backward()

  optimizer.step()
  ltm.mark_step()
```

This snippet highlights how easy it is to switch your model to run on a lazy device. The
model definition, dataloader, optimizer and training loop can work on any device.
The only lazy-specific code is a couple lines that acquire the lazy device and
mark the step. Calling
`ltm.mark_step()` at the end of each training
iteration causes the accelerator to execute its current graph and update the
model's parameters. See [Lazy Tensor Deep Dive](#lazy-tensors-deep-dive) for more on
how accelerators create graphs and runs operations.

## Lazy Tensors Deep Dive

Using lazy tensors and devices requires changing only a few lines of code. But
even though lazy tensors act a lot like CPU and CUDA tensors their internals are
different. This section describes what makes lazy tensors unique.

### Lazy Evaluation

CPU and CUDA tensors launch operations immediately or <b>eagerly</b>. Lazy tensors,
on the other hand, record operations in a graph until the results are needed.
Deferring execution like this lets compiler backends optimize it. A graph of
multiple separate operations might be fused into a single optimized operation,
for example.

Lazy execution is generally invisible to the caller. PyTorch/LTC automatically
constructs the graphs, sends them to lazy devices, and synchronizes when
copying data between a lazy device and the CPU. Inserting a barrier when
taking an optimizer step explicitly synchronizes the CPU and the lazy device.

### Lazy Tensors and bfloat16

PyTorch/LTC can use the
[bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
datatype when running on accelerators. In fact, PyTorch/LTC handles float types
(`torch.float` and `torch.double`) differently on TPUs. This behavior is
controlled by the `LTC_USE_BF16` environment variable:

- By default both `torch.float` and `torch.double` are
`torch.float` on TPUs.
- If `LTC_USE_BF16` is set, then `torch.float` and `torch.double` are both
`bfloat16` on TPUs.
- If a PyTorch tensor has `torch.bfloat16` data type, this will be directly
mapped to the accelerator `bfloat16`.

Developers should note that *lazy tensors on accelerators will always report
their PyTorch datatype* regardless of the actual datatype they're using. This
conversion is automatic and opaque.
If a lazy tensor on an accelerator is moved back to the CPU it will be converted
from its actual datatype to its PyTorch datatype. Depending on how your code
operates, this conversion triggered by the type of processing unit can be
important.

### Memory Layout

The internal data representation of lazy tensors is opaque to the user. They
do not expose their storage and they always appear to be contiguous, unlike
CPU and CUDA tensors. This allows accelerator backends to adjust a tensor's
memory layout for better performance.

### Moving Lazy Tensors to and from the CPU

Lazy tensors can be moved from the CPU to a lazy device and from a lazy device
to the CPU. If a view is moved then the data its viewing is also copied to the
other device and the view relationship is not preserved. Put another way,
once data is copied to another device it has no relationship with its
previous device or any tensors on it. Again, depending on how your code operates,
appreciating and accommodating this transition can be important.

### Saving and Loading lazy tensors

Lazy tensors should be moved to the CPU before saving, as in the following snippet:

```python
import torch
import lazy_tensor_core
import lazy_tensor_core.core.lazy_model as ltm

device = ltm.lazy_device()

t0 = torch.randn(2, 2, device=device)
t1 = torch.randn(2, 2, device=device)

tensors = (t0.cpu(), t1.cpu())

torch.save(tensors, 'tensors.pt')

tensors = torch.load('tensors.pt')

t0 = tensors[0].to(device)
t1 = tensors[1].to(device)
```

This lets you put the loaded tensors on any available device, not just the one on which they were initialized.

Per the above note on moving device tensors to the CPU, care must be taken when
working with views. Instead of saving views it is recommended that you recreate
them after the tensors have been loaded and moved to their destination device(s).

A utility API is provided to save data by taking care of previously moving it
to CPU:

```python
import torch
import lazy_tensor_core
import lazy_tensor_core.core.lazy_model as ltm

ltm.save(model.state_dict(), path)
```

In case of multiple devices, the above API will only save the data for the master
device ordinal (0).

In case where memory is limited compared to the size of the model parameters, an
API is provided that reduces the memory footprint on the host:

```python
import lazy_tensor_core.utils.serialization as xser

xser.save(model.state_dict(), path)
```

This API streams lazy tensors to CPU one at a time, reducing the amount of host
memory used, but it requires a matching load API to restore:

```python
import lazy_tensor_core.utils.serialization as xser

state_dict = xser.load(path)
model.load_state_dict(state_dict)
```

Directly saving lazy tensors is possible but not recommended. Lazy
tensors are always loaded back to the device they were saved from, and if
that device is unavailable the load will fail. PyTorch/LTC, like all of PyTorch,
is under active development and this behavior may change in the future.

## Further Reading

Additional documentation is available at the
[PyTorch/LTC repo](https://github.com/pytorch/ltc/). More examples of running
networks on TPUs are available
[here](https://github.com/pytorch-tpu/examples).
