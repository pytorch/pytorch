# functorch

[**Why functorch?**](#why-composable-function-transforms)
| [**Install guide**](#install)
| [**Transformations**](#what-are-the-transforms)
| [**Future Plans**](#future-plans)

`functorch` is a prototype of [JAX-like](https://github.com/google/jax)
composable FUNCtion transforms for pyTORCH.

It aims to provide composable `vmap` and `grad` transforms that work with
PyTorch modules and PyTorch autograd with good eager-mode performance. Because
this project requires some investment, we'd love to hear from and work with
early adopters to shape the design. Please reach out on the issue tracker
if you're interested in using this for your project.

## Why composable function transforms?

There are a number of use cases that are tricky to do in
PyTorch today:
- computing per-sample-gradients (or other per-sample quantities)
- running ensembles of models on a single machine
- efficiently batching together tasks in the inner-loop of MAML
- efficiently computing Jacobians and Hessians
- efficiently computing batched Jacobians and Hessians

Composing `vmap`, `grad`, and `vjp` transforms allows us to express the above
without designing a separate subsystem for each. This idea of composable function
transforms comes from the [JAX framework](https://github.com/google/jax).

## Install

### Colab

Follow the instructions [in this Colab notebook](https://colab.research.google.com/drive/1CrLkqIrydBYP_svnF89UUO-aQEqNPE8x?usp=sharing)

### Binaries

First, set up an environment. We will be installing a nightly PyTorch binary
as well as functorch. If you're using conda, create a conda environment:
```
conda create --name functorch
conda activate functorch
```
If you wish to use `venv` instead:
```
python -m venv functorch-env
source functorch-env/bin/activate
```

Next, install one of the following following PyTorch nightly binaries.
```
# For CUDA 10.2
pip install --pre torch>=1.9.0.dev -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html
# For CUDA 11.1
pip install --pre torch>=1.9.0.dev -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html
# For CPU-only build
pip install --pre torch>=1.9.0.dev -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
```

Install functorch:
```
pip install --user "git+https://github.com/zou3519/functorch.git"
```

Run a quick sanity check in python:
```py
>>> import torch
>>> from functorch import vmap
>>> x = torch.randn(3)
>>> y = vmap(torch.sin)(x)
>>> assert torch.allclose(y, x.sin())
```

### From Source

`functorch` is a PyTorch C++ Extension module. To install,

- Install [PyTorch from source](https://github.com/pytorch/pytorch#from-source).
`functorch` usually runs on the latest development version of PyTorch.
- Run `python setup.py install`. You can use `DEBUG=1` to compile in debug mode.

Then, try to run some tests to make sure all is OK:
```
pytest test/test_vmap.py -v
pytest test/test_eager_transforms.py -v
```

## What are the transforms?

Right now, we support the following transforms:
- `grad`, `vjp`, `jacrev`
- `vmap`

Furthermore, we have some utilities for working with PyTorch modules.
- `make_functional(model)` takes a model and returns its weights and a function
version of the model that has no state.
- `make_functional_with_buffers(model)` takes a model and returns its weights
and buffers and a function version of the model that has no state.

### vmap

Note: `vmap` imposes restrictions on the code that it can be used on.
For more details, please read its docstring.

`vmap(func)(*inputs)` is a transform that adds a dimension to all Tensor
operations in `func`. `vmap(func)` returns a few function that maps `func` over
some dimension (default: 0) of each Tensor in `inputs`.

`vmap` is useful for hiding batch dimensions: one can write a function `func`
that runs on examples and then lift it to a function that can take batches of
examples with `vmap(func)`, leading to a simpler modeling experience:

```py
>>> from functorch import vmap
>>> batch_size, feature_size = 3, 5
>>> weights = torch.randn(feature_size, requires_grad=True)
>>>
>>> def model(feature_vec):
>>>     # Very simple linear model with activation
>>>     assert feature_vec.dim() == 1
>>>     return feature_vec.dot(weights).relu()
>>>
>>> examples = torch.randn(batch_size, feature_size)
>>> result = vmap(model)(examples)
```

### grad

`grad(func)(*inputs)` assumes `func` returns a single-element Tensor. It compute
the gradients of the output of func w.r.t. to `inputs[0]`.

```py
>>> from functorch import grad
>>> x = torch.randn([])
>>> cos_x = grad(lambda x: torch.sin(x))(x)
>>> assert torch.allclose(cos_x, x.cos())
>>>
>>> # Second-order gradients
>>> neg_sin_x = grad(grad(lambda x: torch.sin(x)))(x)
>>> assert torch.allclose(neg_sin_x, -x.sin())
```

When composed with `vmap`, `grad` can be used to compute per-sample-gradients:
```py
>>> from functorch import vmap
>>> batch_size, feature_size = 3, 5
>>>
>>> def model(weights,feature_vec):
>>>     # Very simple linear model with activation
>>>     assert feature_vec.dim() == 1
>>>     return feature_vec.dot(weights).relu()
>>>
>>> def compute_loss(weights, example, target):
>>>     y = model(weights, example)
>>>     return ((y - target) ** 2).mean()  # MSELoss
>>>
>>> weights = torch.randn(feature_size, requires_grad=True)
>>> examples = torch.randn(batch_size, feature_size)
>>> targets = torch.randn(batch_size)
>>> inputs = (weights,examples, targets)
>>> grad_weight_per_example = vmap(grad(compute_loss), in_dims=(None, 0, 0))(*inputs)
```

### vjp and jacrev

```
>>> from functorch import vjp
>>> outputs, vjp_fn = vjp(func, inputs); vjps = vjp_fn(*cotangents)
```
The `vjp` transform applies `func` to `inputs` and returns a new function that
computes vjps given some `contangents` Tensors.

```py
>>> from functorch import jacrev
>>> x = torch.randn(5)
>>> jacobian = jacrev(torch.sin)(x)
>>> expected = torch.diag(x)
>>> assert torch.allclose(jacobian, expected)
```
Use `jacrev` to compute the jacobian. This can be composed with vmap to produce
batched jacobians:

```py
>>> x = torch.randn(64, 5)
>>> jacobian = vmap(jacrev(torch.sin))(x)
>>> assert jacobian.shape == (64, 5, 5)
```

`jacrev` can be composed with itself to produce hessians:
```py
>>> def f(x):
>>>   return x.sin().sum()
>>>
>>> x = torch.randn(5)
>>> hessian = jacrev(jacrev(f))(x)
```

## Debugging
`functorch._C.dump_tensor`: Dumps dispatch keys on stack
`functorch._C._set_vmap_fallback_warning_enabled(False)` if the vmap warning spam bothers you.

## Future Plans

In the end state, we'd like to upstream this into PyTorch once we iron out the
design details. To figure out the details, we need your help -- please send us
your use cases by starting a conversation in the issue tracker or try out the
prototype.

## License
Functorch has a BSD-style license, as found in the [LICENSE](LICENSE) file.
