---
myst:
  html_meta:
    description: PyTorch C++ neural network modules — torch::nn API for defining and training models.
    keywords: PyTorch, C++, nn, Module, neural network, torch::nn
---

# Neural Network Modules (torch::nn)

The `torch::nn` namespace provides neural network building blocks that mirror
Python's `torch.nn` module. It uses a PIMPL (Pointer to Implementation) pattern
where user-facing classes like `Conv2d` wrap internal `Conv2dImpl` classes.

**When to use torch::nn:**

- Building neural network models in C++
- Creating custom layers and modules
- Porting Python models to C++ for production inference
- Training models entirely in C++

**Basic usage:**

```cpp
#include <torch/torch.h>

// Define a simple model
struct Net : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Linear fc1{nullptr};

    Net() {
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(1, 32, 3).stride(1).padding(1)));
        fc1 = register_module("fc1", torch::nn::Linear(32 * 28 * 28, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1->forward(x));
        x = x.view({-1, 32 * 28 * 28});
        return fc1->forward(x);
    }
};

// Create and use the model
auto model = std::make_shared<Net>();
auto input = torch::randn({1, 1, 28, 28});
auto output = model->forward(input);
```

## Header Files

- `torch/nn.h` - Main neural network header (includes all modules)
- `torch/nn/module.h` - Base Module class
- `torch/nn/modules.h` - All module implementations
- `torch/nn/options.h` - Options structs for modules
- `torch/nn/functional.h` - Functional API

## Module Base Class

All neural network modules inherit from `torch::nn::Module`, which provides
parameter management, serialization, device/dtype conversion, and hooks.

```{doxygenclass} torch::nn::Module
```

**Key features:**

- `register_module()`: Register submodules for parameter tracking
- `register_parameter()`: Register learnable parameters
- `register_buffer()`: Register non-learnable state (e.g., running mean)
- `parameters()` / `named_parameters()`: Iterate over all parameters
- `to()`: Move module to a device or convert dtype
- `train()` / `eval()`: Toggle training/evaluation mode
- `save()` / `load()`: Serialize and deserialize module state

## Module Categories

```{toctree}
:maxdepth: 1

containers
convolution
pooling
linear
activation
normalization
dropout
embedding
recurrent
transformer
loss
functional
utilities
```
