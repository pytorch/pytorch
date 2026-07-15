---
myst:
  html_meta:
    description: Save and load functions in PyTorch C++ — torch::save and torch::load for tensors and modules.
    keywords: PyTorch, C++, save, load, torch::save, torch::load, tensor, module
---

# Saving and Loading

The primary interface for serialization uses the `torch::save` and
`torch::load` functions, which can save and load tensors, modules,
and optimizers.

## Save Functions

```{doxygenfunction} torch::save(const Value &value, SaveToArgs&&... args)
```

```{doxygenfunction} torch::save(const std::vector<torch::Tensor> &tensor_vec, SaveToArgs&&... args)
```

## Load Functions

```{doxygenfunction} torch::load(Value &value, LoadFromArgs&&... args)
```

```{doxygenfunction} torch::load(std::vector<torch::Tensor> &tensor_vec, LoadFromArgs&&... args)
```

## Saving and Loading Tensors

```cpp
// Save a tensor
torch::Tensor tensor = torch::randn({2, 3});
torch::save(tensor, "tensor.pt");

// Load a tensor
torch::Tensor loaded;
torch::load(loaded, "tensor.pt");
```

## Saving and Loading Modules

```cpp
// Define a model
struct Net : torch::nn::Module {
  Net() {
    fc1 = register_module("fc1", torch::nn::Linear(784, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 10));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    return fc2->forward(x);
  }

  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

// Save model
auto model = std::make_shared<Net>();
torch::save(model, "model.pt");

// Load model
auto loaded_model = std::make_shared<Net>();
torch::load(loaded_model, "model.pt");
```

## Saving Optimizer State

```cpp
auto model = std::make_shared<Net>();
auto optimizer = torch::optim::Adam(model->parameters(), 0.001);

// Train...

// Save both model and optimizer
torch::save(model, "model.pt");
torch::save(optimizer, "optimizer.pt");

// Load both
torch::load(model, "model.pt");
torch::load(optimizer, "optimizer.pt");
```
