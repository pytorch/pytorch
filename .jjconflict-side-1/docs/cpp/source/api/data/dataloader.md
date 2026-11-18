---
myst:
  html_meta:
    description: DataLoader in PyTorch C++ — parallel data loading with batching, sampling, and multi-worker support.
    keywords: PyTorch, C++, DataLoader, data loading, batching, workers, make_data_loader
---

# DataLoader

The DataLoader batches samples from a dataset and optionally shuffles and
parallelizes the loading process. It is the main interface for iterating
over training data.

## DataLoader Classes

```{doxygenclass} torch::data::DataLoaderBase
:members:
:undoc-members:
```

## DataLoaderOptions

```{doxygenstruct} torch::data::DataLoaderOptions
:members:
:undoc-members:
```

## StatefulDataLoader

A DataLoader for `StatefulDataset` types that manage their own batching logic
internally.

```{doxygenclass} torch::data::StatefulDataLoader
:members:
:undoc-members:
```

## StatelessDataLoader

A DataLoader for `Dataset` types that use external samplers for batching.

```{doxygenclass} torch::data::StatelessDataLoader
:members:
:undoc-members:
```

## Iterator

```{doxygenclass} torch::data::Iterator
:members:
:undoc-members:
```

## Creating a DataLoader

Use `make_data_loader` to create a DataLoader from a dataset:

```cpp
auto data_loader = torch::data::make_data_loader(
    std::move(dataset),
    torch::data::DataLoaderOptions()
        .batch_size(64)
        .workers(4));

for (auto& batch : *data_loader) {
    auto data = batch.data;
    auto target = batch.target;
    // Train on batch
}
```

## Complete Training Example

```cpp
#include <torch/torch.h>

int main() {
  // Load dataset
  auto dataset = torch::data::datasets::MNIST("./data")
      .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
      .map(torch::data::transforms::Stack<>());

  // Create data loader
  auto data_loader = torch::data::make_data_loader(
      std::move(dataset),
      torch::data::DataLoaderOptions().batch_size(64).workers(2));

  // Create model and optimizer
  auto model = std::make_shared<Net>();
  auto optimizer = torch::optim::Adam(model->parameters(), 0.001);

  // Training loop
  for (size_t epoch = 1; epoch <= 10; ++epoch) {
    for (auto& batch : *data_loader) {
      optimizer.zero_grad();
      auto output = model->forward(batch.data);
      auto loss = torch::nll_loss(output, batch.target);
      loss.backward();
      optimizer.step();
    }
  }
}
```
