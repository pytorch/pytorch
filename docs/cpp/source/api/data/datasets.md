---
myst:
  html_meta:
    description: Dataset classes in PyTorch C++ — Dataset, MapDataset, StreamDataset, and built-in datasets like MNIST.
    keywords: PyTorch, C++, Dataset, MapDataset, StreamDataset, MNIST, data
---

# Datasets

The dataset abstraction defines how to access individual samples in your data.
All datasets inherit from `Dataset` and must implement `get()` and `size()`.

## Dataset Base Class

```{doxygenclass} torch::data::datasets::Dataset
:members:
:undoc-members:
```

```{doxygenclass} torch::data::datasets::BatchDataset
:members:
:undoc-members:
```

## StatefulDataset

A dataset that manages its own state across batches (e.g., position in a stream).
Unlike `Dataset`, it produces batches directly without external samplers.

```{doxygenclass} torch::data::datasets::StatefulDataset
:members:
:undoc-members:
```

## ChunkDataReader

Interface for reading chunks of data from a data source. Used with
`ChunkDataset` for large-scale data loading.

```{doxygenclass} torch::data::datasets::ChunkDataReader
:members:
:undoc-members:
```

## Custom Dataset Example

```cpp
class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {
 public:
  explicit CustomDataset(const std::string& root) {
    // Load data from root directory
  }

  torch::data::Example<> get(size_t index) override {
    return {images_[index], labels_[index]};
  }

  torch::optional<size_t> size() const override {
    return images_.size(0);
  }

 private:
  torch::Tensor images_, labels_;
};
```

## MapDataset

```{doxygenclass} torch::data::datasets::MapDataset
:members:
:undoc-members:
```

## ChunkDataset

```{doxygenclass} torch::data::datasets::ChunkDataset
:members:
:undoc-members:
```

## SharedBatchDataset

```{doxygenclass} torch::data::datasets::SharedBatchDataset
:members:
:undoc-members:
```

## Built-in Datasets

### MNIST

```{doxygenclass} torch::data::datasets::MNIST
:members:
:undoc-members:
```

**Example:**

```cpp
auto dataset = torch::data::datasets::MNIST("./data")
    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
    .map(torch::data::transforms::Stack<>());
```

## Example Struct

```{doxygenstruct} torch::data::Example
:members:
:undoc-members:
```
