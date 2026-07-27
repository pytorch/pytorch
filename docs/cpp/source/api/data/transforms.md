---
myst:
  html_meta:
    description: Data transforms in PyTorch C++ — Stack, Normalize, Lambda, and Collate transforms for data pipelines.
    keywords: PyTorch, C++, transforms, Stack, Normalize, Lambda, Collate, data pipeline
---

# Transforms

Transforms apply preprocessing to data samples, such as normalization or
augmentation. They can be chained using the `.map()` method on datasets.

## Transform (Base Class)

The base class for all transforms. Subclass this to create custom transforms.

```{doxygenclass} torch::data::transforms::Transform
:members:
:undoc-members:
```

## BatchTransform (Base Class)

Base class for transforms that operate on entire batches.

```{doxygenclass} torch::data::transforms::BatchTransform
:members:
:undoc-members:
```

## TensorTransform

Base class for transforms that operate on tensors specifically.

```{doxygenclass} torch::data::transforms::TensorTransform
:members:
:undoc-members:
```

## Normalize

Normalizes tensors with a given mean and standard deviation.

```{doxygenstruct} torch::data::transforms::Normalize
:members:
:undoc-members:
```

## Stack

Stacks a batch of tensors into a single tensor.

```{doxygenstruct} torch::data::transforms::Stack
:members:
:undoc-members:
```

**Example:**

```cpp
auto dataset = torch::data::datasets::MNIST("./data")
    .map(torch::data::transforms::Normalize<>(0.5, 0.5))
    .map(torch::data::transforms::Stack<>());
```

## Lambda

```{doxygenclass} torch::data::transforms::Lambda
:members:
:undoc-members:
```

## TensorLambda

```{doxygenclass} torch::data::transforms::TensorLambda
:members:
:undoc-members:
```

## BatchLambda

```{doxygenclass} torch::data::transforms::BatchLambda
:members:
:undoc-members:
```

## Chaining Transforms

Transforms can be chained together using `.map()`:

```cpp
auto dataset = torch::data::datasets::MNIST("./data")
    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
    .map(torch::data::transforms::Stack<>());
```
