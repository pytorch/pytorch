---
myst:
  html_meta:
    description: Functional API in PyTorch C++ — torch::nn::functional stateless operations for neural networks.
    keywords: PyTorch, C++, functional, torch::nn::functional, relu, conv2d, linear, softmax
---

# Functional API

The `torch::nn::functional` namespace provides stateless versions of neural
network operations. Unlike module classes, functional operations do not hold
learnable parameters — you pass weights explicitly.

**When to use functional vs modules:**

- Use **modules** (`torch::nn::Conv2d`) when you need learnable parameters
  managed automatically (training, saving, loading).
- Use **functional** (`torch::nn::functional::conv2d`) when you already have
  weights as tensors, or for operations without parameters (e.g., `relu`).

```cpp
#include <torch/nn/functional.h>
namespace F = torch::nn::functional;

// Stateless activation — no module needed
auto output = F::relu(input);

// Convolution with explicit weight tensor
auto output = F::conv2d(input, weight, F::Conv2dFuncOptions().stride(1).padding(1));

// Softmax along a dimension
auto probs = F::softmax(logits, F::SoftmaxFuncOptions(/*dim=*/1));
```

## Activation Functions

```{doxygenfunction} torch::nn::functional::elu
```
```{doxygenfunction} torch::nn::functional::selu
```
```{doxygenfunction} torch::nn::functional::hardshrink
```
```{doxygenfunction} torch::nn::functional::hardtanh
```
```{doxygenfunction} torch::nn::functional::leaky_relu
```
```{doxygenfunction} torch::nn::functional::logsigmoid
```
```{doxygenfunction} torch::nn::functional::glu
```
```{doxygenfunction} torch::nn::functional::gelu
```
```{doxygenfunction} torch::nn::functional::silu
```
```{doxygenfunction} torch::nn::functional::mish
```
```{doxygenfunction} torch::nn::functional::prelu
```
```{doxygenfunction} torch::nn::functional::relu
```
```{doxygenfunction} torch::nn::functional::relu6
```
```{doxygenfunction} torch::nn::functional::rrelu
```
```{doxygenfunction} torch::nn::functional::celu
```
```{doxygenfunction} torch::nn::functional::softplus
```
```{doxygenfunction} torch::nn::functional::softshrink
```
```{doxygenfunction} torch::nn::functional::softsign
```
```{doxygenfunction} torch::nn::functional::tanhshrink
```
```{doxygenfunction} torch::nn::functional::threshold
```
```{doxygenfunction} torch::nn::functional::softmax
```
```{doxygenfunction} torch::nn::functional::softmin
```
```{doxygenfunction} torch::nn::functional::log_softmax
```
```{doxygenfunction} torch::nn::functional::gumbel_softmax
```

## Convolution Functions

```{doxygenfunction} torch::nn::functional::conv1d
```
```{doxygenfunction} torch::nn::functional::conv2d
```
```{doxygenfunction} torch::nn::functional::conv3d
```
```{doxygenfunction} torch::nn::functional::conv_transpose1d
```
```{doxygenfunction} torch::nn::functional::conv_transpose2d
```
```{doxygenfunction} torch::nn::functional::conv_transpose3d
```

## Pooling Functions

```{doxygenfunction} torch::nn::functional::avg_pool1d
```
```{doxygenfunction} torch::nn::functional::avg_pool2d
```
```{doxygenfunction} torch::nn::functional::avg_pool3d
```
```{doxygenfunction} torch::nn::functional::max_pool1d
```
```{doxygenfunction} torch::nn::functional::max_pool2d
```
```{doxygenfunction} torch::nn::functional::max_pool3d
```
```{doxygenfunction} torch::nn::functional::max_pool1d_with_indices
```
```{doxygenfunction} torch::nn::functional::max_pool2d_with_indices
```
```{doxygenfunction} torch::nn::functional::max_pool3d_with_indices
```
```{doxygenfunction} torch::nn::functional::adaptive_max_pool1d
```
```{doxygenfunction} torch::nn::functional::adaptive_max_pool2d
```
```{doxygenfunction} torch::nn::functional::adaptive_max_pool3d
```
```{doxygenfunction} torch::nn::functional::adaptive_avg_pool1d
```
```{doxygenfunction} torch::nn::functional::adaptive_avg_pool2d
```
```{doxygenfunction} torch::nn::functional::adaptive_avg_pool3d
```
```{doxygenfunction} torch::nn::functional::max_unpool1d
```
```{doxygenfunction} torch::nn::functional::max_unpool2d
```
```{doxygenfunction} torch::nn::functional::max_unpool3d
```
```{doxygenfunction} torch::nn::functional::fractional_max_pool2d
```
```{doxygenfunction} torch::nn::functional::fractional_max_pool3d
```
```{doxygenfunction} torch::nn::functional::lp_pool1d
```
```{doxygenfunction} torch::nn::functional::lp_pool2d
```
```{doxygenfunction} torch::nn::functional::lp_pool3d
```

## Linear Functions

```{doxygenfunction} torch::nn::functional::linear
```
```{doxygenfunction} torch::nn::functional::bilinear
```

## Dropout Functions

```{doxygenfunction} torch::nn::functional::dropout
```
```{doxygenfunction} torch::nn::functional::dropout2d
```
```{doxygenfunction} torch::nn::functional::dropout3d
```
```{doxygenfunction} torch::nn::functional::alpha_dropout
```
```{doxygenfunction} torch::nn::functional::feature_alpha_dropout
```

## Embedding Functions

```{doxygenfunction} torch::nn::functional::one_hot
```
```{doxygenfunction} torch::nn::functional::embedding
```
```{doxygenfunction} torch::nn::functional::embedding_bag
```

## Normalization Functions

```{doxygenfunction} torch::nn::functional::batch_norm
```
```{doxygenfunction} torch::nn::functional::instance_norm
```
```{doxygenfunction} torch::nn::functional::layer_norm
```
```{doxygenfunction} torch::nn::functional::group_norm
```
```{doxygenfunction} torch::nn::functional::local_response_norm
```
```{doxygenfunction} torch::nn::functional::normalize
```

## Loss Functions

```{doxygenfunction} torch::nn::functional::l1_loss
```
```{doxygenfunction} torch::nn::functional::mse_loss
```
```{doxygenfunction} torch::nn::functional::binary_cross_entropy
```
```{doxygenfunction} torch::nn::functional::binary_cross_entropy_with_logits
```
```{doxygenfunction} torch::nn::functional::cross_entropy
```
```{doxygenfunction} torch::nn::functional::nll_loss
```
```{doxygenfunction} torch::nn::functional::kl_div
```
```{doxygenfunction} torch::nn::functional::smooth_l1_loss(const Tensor& input, const Tensor& target, const SmoothL1LossFuncOptions& options)
```
```{doxygenfunction} torch::nn::functional::huber_loss
```
```{doxygenfunction} torch::nn::functional::hinge_embedding_loss
```
```{doxygenfunction} torch::nn::functional::multi_margin_loss
```
```{doxygenfunction} torch::nn::functional::cosine_embedding_loss
```
```{doxygenfunction} torch::nn::functional::margin_ranking_loss
```
```{doxygenfunction} torch::nn::functional::multilabel_margin_loss
```
```{doxygenfunction} torch::nn::functional::soft_margin_loss
```
```{doxygenfunction} torch::nn::functional::multilabel_soft_margin_loss
```
```{doxygenfunction} torch::nn::functional::triplet_margin_loss
```
```{doxygenfunction} torch::nn::functional::triplet_margin_with_distance_loss
```
```{doxygenfunction} torch::nn::functional::ctc_loss
```
```{doxygenfunction} torch::nn::functional::poisson_nll_loss
```

## Distance Functions

```{doxygenfunction} torch::nn::functional::cosine_similarity
```
```{doxygenfunction} torch::nn::functional::pairwise_distance
```
```{doxygenfunction} torch::nn::functional::pdist
```

## Vision Functions

```{doxygenfunction} torch::nn::functional::interpolate
```
```{doxygenfunction} torch::nn::functional::affine_grid
```
```{doxygenfunction} torch::nn::functional::grid_sample
```
```{doxygenfunction} torch::nn::functional::pad
```
```{doxygenfunction} torch::nn::functional::pixel_shuffle
```
```{doxygenfunction} torch::nn::functional::pixel_unshuffle
```

## Fold/Unfold

```{doxygenfunction} torch::nn::functional::fold
```
```{doxygenfunction} torch::nn::functional::unfold
```

## Functional Options Structs

Each functional operation that takes configuration uses a corresponding options
struct. The naming convention is `<Operation>FuncOptions`.

**Activation Options:**

```{doxygentypedef} torch::nn::functional::ELUFuncOptions
```
```{doxygentypedef} torch::nn::functional::SELUFuncOptions
```
```{doxygentypedef} torch::nn::functional::GLUFuncOptions
```
```{doxygentypedef} torch::nn::functional::GELUFuncOptions
```
```{doxygentypedef} torch::nn::functional::HardshrinkFuncOptions
```
```{doxygentypedef} torch::nn::functional::HardtanhFuncOptions
```
```{doxygentypedef} torch::nn::functional::LeakyReLUFuncOptions
```
```{doxygentypedef} torch::nn::functional::ReLUFuncOptions
```
```{doxygentypedef} torch::nn::functional::ReLU6FuncOptions
```
```{doxygentypedef} torch::nn::functional::CELUFuncOptions
```
```{doxygentypedef} torch::nn::functional::SoftplusFuncOptions
```
```{doxygentypedef} torch::nn::functional::SoftshrinkFuncOptions
```
```{doxygentypedef} torch::nn::functional::ThresholdFuncOptions
```

**Convolution Options:**

```{doxygentypedef} torch::nn::functional::Conv1dFuncOptions
```
```{doxygentypedef} torch::nn::functional::Conv2dFuncOptions
```
```{doxygentypedef} torch::nn::functional::Conv3dFuncOptions
```
```{doxygentypedef} torch::nn::functional::ConvTranspose1dFuncOptions
```
```{doxygentypedef} torch::nn::functional::ConvTranspose2dFuncOptions
```
```{doxygentypedef} torch::nn::functional::ConvTranspose3dFuncOptions
```

**Pooling Options:**

```{doxygentypedef} torch::nn::functional::AvgPool1dFuncOptions
```
```{doxygentypedef} torch::nn::functional::AvgPool2dFuncOptions
```
```{doxygentypedef} torch::nn::functional::AvgPool3dFuncOptions
```
```{doxygentypedef} torch::nn::functional::MaxPool1dFuncOptions
```
```{doxygentypedef} torch::nn::functional::MaxPool2dFuncOptions
```
```{doxygentypedef} torch::nn::functional::MaxPool3dFuncOptions
```
```{doxygentypedef} torch::nn::functional::AdaptiveMaxPool1dFuncOptions
```
```{doxygentypedef} torch::nn::functional::AdaptiveMaxPool2dFuncOptions
```
```{doxygentypedef} torch::nn::functional::AdaptiveMaxPool3dFuncOptions
```
```{doxygentypedef} torch::nn::functional::AdaptiveAvgPool1dFuncOptions
```
```{doxygentypedef} torch::nn::functional::AdaptiveAvgPool2dFuncOptions
```
```{doxygentypedef} torch::nn::functional::AdaptiveAvgPool3dFuncOptions
```

**Other Options:**

```{doxygentypedef} torch::nn::functional::CosineSimilarityFuncOptions
```
```{doxygentypedef} torch::nn::functional::PairwiseDistanceFuncOptions
```
```{doxygentypedef} torch::nn::functional::Dropout2dFuncOptions
```
```{doxygentypedef} torch::nn::functional::Dropout3dFuncOptions
```
```{doxygentypedef} torch::nn::functional::L1LossFuncOptions
```
```{doxygentypedef} torch::nn::functional::FoldFuncOptions
```
```{doxygentypedef} torch::nn::functional::UnfoldFuncOptions
```
```{doxygentypedef} torch::nn::functional::PixelShuffleFuncOptions
```
```{doxygentypedef} torch::nn::functional::PixelUnshuffleFuncOptions
```
