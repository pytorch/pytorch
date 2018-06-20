#pragma once

#include <torch/expanding_array.h>
#include <torch/nn/modules/functional.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <ATen/ATen.h>

#include <cstddef>
#include <functional>
#include <tuple>

namespace torch {
namespace nn {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxPool ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <size_t D>
struct MaxPoolOptions {
  MaxPoolOptions(const ExpandingArray<D>& kernel_size);
  TORCH_ARG(ExpandingArray<D>, kernel_size);
  TORCH_ARG(ExpandingArray<D>, stride); // = kernel_size
  TORCH_ARG(ExpandingArray<D>, padding) = 0;
  TORCH_ARG(ExpandingArray<D>, dilation) = 1;
  TORCH_ARG(bool, ceil_mode) = false;
};

namespace detail {
template <size_t D, typename Derived>
class MaxPoolImplBase : public FunctionalImpl {
 public:
  using MaxPoolFunction = std::function<std::tuple<at::Tensor, at::Tensor>(
      const at::Tensor&,
      at::IntList,
      at::IntList,
      at::IntList,
      at::IntList,
      bool)>;

  explicit MaxPoolImplBase(MaxPoolOptions<D> options, MaxPoolFunction max_pool);

  const MaxPoolOptions<D>& options() const noexcept;

 protected:
  MaxPoolOptions<D> options_;
};
} // namespace detail

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxPool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

using MaxPool1dOptions = MaxPoolOptions<1>;

class MaxPool1dImpl : public detail::MaxPoolImplBase<1, MaxPool1dImpl> {
 public:
  explicit MaxPool1dImpl(MaxPool1dOptions options);
  explicit MaxPool1dImpl(const ExpandingArray<1>& kernel_size);
};

TORCH_MODULE(MaxPool1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxPool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

using MaxPool2dOptions = MaxPoolOptions<2>;

class MaxPool2dImpl : public detail::MaxPoolImplBase<2, MaxPool2dImpl> {
 public:
  explicit MaxPool2dImpl(MaxPool2dOptions options);
  explicit MaxPool2dImpl(const ExpandingArray<2>& kernel_size);
};

TORCH_MODULE(MaxPool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxPool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

using MaxPool3dOptions = MaxPoolOptions<3>;

class MaxPool3dImpl : public detail::MaxPoolImplBase<3, MaxPool3dImpl> {
 public:
  explicit MaxPool3dImpl(MaxPool3dOptions options);
  explicit MaxPool3dImpl(const ExpandingArray<3>& kernel_size);
};

TORCH_MODULE(MaxPool3d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AvgPool ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <size_t D>
struct AvgPoolOptions {
  AvgPoolOptions(const ExpandingArray<D>& kernel_size);
  TORCH_ARG(ExpandingArray<D>, kernel_size);
  TORCH_ARG(ExpandingArray<D>, stride); // = kernel_size
  TORCH_ARG(ExpandingArray<D>, padding) = 0;
  TORCH_ARG(bool, count_include_pad) = true;
  TORCH_ARG(bool, ceil_mode) = false;
};

namespace detail {
template <size_t D, typename Derived>
class AvgPoolImplBase : public FunctionalImpl {
 public:
  using AvgPoolFunction = std::function<at::Tensor(
      const at::Tensor&,
      at::IntList,
      at::IntList,
      at::IntList,
      bool,
      bool)>;

  explicit AvgPoolImplBase(AvgPoolOptions<D> options, AvgPoolFunction avg_pool);

  const AvgPoolOptions<D>& options() const noexcept;

 protected:
  AvgPoolOptions<D> options_;
};
} // namespace detail

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AvgPool1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

using AvgPool1dOptions = AvgPoolOptions<1>;

class AvgPool1dImpl : public detail::AvgPoolImplBase<1, AvgPool1dImpl> {
 public:
  explicit AvgPool1dImpl(AvgPool1dOptions options);
  explicit AvgPool1dImpl(const ExpandingArray<1>& kernel_size);
};

TORCH_MODULE(AvgPool1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AvgPool2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

using AvgPool2dOptions = AvgPoolOptions<2>;

class AvgPool2dImpl : public detail::AvgPoolImplBase<2, AvgPool2dImpl> {
 public:
  explicit AvgPool2dImpl(AvgPool2dOptions options);
  explicit AvgPool2dImpl(const ExpandingArray<2>& kernel_size);
};

TORCH_MODULE(AvgPool2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AvgPool3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

using AvgPool3dOptions = AvgPoolOptions<3>;

class AvgPool3dImpl : public detail::AvgPoolImplBase<3, AvgPool3dImpl> {
 public:
  explicit AvgPool3dImpl(AvgPool3dOptions options);
  explicit AvgPool3dImpl(const ExpandingArray<3>& kernel_size);
};

TORCH_MODULE(AvgPool3d);

} // namespace nn
} // namespace torch
