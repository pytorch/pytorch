#pragma once

#include <torch/csrc/utils/variadic.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <functional>

namespace torch {
namespace nn {

// Lets you create a container from a function, designed for use in
// Sequential.
class FunctionalImpl : public torch::nn::Cloneable<FunctionalImpl> {
 public:
  using Function = std::function<Tensor(Tensor)>;

  explicit FunctionalImpl(Function function);

  template <
      typename SomeFunction,
      typename... Args,
      typename = torch::enable_if_t<(sizeof...(Args) > 0)>>
  explicit FunctionalImpl(SomeFunction original_function, Args&&... args)
      : function_(std::bind(
            original_function,
            /*input=*/std::placeholders::_1,
            std::forward<Args>(args)...)) {
    // std::bind is normally evil, but (1) gcc is broken w.r.t. handling
    // parameter pack expansion in lambdas and (2) moving parameter packs into
    // a lambda only works with C++14, so std::bind is the more move-aware
    // solution here.
  }

  void reset() override;
  Tensor forward(Tensor input);

  /// Calls forward(input).
  Tensor operator()(Tensor input);

 private:
  Function function_;
};

TORCH_MODULE(Functional);

} // namespace nn
} // namespace torch
