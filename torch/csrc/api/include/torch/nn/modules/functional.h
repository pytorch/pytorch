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

  /// A small type that is used only in the constructor of `FunctionalImpl`,
  /// that allows constructing it with a function with more than one argument,
  /// and binding all but the first parameter to specific values. It is
  /// necessary due to interaction with the `ModuleHolder` class, which expects
  /// to construct a module with `Module({...})` when there is more than one
  /// argument. It also deals with argument binding.
  struct BoundFunction {
    template <
        typename AnyFunction,
        typename... Args,
        typename = torch::enable_if_t<(sizeof...(Args) > 0)>>
    /* implicit */ BoundFunction(AnyFunction original_function, Args&&... args)
        : function_(std::bind(
              original_function,
              /*input=*/std::placeholders::_1,
              std::forward<Args>(args)...)) {
      // std::bind is normally evil, but (1) gcc is broken w.r.t. handling
      // parameter pack expansion in lambdas and (2) moving parameter packs into
      // a lambda only works with C++14, so std::bind is the more move-aware
      // solution here.
    }

    Function function_;
  };

  explicit FunctionalImpl(Function function);
  explicit FunctionalImpl(BoundFunction bound_function);

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
