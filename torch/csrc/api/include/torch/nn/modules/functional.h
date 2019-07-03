#pragma once

#include <torch/csrc/utils/variadic.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <functional>
#include <utility>
//#include <torch/nn/modules/bind_utils.h>

namespace torch {
namespace nn {

template<class Func>
struct function_traits {
  static_assert(!std::is_same<Func, Func>::value, "In function_traits<Func>, Func must be a plain function type.");
};

template<class Result, class... Args>
struct function_traits<Result(*)(Args...)> {
  static constexpr auto number_of_parameters = sizeof...(Args);
};

/// Wraps a function in a `Module`.
///
/// The `Functional` module allows wrapping an arbitrary function or function
/// object in an `nn::Module`. This is primarily handy for usage in
/// `Sequential`.
///
/// \rst
/// .. code-block:: cpp
///
///   Sequential sequential(
///     Linear(3, 4),
///     Functional(torch::relu),
///     BatchNorm(3),
///     Functional(torch::elu, /*alpha=*/1));
/// \endrst
///
/// While a `Functional` module only accepts a single `Tensor` as input, it is
/// possible for the the wrapped function to accept further arguments. However,
/// these have to be bound *at construction time*. For example, if
/// you want to wrap `torch::leaky_relu`, which accepts a `slope` scalar as its
/// second argument, with a particular value for its `slope` in a `Functional`
/// module, you could write
///
/// \rst
/// .. code-block:: cpp
///
///   Functional(torch::leaky_relu, /*slope=*/0.5)
/// \endrst
///
/// The value of `0.5` is then stored within the `Functional` object and supplied
/// to the function call at invocation time. Note that such bound values are
/// evaluated eagerly and stored a single time. See the documentation of
/// [std::bind](https://en.cppreference.com/w/cpp/utility/functional/bind) for
/// more information on the semantics of argument binding.
///
/// \rst
/// .. attention::
///   After passing any bound arguments, the function must accept a single
///   tensor and return a single tensor.
/// \endrst
///
/// Note that `Functional` overloads the call operator (`operator()`) such that
/// you can invoke it with `my_func(...)`.
class TORCH_API FunctionalImpl : public torch::nn::Cloneable<FunctionalImpl> {
 public:
  using Function = std::function<Tensor(Tensor)>;

  /// Constructs a `Functional` from a function object.
  explicit FunctionalImpl(Function function);

  template <typename SomeFunction,
      typename... Args,
      typename std::enable_if<(sizeof...(Args) >= function_traits<SomeFunction>::number_of_parameters)>::type* = nullptr>
  Function bind(SomeFunction f, Args... args) {
    static_assert(!std::is_same<SomeFunction, SomeFunction>::value, "Number of arguments is more then function can accept");
    return nullptr;
  }

  template <typename SomeFunction,
      typename... Args,
      typename std::enable_if<(sizeof...(Args) == function_traits<SomeFunction>::number_of_parameters - 1)>::type* = nullptr>
  Function bind(SomeFunction f, Args... args) {
    return std::bind(f, std::placeholders::_1, std::forward<Args>(args)...);
  }

  // This is a workaround to be able to bind a function that has optional argument at the end of function's signature
  // and this argument was not provided through args
  // It was added to be able to support backward compatibility with previous signature of the function avg_pool2d
  // Other solutions can be: move to gcc 4.9.0 and use lambda or break BC and force to add optional parameters to Functional
  template <typename SomeFunction,
      typename... Args,
      typename std::enable_if<(sizeof...(Args) < function_traits<SomeFunction>::number_of_parameters - 1)>::type* = nullptr>
  Function bind(SomeFunction f, Args... args) {
    return bind(f, std::forward<Args>(args)..., c10::nullopt);
  }

  template <
      typename SomeFunction,
      typename... Args,
      typename = torch::enable_if_t<(sizeof...(Args) > 0)>>
  explicit FunctionalImpl(SomeFunction original_function, Args&&... args)
    :function_(bind(
        original_function,
        args...)) {
    // std::bind is normally evil, but (1) gcc is broken w.r.t. handling
    // parameter pack expansion in lambdas and (2) moving parameter packs into
    // a lambda only works with C++14, so std::bind is the more move-aware
    // solution here.
  }

  void reset() override;

  /// Pretty prints the `Functional` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Forwards the `input` tensor to the underlying (bound) function object.
  Tensor forward(Tensor input);

  /// Calls forward(input).
  Tensor operator()(Tensor input);

  bool is_serializable() const override;

 private:
  Function function_;
};

/// A `ModuleHolder` subclass for `FunctionalImpl`.
/// See the documentation for `FunctionalImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Functional);

} // namespace nn
} // namespace torch
