#pragma once

#include <c10/util/irange.h>

/// This macro enables a module with default arguments in its forward method
/// to be used in a Sequential module.
///
/// Example usage:
///
/// Let's say we have a module declared like this:
/// ```
/// struct MImpl : torch::nn::Module {
///  public:
///   explicit MImpl(int value_) : value(value_) {}
///   torch::Tensor forward(int a, int b = 2, double c = 3.0) {
///     return torch::tensor(a + b + c);
///   }
///  private:
///   int value;
/// };
/// TORCH_MODULE(M);
/// ```
///
/// If we try to use it in a Sequential module and run forward:
/// ```
/// torch::nn::Sequential seq(M(1));
/// seq->forward(1);
/// ```
///
/// We will receive the following error message:
/// ```
/// MImpl's forward() method expects 3 argument(s), but received 1.
/// If MImpl's forward() method has default arguments, please make sure
/// the forward() method is declared with a corresponding
/// `FORWARD_HAS_DEFAULT_ARGS` macro.
/// ```
///
/// The right way to fix this error is to use the `FORWARD_HAS_DEFAULT_ARGS`
/// macro when declaring the module:
/// ```
/// struct MImpl : torch::nn::Module {
///  public:
///   explicit MImpl(int value_) : value(value_) {}
///   torch::Tensor forward(int a, int b = 2, double c = 3.0) {
///     return torch::tensor(a + b + c);
///   }
///  protected:
///   /*
///   NOTE: looking at the argument list of `forward`:
///   `forward(int a, int b = 2, double c = 3.0)`
///   we saw the following default arguments:
///   ----------------------------------------------------------------
///   0-based index of default |         Default value of arg
///   arg in forward arg list  |  (wrapped by `torch::nn::AnyValue()`)
///   ----------------------------------------------------------------
///               1            |       torch::nn::AnyValue(2)
///               2            |       torch::nn::AnyValue(3.0)
///   ----------------------------------------------------------------
///   Thus we pass the following arguments to the `FORWARD_HAS_DEFAULT_ARGS`
///   macro:
///   */
///   FORWARD_HAS_DEFAULT_ARGS({1, torch::nn::AnyValue(2)}, {2,
///   torch::nn::AnyValue(3.0)})
///  private:
///   int value;
/// };
/// TORCH_MODULE(M);
/// ```
/// Now, running the following would work:
/// ```
/// torch::nn::Sequential seq(M(1));
/// seq->forward(1);  // This correctly populates the default arguments for
/// `MImpl::forward`
/// ```
#define FORWARD_HAS_DEFAULT_ARGS(...)                                       \
  template <typename ModuleType, typename... ArgumentTypes>                 \
  friend struct torch::nn::AnyModuleHolder;                                 \
  bool _forward_has_default_args() override {                               \
    return true;                                                            \
  }                                                                         \
  unsigned int _forward_num_required_args() override {                      \
    std::vector<std::pair<unsigned int, torch::nn::AnyValue>> args_info = { \
        __VA_ARGS__};                                                       \
    return args_info[0].first;                                              \
  }                                                                         \
  std::vector<torch::nn::AnyValue> _forward_populate_default_args(          \
      std::vector<torch::nn::AnyValue>&& arguments) override {              \
    std::vector<std::pair<unsigned int, torch::nn::AnyValue>> args_info = { \
        __VA_ARGS__};                                                       \
    unsigned int num_all_args = args_info[args_info.size() - 1].first + 1;  \
    TORCH_INTERNAL_ASSERT(                                                  \
        arguments.size() >= _forward_num_required_args() &&                 \
        arguments.size() <= num_all_args);                                  \
    std::vector<torch::nn::AnyValue> ret;                                   \
    ret.reserve(num_all_args);                                              \
    for (const auto i : c10::irange(arguments.size())) {                    \
      ret.emplace_back(std::move(arguments[i]));                            \
    }                                                                       \
    for (auto& arg_info : args_info) {                                      \
      if (arg_info.first > ret.size() - 1)                                  \
        ret.emplace_back(std::move(arg_info.second));                       \
    }                                                                       \
    return ret;                                                             \
  }
