#include <torch/nn/modules/container/named_any.h>

namespace torch {
namespace nn {

torch::OrderedDict<std::string, AnyModule> modules_ordered_dict(
  std::initializer_list<NamedAnyModule> named_modules) {
  TORCH_WARN(
    "`torch::nn::modules_ordered_dict` is deprecated. "
    "To construct a `Sequential` with named submodules, "
    "you can do `Sequential sequential({{\"m1\", MyModule(1)}, {\"m2\", MyModule(2)}})`");
  torch::OrderedDict<std::string, AnyModule> dict;
  for (auto named_module : named_modules) {
    dict.insert(named_module.name(), std::move(named_module.module()));
  }
  return dict;
}

} // namespace nn
} // namespace torch
