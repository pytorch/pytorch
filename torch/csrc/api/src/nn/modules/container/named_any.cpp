#include <torch/nn/modules/container/named_any.h>

namespace torch {
namespace nn {

torch::OrderedDict<std::string, AnyModule> modules_ordered_dict(
  std::initializer_list<NamedAnyModule> named_modules) {
  torch::OrderedDict<std::string, AnyModule> dict;
  for (auto named_module : named_modules) {
    dict.insert(named_module.name(), std::move(named_module.module()));
  }
  return dict;
}

} // namespace nn
} // namespace torch
