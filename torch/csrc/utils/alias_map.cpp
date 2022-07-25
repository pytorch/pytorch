#include <torch/csrc/utils/alias_map.h>

namespace torch {
namespace utils {
AliasMap::AliasMap(const c10::FunctionSchema& schema) {
  generateMaps(schema);
}

void AliasMap::generateMaps(const c10::FunctionSchema& schema) {
  std::unordered_map<at::Symbol, size_t> alias_set_to_inputs;
  alias_map_ = std::vector<std::unordered_set<size_t>>(
      schema.returns().size(), std::unordered_set<size_t>());
  for (const auto i : c10::irange(schema.arguments().size())) {
    if (schema.is_aliasing({c10::SchemaArgType::input, i})) {
      const at::AliasInfo* alias_info = schema.arguments()[i].alias_info();
      const auto& set = alias_info->beforeSet();
      // skip if we've already bound this alias
      if (!alias_set_to_inputs.count(set)) {
        alias_set_to_inputs.insert({set, i});
      }
      aliasing_inputs_.insert(i);
      if (schema.is_mutable({c10::SchemaArgType::input, i})) {
        mutating_inputs_.insert(i);
      } else {
        const_aliasing_inputs_.insert(i);
      }
    }
  }

  for (const auto i : c10::irange(schema.returns().size())) {
    if (schema.is_aliasing({c10::SchemaArgType::output, i})) {
      const at::AliasInfo* alias_info = schema.returns()[i].alias_info();
      bool inputs_has_alias = false;
      for (const auto& set : alias_info->beforeSets()) {
        if (alias_set_to_inputs.count(set)) {
          inputs_has_alias = true;
          alias_map_[i].insert(alias_set_to_inputs[set]);
        }
      }
      if (inputs_has_alias) {
        aliasing_outputs_.insert(i);
        if (schema.is_mutable({c10::SchemaArgType::output, i})) {
          mutating_outputs_.insert(i);
        } else {
          const_aliasing_outputs_.insert(i);
        }
      }
    }
  }
}

} // namespace utils
} // namespace torch
