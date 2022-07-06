#include <torch/csrc/utils/schema_info.h>

namespace torch {
namespace utils {
void SchemaInfo::addArgumentValue(
    const std::string& name,
    const at::IValue& value) {
  c10::optional<int> index = argumentIndexWithName(name);
  TORCH_INTERNAL_ASSERT(
      index != c10::nullopt, "Schema has no argument named ", name);
  value_map_[name] = flattenZeroDimIValue(value);
  updated_ = false;
}

void SchemaInfo::addArgumentValues(
    const std::vector<c10::optional<at::IValue>>& value_list) {
  for (size_t i = 0; i < value_list.size(); i++) {
    if (i < arguments().size() && value_list[i] != c10::nullopt) {
      value_map_[arguments()[i].name()] =
          flattenZeroDimIValue(*(value_list[i]));
      updated_ = false;
    }
  }
}

void SchemaInfo::addArgumentValues(
    const std::unordered_map<std::string, at::IValue>& values) {
  for (const auto& key_pair : values) {
    addArgumentValue(key_pair.first, key_pair.second);
  }
}

bool SchemaInfo::is_mutable() {
  for (size_t i = 0; i < arguments().size(); i++) {
    if (is_mutable(i)) {
      return true;
    }
  }
  return false;
}

bool SchemaInfo::is_mutable(size_t index) {
  TORCH_INTERNAL_ASSERT(
      index < arguments().size(), "Invalid index for schema.");
  if (!updated_) {
    generateAliasMaps();
  }
  return std::any_of(
      input_alias_map_[index].begin(),
      input_alias_map_[index].end(),
      [this](size_t index) { return FunctionSchema::is_mutable(index); });
}

bool SchemaInfo::is_mutable(c10::string_view name) {
  c10::optional<int> index = argumentIndexWithName(name);
  TORCH_INTERNAL_ASSERT(
      index != c10::nullopt, "Schema has no argument named ", name);

  return is_mutable(*index);
}

at::IValue SchemaInfo::flattenZeroDimIValue(const at::IValue& value) const {
  if (value.isList()) {
    c10::List<at::IValue> value_list = value.toList();
    if (value_list.size() == 1) {
      return value_list[0];
    }
  }
  return value;
}

void SchemaInfo::generateAliasMaps() {
  updated_ = true;
  input_alias_map_ = std::vector<std::unordered_set<size_t>>(
      arguments().size(), std::unordered_set<size_t>());
  for (size_t i = 0; i < arguments().size(); i++) {
    for (size_t j = i; j < arguments().size(); j++) {
      if (i == j) {
        input_alias_map_[i].insert(i);
      } else if (
          value_map_.count(arguments()[i].name()) &&
          value_map_.count(arguments()[j].name())) {
        if (value_map_[arguments()[i].name()].isAliasOf(
                value_map_[arguments()[j].name()])) {
          input_alias_map_[i].insert(j);
          input_alias_map_[j].insert(i);
        }
      }
    }
  }
}

} // namespace utils
} // namespace torch
