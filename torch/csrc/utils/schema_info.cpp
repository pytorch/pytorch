#include <torch/csrc/utils/schema_info.h>

namespace torch {
namespace utils {
void SchemaInfo::addArgumentValue(
    const std::string& name,
    const at::IValue& value) {
  c10::optional<int> index = schema_.argumentIndexWithName(name);
  TORCH_INTERNAL_ASSERT(
      index != c10::nullopt, "Schema has no argument named ", name);
  value_map_[name] = value;
  alias_maps_current_ = false;
}

void SchemaInfo::addArgumentValues(
    const std::vector<c10::optional<at::IValue>>& value_list) {
  TORCH_INTERNAL_ASSERT(
      value_list.size() <= schema_.arguments().size(),
      "Schema does not have enough arguments for value list");

  for (size_t i = 0; i < value_list.size(); i++) {
    if (value_list[i] != c10::nullopt) {
      value_map_[schema_.arguments()[i].name()] = *(value_list[i]);
      alias_maps_current_ = false;
    }
  }
}

void SchemaInfo::addArgumentValues(
    const std::unordered_map<std::string, at::IValue>& values) {
  for (const auto& key_pair : values) {
    addArgumentValue(key_pair.first, key_pair.second);
  }
}

bool SchemaInfo::has_side_effects() const {
  static const std::vector<std::string> side_effects_ops = {
      "aten::warn",
      "aten::save",
      "aten::manual_seed",
      "aten::wait",
      "cuda::set_stream",
      "cuda::_set_device",
      "cuda::_current_device",
      "cuda::synchronize",
  };
  return std::any_of(
      side_effects_ops.begin(),
      side_effects_ops.end(),
      [this](const std::string& side_effect_op) {
        return side_effect_op == schema_.name();
      });
}

bool SchemaInfo::is_mutable() {
  for (size_t i = 0; i < schema_.arguments().size(); i++) {
    if (is_mutable(i)) {
      return true;
    }
  }
  return false;
}

bool SchemaInfo::is_mutable(size_t index) {
  TORCH_INTERNAL_ASSERT(
      index < schema_.arguments().size(), "Invalid index for schema.");
  if (!alias_maps_current_) {
    generateAliasMaps();
  }
  return std::any_of(
      input_alias_map_[index].begin(),
      input_alias_map_[index].end(),
      [this](size_t index) { return this->schema_.is_mutable(index); });
}

bool SchemaInfo::is_mutable(c10::string_view name) {
  c10::optional<int> index = schema_.argumentIndexWithName(name);
  TORCH_INTERNAL_ASSERT(
      index != c10::nullopt, "Schema has no argument named ", name);

  return is_mutable(*index);
}

bool SchemaInfo::is_nondeterministic() const {
  static const std::vector<c10::FunctionSchema> nondeterministic_ops =
      getNonDeterministicOps();

  return std::any_of(
      nondeterministic_ops.begin(),
      nondeterministic_ops.end(),
      [this](const c10 ::FunctionSchema& nondeterministic_op) {
        return nondeterministic_op == this->schema_;
      });
}

bool SchemaInfo::may_alias(
    const c10::SchemaArgument& lhs,
    const c10::SchemaArgument& rhs) {
  bool basic_check = schema_.may_alias(lhs, rhs);
  if (basic_check) {
    return true;
  }
  if (!alias_maps_current_) {
    generateAliasMaps();
  }
  if (lhs.type == c10::SchemaArgType::input &&
      rhs.type == c10::SchemaArgType::input) {
    return input_alias_map_[lhs.index].count(rhs.index);
  } else if (
      lhs.type == c10::SchemaArgType::output &&
      rhs.type == c10::SchemaArgType::output) {
    for (size_t lhs_alias_input : output_alias_map_[lhs.index]) {
      if (output_alias_map_[rhs.index].count(lhs_alias_input)) {
        return true;
      }
    }
    return false;
  } else if (lhs.type == c10::SchemaArgType::output) {
    return output_alias_map_[lhs.index].count(rhs.index);
  } else {
    return output_alias_map_[rhs.index].count(lhs.index);
  }
}

std::vector<c10::FunctionSchema> SchemaInfo::getNonDeterministicOps() {
  // This list of nondeterministic ops is copied from JIT ir.cpp.
  static const std::vector<std::string> nondeterministic_op_strings = {
      "aten::dropout(Tensor input, float p, bool train) -> Tensor",
      "aten::_fused_dropout(Tensor self, float p, Generator? generator) -> (Tensor, Tensor)",
      "aten::_standard_gamma(Tensor self, Generator? generator) -> Tensor",
      "aten::bernoulli(Tensor self, *, Generator? generator) -> Tensor",
      "aten::bernoulli(Tensor self, float p, *, Generator? generator) -> Tensor",
      "aten::multinomial(Tensor self, int num_samples, bool replacement, *, Generator? generator) -> Tensor",
      "aten::native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)",
      "aten::normal(Tensor mean, Tensor std, *, Generator? generator) -> Tensor",
      "aten::normal(float mean, Tensor std, *, Generator? generator) -> Tensor",
      "aten::normal(Tensor mean, float std, *, Generator? generator) -> Tensor",
      "aten::poisson(Tensor self, Generator? generator) -> Tensor",
      "aten::binomial(Tensor count, Tensor prob, Generator? generator=None) -> Tensor",
      "aten::rrelu(Tensor self, Scalar lower, Scalar upper, bool training, Generator? generator) -> Tensor",
      "aten::rrelu_with_noise(Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, Generator? generator) -> Tensor",
      "aten::rand(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
      "aten::rand_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      "aten::randint(int high, int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
      "aten::randint(int low, int high, int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
      "aten::randint_like(Tensor self, int high, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      "aten::randint_like(Tensor self, int low, int high, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      "aten::randn(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
      "aten::randn_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      "aten::randperm(int n, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor"};

  std::vector<c10::FunctionSchema> nondeterministic_ops;
  nondeterministic_ops.reserve(nondeterministic_op_strings.size());
  for (const std::string& signature : nondeterministic_op_strings) {
    nondeterministic_ops.push_back(torch::jit::parseSchema(signature));
  }

  return nondeterministic_ops;
}

void SchemaInfo::generateAliasMaps() {
  alias_maps_current_ = true;
  input_alias_map_ = std::vector<std::unordered_set<size_t>>(
      schema_.arguments().size(), std::unordered_set<size_t>());
  output_alias_map_ = std::vector<std::unordered_set<size_t>>(
      schema_.returns().size(), std::unordered_set<size_t>());
  for (size_t i = 0; i < schema_.arguments().size(); i++) {
    for (size_t j = i; j < schema_.arguments().size(); j++) {
      if (i == j) {
        input_alias_map_[i].insert(i);
      } else if (
          value_map_.count(schema_.arguments()[i].name()) &&
          value_map_.count(schema_.arguments()[j].name())) {
        if (value_map_[schema_.arguments()[i].name()].isAliasOf(
                value_map_[schema_.arguments()[j].name()])) {
          input_alias_map_[i].insert(j);
          input_alias_map_[j].insert(i);
        }
      }
    }
  }
  for (size_t i = 0; i < schema_.arguments().size(); i++) {
    for (size_t j = 0; j < schema_.returns().size(); j++) {
      if (schema_.may_alias(
              {c10::SchemaArgType::input, i},
              {c10::SchemaArgType::output, j})) {
        output_alias_map_[j].insert(
            input_alias_map_[i].begin(), input_alias_map_[i].end());
      }
    }
  }
}

} // namespace utils
} // namespace torch
