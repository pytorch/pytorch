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

bool SchemaInfo::hasSideEffects() const {
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
        return side_effect_op == name();
      });
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
      [this](size_t index) {
        static const char* batch_norm_literal =
            "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor";
        static const char* instance_norm_literal =
            "aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor";
        if (torch::jit::getOperatorForLiteral(batch_norm_literal)->schema() ==
                *this &&
            (arguments()[index].name() == "running_mean" ||
             arguments()[index].name() == "running_var")) {
          return value_map_.count("training") &&
              value_map_.at("training").toBool();
        } else if (
            torch::jit::getOperatorForLiteral(instance_norm_literal)
                    ->schema() == *this &&
            (arguments()[index].name() == "running_mean" ||
             arguments()[index].name() == "running_var")) {
          return value_map_.count("use_input_stats") &&
              value_map_.at("use_input_stats").toBool();
        } else {
          return FunctionSchema::is_mutable(index);
        }
      });
}

bool SchemaInfo::is_mutable(c10::string_view name) {
  c10::optional<int> index = argumentIndexWithName(name);
  TORCH_INTERNAL_ASSERT(
      index != c10::nullopt, "Schema has no argument named ", name);

  return is_mutable(*index);
}

bool SchemaInfo::isNonDeterministic() const {
  if (torch::jit::getOperatorForLiteral(
          "aten::dropout(Tensor input, float p, bool train) -> Tensor")
              ->schema() == *this &&
      value_map_.count("train") && !value_map_.at("train").toBool()) {
    return false;
  }

  static const std::vector<const char*> nondeterministic_ops = {
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
  return std::any_of(
      nondeterministic_ops.begin(),
      nondeterministic_ops.end(),
      [this](const char* nondeterministic_op) {
        return torch::jit::getOperatorForLiteral(nondeterministic_op)
                   ->schema() == *this;
      });
}

bool SchemaInfo::areAliasing(
    const c10::SchemaArgument& lhs,
    const c10::SchemaArgument& rhs,
    bool check_additional) {
  bool basic_check = FunctionSchema::areAliasing(lhs, rhs, true);
  if (check_additional) {
    if (!updated_) {
      generateAliasMaps();
    }
    if (lhs.type == c10::input && rhs.type == c10::input) {
      return input_alias_map_[lhs.index].count(rhs.index) || basic_check;
    } else if (lhs.type == c10::output && rhs.type == c10::output) {
      for (size_t lhs_alias_input : output_alias_map_[lhs.index]) {
        for (size_t rhs_alias_input : output_alias_map_[rhs.index]) {
          if (lhs_alias_input == rhs_alias_input) {
            return true;
          }
        }
      }
      return basic_check;
    } else if (lhs.type == c10::output) {
      return output_alias_map_[lhs.index].count(rhs.index) || basic_check;
    } else {
      return output_alias_map_[rhs.index].count(lhs.index) || basic_check;
    }
  }
  return basic_check;
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
  output_alias_map_ = std::vector<std::unordered_set<size_t>>(
      returns().size(), std::unordered_set<size_t>());
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
  for (size_t i = 0; i < arguments().size(); i++) {
    for (size_t j = 0; j < returns().size(); j++) {
      if (FunctionSchema::areAliasing(
              {c10::input, i}, {c10::output, j}, true)) {
        output_alias_map_[j].insert(
            input_alias_map_[i].begin(), input_alias_map_[i].end());
      }
    }
  }
}

} // namespace utils
} // namespace torch
