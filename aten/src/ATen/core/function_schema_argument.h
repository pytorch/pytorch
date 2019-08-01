#pragma once

#include <string>

namespace c10 {

struct Argument {
  Argument(
      std::string name = "",
      TypePtr type = nullptr,
      c10::optional<int32_t> N = c10::nullopt,
      c10::optional<IValue> default_value = c10::nullopt,
      bool kwarg_only = false,
      c10::optional<AliasInfo> alias_info = c10::nullopt,
      bool is_inferred_type = false)
      : name_(std::move(name)),
        type_(type ? type : TensorType::get()),
        N_(std::move(N)),
        default_value_(std::move(default_value)),
        kwarg_only_(kwarg_only),
        alias_info_(std::move(alias_info)),
        is_inferred_type_(is_inferred_type) {
    if (default_value_ && default_value_->isTensor()) {
      auto t = default_value_->toTensor();
      AT_ASSERT(!t.defined() || t.is_variable());
    }
  }
  const std::string& name() const {
    return name_;
  }
  TypePtr type() const {
    return type_;
  }
  c10::optional<int32_t> N() const {
    return N_;
  }
  const c10::optional<IValue>& default_value() const {
    return default_value_;
  }
  bool kwarg_only() const {
    return kwarg_only_;
  }
  const c10::optional<AliasInfo>& alias_info() const {
    return alias_info_;
  }
  bool is_inferred_type() const {
    return is_inferred_type_;
  }
  std::string formatTypeMismatchMsg(const std::string& actual_type) const {
    std::string inferred_type_hint;
    if (is_inferred_type()) {
      inferred_type_hint = c10::str(
          "Inferred '",
          name(),
          "' to be of type 'Tensor' ",
          "because it was not annotated with an explicit type.\n");
    }
    return c10::str(
        "Expected a value of type '",
        type()->python_str(),
        "' for argument '",
        name(),
        "' but instead found type '",
        actual_type,
        "'.\n",
        inferred_type_hint);
  }

  Argument cloneWithType(TypePtr new_type) const {
    return Argument(name_, new_type, N_, default_value_, kwarg_only_, alias_info_);
  }

private:
  std::string name_;
  TypePtr type_;
  // for list types, an optional statically known length for the list
  // e.g. for int[3]: type = ListType::ofInts(), N = 3
  // If present, this will allow scalars to be broadcast to this length to
  // become a list.
  c10::optional<int32_t> N_;

  c10::optional<IValue> default_value_;
  // is this only specifyable as a keyword argument?
  bool kwarg_only_;
  c10::optional<AliasInfo> alias_info_;
  bool is_inferred_type_;
};

namespace detail {
inline bool defaultValueEquals_(const c10::optional<IValue>& lhs, const c10::optional<IValue>& rhs) {
  if (lhs.has_value()) {
    return rhs.has_value() && impl::shallowEquals(*lhs, *rhs);
  } else {
    return !rhs.has_value();
  }
}
}

inline bool operator==(const Argument& lhs, const Argument& rhs) {
  return lhs.name() == rhs.name()
          && *lhs.type() == *rhs.type()
          && lhs.N() == rhs.N()
          && detail::defaultValueEquals_(lhs.default_value(), rhs.default_value())
          && lhs.kwarg_only() == rhs.kwarg_only()
          && lhs.alias_info() == rhs.alias_info();
}

}
