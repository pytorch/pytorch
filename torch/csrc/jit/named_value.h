#pragma once
#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/constants.h>
#include <torch/csrc/jit/source_range.h>
#include <torch/csrc/utils/variadic.h>

namespace torch {
namespace jit {

struct Value;

/**
 * A value with optional extra name and location information. Used during
 * schema matching to provide extra error information and resolve kwargs.
 */
struct NamedValue {
  NamedValue(const SourceRange& loc, const std::string& name, Value* value)
      : loc_(loc), name_(name), value_(value) {}
  NamedValue(const SourceRange& loc, Value* value) : loc_(loc), value_(value) {}

  /* implicit */ NamedValue(Value* value) : value_(value) {}
  NamedValue(const std::string& name, Value* value)
      : name_(name), value_(value) {}

  /* implicit */ NamedValue(IValue value)
      : value_(nullptr), ivalue_(std::move(value)) {}

  NamedValue(const std::string& name, IValue value)
      : name_(name), ivalue_(std::move(value)) {}

  template <
      typename T,
      typename = enable_if_t<
          (!std::is_same<decay_t<T>, NamedValue>::value &&
           !std::is_same<decay_t<T>, Value*>::value &&
           !std::is_same<decay_t<T>, IValue>::value)>>
  NamedValue(T&& t) : NamedValue(IValue(std::forward<T>(t))) {}

  template <
      typename T,
      typename = enable_if_t<
          (!std::is_same<decay_t<T>, Value*>::value &&
           !std::is_same<decay_t<T>, IValue>::value)>>
  NamedValue(const std::string& name, T&& t)
      : NamedValue(name, IValue(std::forward<T>(t))) {}

  SourceRange locOr(const SourceRange& backup_location) const {
    if (!loc_)
      return backup_location;
    return loc();
  }

  // note: this will insert a constant node into the graph at the current
  // insert point if this NamedValue is actually a constant
  Value* value(Graph& g) const {
    if (!value_)
      return insertConstant(
          g, ivalue_); // use insertConstant to remove need to include ir.h here
    return value_;
  }

  const std::string& name() const {
    AT_ASSERT(name_);
    return *name_;
  }

  const SourceRange& loc() const {
    AT_ASSERT(loc_);
    return *loc_;
  }

 private:
  c10::optional<SourceRange> loc_;
  c10::optional<std::string> name_;
  Value* value_{nullptr};
  // only valid if value_ == nullptr;
  IValue ivalue_;
};

} // namespace jit
} // namespace torch
