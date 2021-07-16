#pragma once
#include <ATen/ATen.h>
#include <string>
#include <vector>

#include <ATen/core/interned_strings.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace c10 {
struct Type;
using TypePtr = std::shared_ptr<Type>;
} // namespace c10

namespace torch {
namespace jit {

using ::c10::Symbol;

constexpr int max_tensor_display_size = 10;

enum class AttributeKind {
  f,
  fs,
  c,
  cs,
  i,
  is,
  s,
  ss,
  t,
  ts,
  g,
  gs,
  ty,
  tys,
  ival
};
static inline const char* toString(AttributeKind kind) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  static const char* names[] = {
      "f",
      "c",
      "cs",
      "fs",
      "i",
      "is",
      "s",
      "ss",
      "t",
      "ts",
      "g",
      "gs",
      "ty",
      "tys",
      "ival"};
  AT_ASSERT(size_t(kind) < sizeof(names) / sizeof(*names));
  return names[int(kind)];
}

struct AttributeValue {
  AttributeValue(Symbol name) : name(name) {}
  using Ptr = std::unique_ptr<AttributeValue>;
  Symbol name;
  virtual AttributeKind kind() const = 0;
  virtual Ptr clone() const = 0;
  virtual ~AttributeValue() = default;
};

template <typename T, AttributeKind Kind>
struct ScalarAttributeValue : public AttributeValue {
  using ConstructorType = T;
  using ValueType = T;
  ScalarAttributeValue(Symbol name, ConstructorType value_)
      : AttributeValue(name), value_(std::move(value_)) {}
  ValueType& value() {
    return value_;
  }
  Ptr clone() const override {
    return Ptr(new ScalarAttributeValue(name, value_));
  }
  AttributeKind kind() const override {
    return Kind;
  }

 private:
  ValueType value_;
};

template <typename T, AttributeKind Kind>
struct VectorAttributeValue : public AttributeValue {
  using ConstructorType = std::vector<T>;
  using ValueType = std::vector<T>;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  VectorAttributeValue(Symbol name, ConstructorType value_)
      : AttributeValue(name), value_(std::move(value_)) {}
  ValueType& value() {
    return value_;
  }
  AttributeKind kind() const override {
    return Kind;
  }
  std::unique_ptr<AttributeValue> clone() const override {
    auto copy = value_;
    return Ptr(new VectorAttributeValue(name, std::move(copy)));
  }

 private:
  ValueType value_;
};

using ComplexAttr =
    ScalarAttributeValue<c10::complex<double>, AttributeKind::c>;
using ComplexValsAttr =
    VectorAttributeValue<c10::complex<double>, AttributeKind::cs>;
using FloatAttr = ScalarAttributeValue<double, AttributeKind::f>;
using FloatsAttr = VectorAttributeValue<double, AttributeKind::fs>;
using IntAttr = ScalarAttributeValue<int64_t, AttributeKind::i>;
using IntsAttr = VectorAttributeValue<int64_t, AttributeKind::is>;
using StringAttr = ScalarAttributeValue<std::string, AttributeKind::s>;
using StringsAttr = VectorAttributeValue<std::string, AttributeKind::ss>;
using TensorAttr = ScalarAttributeValue<at::Tensor, AttributeKind::t>;
using TensorsAttr = VectorAttributeValue<at::Tensor, AttributeKind::ts>;
using TypeAttr = ScalarAttributeValue<c10::TypePtr, AttributeKind::ty>;
using TypesAttr = VectorAttributeValue<c10::TypePtr, AttributeKind::tys>;
using IValueAttr = ScalarAttributeValue<at::IValue, AttributeKind::ival>;

struct Graph;

// We special case Graph attributes like this because we want to ensure that
// Graph::copy() is called when we clone() these attributes.
struct TORCH_API GraphAttr : public AttributeValue {
  using ConstructorType = std::shared_ptr<Graph>;
  using ValueType = std::shared_ptr<Graph>;
  GraphAttr(Symbol name, ConstructorType value_)
      : AttributeValue(name), value_(std::move(value_)) {}
  ValueType& value() {
    return value_;
  }
  Ptr clone() const override;
  AttributeKind kind() const override {
    return AttributeKind::g;
  }

 private:
  std::shared_ptr<Graph> value_;
};

struct TORCH_API GraphsAttr : public AttributeValue {
  using ConstructorType = std::vector<std::shared_ptr<Graph>>;
  using ValueType = std::vector<std::shared_ptr<Graph>>;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  GraphsAttr(Symbol name, ConstructorType value_)
      : AttributeValue(name), value_(std::move(value_)) {}
  ValueType& value() {
    return value_;
  }
  AttributeKind kind() const override {
    return AttributeKind::gs;
  }
  std::unique_ptr<AttributeValue> clone() const override;

 private:
  ValueType value_;
};

struct IRAttributeError : public std::exception {
  IRAttributeError(Symbol name, bool defined) {
    std::stringstream ss;
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (!defined) {
      ss << "required keyword attribute '" << name.toUnqualString()
         << "' is undefined";
    } else {
      ss << "required keyword attribute '" << name.toUnqualString()
         << "' has the wrong type";
    }
    msg = ss.str();
  }
  const char* what() const noexcept override {
    return msg.c_str();
  }

 private:
  std::string msg;
};
} // namespace jit
} // namespace torch
