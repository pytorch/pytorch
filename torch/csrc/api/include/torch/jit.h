#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/script/module.h>

#include <string>
#include <memory>

namespace torch {
namespace jit {

/// Compiles script code into an executable graph.
///
/// Takes a string containing functions in script syntax and compiles them into
/// a module (graph). The returned module provides a `run_method` function
/// that may be used to invoke the compiled functions.
///
/// For example:
/// \rst
/// .. code-block:: cpp
///
///   auto module = torch::jit::compile(R"JIT(
///     def relu_script(a, b):
///       return torch.relu(a + b)
///     def test_while(a, i):
///       while i < 10:
///         a += a
///         i += 1
///       return a
///   )JIT");
///   IValue output = module->run_method("relu_script", a, b);
/// \endrst
TORCH_API std::shared_ptr<script::CompilationUnit> compile(const std::string& source);

} // namespace jit
} // namespace torch


#include <torch/csrc/jit/script/module.h>
#include <torch/jit/type.h>

namespace torch {

// We need this since it's not possible to partially specialize on a type, but
// you can do a partial overload. We need this so we can have specializations
// for things like std::vector<T>
template <typename T>
struct type_container{};

struct Value;

 struct TORCH_API Value {
  template <typename T>
  Value(T&& value) : value_(std::forward<T>(value)) {}

  template <typename T>
  T to() const;

  template <typename T>
  bool is() const;

  const jit::IValue& ivalue() const {
    return value_;
  }

 protected:
  c10::IValue value_;
};


template <typename Elem>
std::vector<Elem> generic_to(
    const Value* value,
    type_container<std::vector<Elem>>) {
  if (std::is_same<Elem, int64_t>()) {
    // return value->ivalue().toIntList();
    return fmap(value->ivalue().toGenericListRef(), [](int64_t value){
      return value;
    });
  }
  return fmap(value->ivalue().toGenericListRef(), [](const c10::IValue& value){
    return value.to<Elem>();
  });
}

inline c10::optional<std::string> get_cpp_type(const c10::IValue& ivalue) {
  if (ivalue.isInt()) {
    return "int64_t";
  } else if (ivalue.isIntList()) {
      return "std::vector<int64_t>";
  }
  return c10::nullopt;
}

template <typename T>
T Value::to() const {
  if (!is<T>()) {
    auto cpp_type = get_cpp_type(ivalue());
    std::string message =
        "Could not unwrap " + ivalue().tagKind() + " as the requested type";
    if (cpp_type) {
      message += ", please use " + *cpp_type;
    }
    TORCH_CHECK(false, message);
  }
  return generic_to(this, type_container<T>{});
}

template <typename T>
bool Value::is() const {
  return generic_is(this, type_container<T>{});
}



template <typename T>
bool generic_is(const Value* value, type_container<T>) {
  return false;
}

} // namespace torch
