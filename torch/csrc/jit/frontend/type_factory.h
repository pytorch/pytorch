#pragma once

#include <ATen/core/dynamic_type.h>
#include <ATen/core/jit_type.h>

namespace torch {
namespace jit {

template <typename T>
struct TypeFactory {};

template <>
struct TORCH_API TypeFactory<c10::DynamicType> {
  template <typename T, typename... Args>
  static c10::TypePtr create(Args&&... args) {
    return std::make_shared<c10::DynamicType>(
        c10::DynamicTypeTrait<T>::tagValue,
        c10::DynamicType::Arguments(
            c10::ArrayRef<c10::TypePtr>({std::forward<Args>(args)...})));
  }
  static c10::TypePtr createTuple(std::vector<c10::TypePtr> types) {
    return std::make_shared<c10::DynamicType>(
        c10::DynamicType::Tag::Tuple, c10::DynamicType::Arguments(types));
  }
  static c10::TypePtr createNamedTuple(
      const std::string& name,
      const std::vector<c10::string_view>& fields,
      const std::vector<c10::TypePtr>& types) {
    return std::make_shared<c10::DynamicType>(
        c10::DynamicType::Tag::Tuple,
        name,
        c10::DynamicType::Arguments(fields, types));
  }
  static const std::unordered_map<std::string, c10::TypePtr>& basePythonTypes();
};

template <>
struct TORCH_API TypeFactory<c10::Type> {
  template <typename T, typename... Args>
  static c10::TypePtr create(Args&&... args) {
    return T::create(std::forward<Args>(args)...);
  }
  static c10::TypePtr createTuple(std::vector<c10::TypePtr> types) {
    return c10::TupleType::create(std::move(types));
  }
  static c10::TypePtr createNamedTuple(
      const std::string& name,
      const std::vector<c10::string_view>& fields,
      const std::vector<c10::TypePtr>& types) {
    return c10::TupleType::createNamed(name, fields, types);
  }
  static const std::unordered_map<std::string, c10::TypePtr>& basePythonTypes();
};

} // namespace jit
} // namespace torch
