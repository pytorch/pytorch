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
};

template <>
struct TORCH_API TypeFactory<c10::Type> {
  template <typename T, typename... Args>
  static c10::TypePtr create(Args&&... args) {
    return T::create(std::forward<Args>(args)...);
  }
};

} // namespace jit
} // namespace torch
