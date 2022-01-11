#pragma once

#include <ATen/core/dynamic_type.h>
#include <ATen/core/jit_type.h>

namespace c10 {

struct TORCH_API DynamicTypeFactory {
  template <typename T, typename... Args>
  static c10::TypePtr create(Args&&... args) {
    return std::make_shared<c10::DynamicType>(
        c10::DynamicTypeTrait<T>::tagValue(),
        c10::DynamicType::Arguments(
            c10::ArrayRef<c10::TypePtr>({std::forward<Args>(args)...})));
  }
};

struct TORCH_API DefaultTypeFactory {
  template <typename T, typename... Args>
  static c10::TypePtr create(Args&&... args) {
    return T::create(std::forward<Args>(args)...);
  }
};

} // namespace c10
