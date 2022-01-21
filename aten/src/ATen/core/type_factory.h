#pragma once

#include <type_traits>

#include <ATen/core/dynamic_type.h>
#include <ATen/core/jit_type.h>
#include <c10/macros/Macros.h>

namespace c10 {

struct TORCH_API DynamicTypeFactory {
  template <typename T, typename... Args>
  static c10::DynamicTypePtr create(TypePtr ty, Args&&... args) {
    return std::make_shared<c10::DynamicType>(
        c10::DynamicTypeTrait<T>::tagValue(),
        c10::DynamicType::Arguments(c10::ArrayRef<c10::TypePtr>(
            {std::move(ty), std::forward<Args>(args)...})));
  }
  template <typename T>
  static c10::DynamicTypePtr create(std::vector<c10::TypePtr> types) {
    return std::make_shared<c10::DynamicType>(
        c10::DynamicTypeTrait<T>::tagValue(),
        c10::DynamicType::Arguments(types));
  }
  static c10::DynamicTypePtr createNamedTuple(
      const std::string& name,
      const std::vector<c10::string_view>& fields,
      const std::vector<c10::TypePtr>& types) {
    return std::make_shared<c10::DynamicType>(
        c10::DynamicType::Tag::Tuple,
        name,
        c10::DynamicType::Arguments(fields, types));
  }
  template <typename T>
  C10_ERASE static c10::DynamicTypePtr createNamed(const std::string& name) {
    return std::make_shared<c10::DynamicType>(
        c10::DynamicTypeTrait<T>::tagValue(),
        name,
        c10::DynamicType::Arguments{});
  }
  template <typename T>
  C10_ERASE static c10::DynamicTypePtr get() {
    return DynamicTypeTrait<T>::getBaseType();
  }
  static const std::unordered_map<std::string, c10::TypePtr>& basePythonTypes();
};

// Helper functions for constructing DynamicTypes inline.
template <
    typename T,
    std::enable_if_t<DynamicTypeTrait<T>::isBaseType, int> = 0>
C10_ERASE DynamicTypePtr dynT() {
  return DynamicTypeFactory::get<T>();
}

template <
    typename T,
    typename... Args,
    std::enable_if_t<!DynamicTypeTrait<T>::isBaseType, int> = 0>
C10_ERASE DynamicTypePtr dynT(Args&&... args) {
  return DynamicTypeFactory::create<T>(std::forward<Args>(args)...);
}

struct TORCH_API DefaultTypeFactory {
  template <typename T, typename... Args>
  static c10::TypePtr create(TypePtr ty, Args&&... args) {
    return T::create(std::move(ty), std::forward<Args>(args)...);
  }
  template <typename T>
  static c10::TypePtr create(std::vector<c10::TypePtr> types) {
    return T::create(std::move(types));
  }
  static c10::TypePtr createNamedTuple(
      const std::string& name,
      const std::vector<c10::string_view>& fields,
      const std::vector<c10::TypePtr>& types) {
    return c10::TupleType::createNamed(name, fields, types);
  }
  template <typename T>
  C10_ERASE static c10::TypePtr createNamed(const std::string& name) {
    return T::create(name);
  }
  static const std::unordered_map<std::string, c10::TypePtr>& basePythonTypes();
  template <typename T>
  C10_ERASE static c10::TypePtr get() {
    return T::get();
  }
};

using TypeFactory =
#ifdef C10_MOBILE
    DynamicTypeFactory
#else
    DefaultTypeFactory
#endif
    ;

} // namespace c10
