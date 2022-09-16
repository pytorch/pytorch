#pragma once

#include <type_traits>
#include <unordered_map>

#include <ATen/core/dynamic_type.h>
#include <ATen/core/jit_type_base.h>
#include <c10/macros/Macros.h>

namespace c10 {

template <typename T>
struct TORCH_API TypeFactoryBase {};

template <>
struct TORCH_API TypeFactoryBase<c10::DynamicType> {
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

using DynamicTypeFactory = TypeFactoryBase<c10::DynamicType>;

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

template <>
struct TORCH_API TypeFactoryBase<c10::Type> {
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
      const std::vector<c10::TypePtr>& types);
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

using DefaultTypeFactory = TypeFactoryBase<c10::Type>;

using PlatformType =
#ifdef C10_MOBILE
    c10::DynamicType
#else
    c10::Type
#endif
    ;

using TypeFactory = TypeFactoryBase<PlatformType>;

} // namespace c10
