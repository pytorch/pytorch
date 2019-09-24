#pragma once

#include <atomic>
#include <cassert>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef __GXX_RTTI
#include <typeinfo>
#endif

#include <exception>

#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Backtrace.h>
#include <c10/util/C++17.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <c10/util/IdWrapper.h>
#include <c10/util/Type.h>
#include <c10/util/TypeIndex.h>
#include <c10/util/qint32.h>
#include <c10/util/qint8.h>
#include <c10/util/quint8.h>

// TODO: This file is still in the caffe2 namespace, despite living
// in the ATen directory. Move to c10.

// Make at::Half a fundamental type.
namespace std {
template <>
struct is_fundamental<at::Half> : std::true_type {};
} // namespace std

namespace caffe2 {

/**
 * A TypeIdentifier is a unique id for a given C++ type.
 * This is for example used to store the dtype of tensors.
 */
class C10_API TypeIdentifier final
    : public at::IdWrapper<TypeIdentifier, c10::util::type_index> {
 public:
  friend std::ostream& operator<<(std::ostream& stream, TypeIdentifier typeId);
  friend constexpr bool operator<(TypeIdentifier lhs, TypeIdentifier rhs);

  /**
   * Returns the unique id for the given type T. The id is unique for the type T
   * in the sense that for any two different types, their ids are different; for
   * the same type T, the id remains the same over different calls of the
   * function. However, this is not guaranteed over different runs, as the id
   * is generated during run-time. Do NOT serialize the id for storage.
   */
  template <typename T>
  static C10_HOST_CONSTEXPR TypeIdentifier Get() noexcept {
    return TypeIdentifier(c10::util::get_type_index<T>());
  }

  static constexpr TypeIdentifier uninitialized() {
    return TypeIdentifier(c10::util::type_index{0});
  }

 private:
  constexpr explicit TypeIdentifier(c10::util::type_index id) : IdWrapper(id) {}
  friend class TypeMeta;
};

// Allow usage in std::map / std::set
// TODO Disallow this and rather use std::unordered_map/set everywhere
inline constexpr bool operator<(TypeIdentifier lhs, TypeIdentifier rhs) {
  return lhs.underlyingId() < rhs.underlyingId();
}

inline std::ostream& operator<<(
    std::ostream& stream,
    caffe2::TypeIdentifier typeId) {
  return stream << typeId.underlyingId();
}

} // namespace caffe2

namespace at {
using DataType = caffe2::TypeIdentifier;
}

C10_DEFINE_HASH_FOR_IDWRAPPER(caffe2::TypeIdentifier)

namespace caffe2 {

namespace detail {

// This struct holds the actual type information. There will be
// one allocated per type. TypeMeta objects will then point to the struct
// instance for the type they're configured for.
struct TypeMetaData final {
  using New = void*();
  using PlacementNew = void(void*, size_t);
  using Copy = void(const void*, void*, size_t);
  using PlacementDelete = void(void*, size_t);
  using Delete = void(void*);

  TypeMetaData() = delete;
  constexpr TypeMetaData(
      size_t itemsize,
      New* newFn,
      PlacementNew* placementNew,
      Copy* copy,
      PlacementDelete* placementDelete,
      Delete* deleteFn,
      TypeIdentifier id,
      c10::string_view name) noexcept
      : itemsize_(itemsize),
        new_(newFn),
        placementNew_(placementNew),
        copy_(copy),
        placementDelete_(placementDelete),
        delete_(deleteFn),
        id_(id),
        name_(name) {}

  size_t itemsize_;
  New* new_;
  PlacementNew* placementNew_;
  Copy* copy_;
  PlacementDelete* placementDelete_;
  Delete* delete_;
  TypeIdentifier id_;
  c10::string_view name_;
};

// Mechanism for throwing errors which can't be prevented at compile time
// due to type erasure. E.g. somebody calling TypeMeta::copy() for
// non-copyable type. Right now just throws exception but is implemented
// in .cpp to manage dependencies
[[noreturn]] C10_API void _ThrowRuntimeTypeLogicError(const std::string& msg);

/**
 * Placement new function for the type.
 */
template <typename T>
inline void _PlacementNew(void* ptr, size_t n) {
  T* typed_ptr = static_cast<T*>(ptr);
  for (size_t i = 0; i < n; ++i) {
    new (typed_ptr + i) T;
  }
}

template <typename T>
inline void _PlacementNewNotDefault(void* /*ptr*/, size_t /*n*/) {
  _ThrowRuntimeTypeLogicError(
      "Type " + std::string(c10::util::get_fully_qualified_type_name<T>()) +
      " is not default-constructible.");
}

template <
    typename T,
    c10::guts::enable_if_t<std::is_default_constructible<T>::value>* = nullptr>
inline constexpr TypeMetaData::PlacementNew* _PickPlacementNew() {
  return (std::is_fundamental<T>::value || std::is_pointer<T>::value)
      ? nullptr
      : &_PlacementNew<T>;
}

template <
    typename T,
    c10::guts::enable_if_t<!std::is_default_constructible<T>::value>* = nullptr>
inline constexpr TypeMetaData::PlacementNew* _PickPlacementNew() {
  static_assert(
      !std::is_fundamental<T>::value && !std::is_pointer<T>::value,
      "this should have picked the other SFINAE case");
  return &_PlacementNewNotDefault<T>;
}

template <typename T>
inline void* _New() {
  return new T;
}

template <typename T>
inline void* _NewNotDefault() {
  _ThrowRuntimeTypeLogicError(
      "Type " + std::string(c10::util::get_fully_qualified_type_name<T>()) +
      " is not default-constructible.");
}

template <
    typename T,
    c10::guts::enable_if_t<std::is_default_constructible<T>::value>* = nullptr>
inline constexpr TypeMetaData::New* _PickNew() {
  return &_New<T>;
}

template <
    typename T,
    c10::guts::enable_if_t<!std::is_default_constructible<T>::value>* = nullptr>
inline constexpr TypeMetaData::New* _PickNew() {
  return &_NewNotDefault<T>;
}

/**
 * Typed copy function for classes.
 */
template <typename T>
inline void _Copy(const void* src, void* dst, size_t n) {
  const T* typed_src = static_cast<const T*>(src);
  T* typed_dst = static_cast<T*>(dst);
  for (size_t i = 0; i < n; ++i) {
    typed_dst[i] = typed_src[i];
  }
}

/**
 * A placeholder function for types that do not allow assignment.
 */
template <typename T>
inline void _CopyNotAllowed(const void* /*src*/, void* /*dst*/, size_t /*n*/) {
  _ThrowRuntimeTypeLogicError(
      "Type " + std::string(c10::util::get_fully_qualified_type_name<T>()) +
      " does not allow assignment.");
}

template <
    typename T,
    c10::guts::enable_if_t<std::is_copy_assignable<T>::value>* = nullptr>
inline constexpr TypeMetaData::Copy* _PickCopy() {
  return (std::is_fundamental<T>::value || std::is_pointer<T>::value)
      ? nullptr
      : &_Copy<T>;
}

template <
    typename T,
    c10::guts::enable_if_t<!std::is_copy_assignable<T>::value>* = nullptr>
inline constexpr TypeMetaData::Copy* _PickCopy() {
  static_assert(
      !std::is_fundamental<T>::value && !std::is_pointer<T>::value,
      "this should have picked the other SFINAE case");
  return &_CopyNotAllowed<T>;
}

/**
 * Destructor for non-fundamental types.
 */
template <typename T>
inline void _PlacementDelete(void* ptr, size_t n) {
  T* typed_ptr = static_cast<T*>(ptr);
  for (size_t i = 0; i < n; ++i) {
    typed_ptr[i].~T();
  }
}

template <typename T>
inline constexpr TypeMetaData::PlacementDelete* _PickPlacementDelete() {
  return (std::is_fundamental<T>::value || std::is_pointer<T>::value)
      ? nullptr
      : &_PlacementDelete<T>;
}

template <typename T>
inline void _Delete(void* ptr) {
  T* typed_ptr = static_cast<T*>(ptr);
  delete typed_ptr;
}

template <class T>
inline constexpr TypeMetaData::Delete* _PickDelete() noexcept {
  return &_Delete<T>;
}

class _Uninitialized final {};

template <class T>
inline constexpr TypeMetaData _makeTypeMetaDataInstance() {
  return {sizeof(T),
          _PickNew<T>(),
          _PickPlacementNew<T>(),
          _PickCopy<T>(),
          _PickPlacementDelete<T>(),
          _PickDelete<T>(),
          TypeIdentifier::Get<T>(),
          c10::util::get_fully_qualified_type_name<T>()};
}

template <>
inline constexpr TypeMetaData _makeTypeMetaDataInstance<_Uninitialized>() {
  return {0,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          TypeIdentifier::uninitialized(),
          "nullptr (uninitialized)"};
}

} // namespace detail

/**
 * TypeMeta is a thin class that allows us to store the type of a container such
 * as a blob, or the data type of a tensor, with a unique run-time id. It also
 * stores some additional data such as the item size and the name of the type
 * for run-time inspection.
 */
class C10_API TypeMeta {
 public:
  using New = detail::TypeMetaData::New;
  using PlacementNew = detail::TypeMetaData::PlacementNew;
  using Copy = detail::TypeMetaData::Copy;
  using PlacementDelete = detail::TypeMetaData::PlacementDelete;
  using Delete = detail::TypeMetaData::Delete;

  /**
   * Create a dummy TypeMeta object. To create a TypeMeta object for a specific
   * type, use TypeMeta::Make<T>().
   */
  TypeMeta() noexcept;

  /**
   * Copy constructor.
   */
  constexpr TypeMeta(const TypeMeta& src) noexcept = default;

  /**
   * Assignment operator.
   */
  AT_CPP14_CONSTEXPR TypeMeta& operator=(const TypeMeta& src) noexcept =
      default;

  constexpr TypeMeta(TypeMeta&& rhs) noexcept = default;

 private:
  // TypeMeta can only be created by Make, making sure that we do not
  // create incorrectly mixed up TypeMeta objects.
  explicit constexpr TypeMeta(const detail::TypeMetaData* data) noexcept
      : data_(data) {}

 public:
  /**
   * Returns the type id.
   */
  constexpr TypeIdentifier id() const noexcept {
    return data_->id_;
  }
  /**
   * Returns the size of the item.
   */
  constexpr size_t itemsize() const noexcept {
    return data_->itemsize_;
  }
  constexpr New* newFn() const noexcept {
    return data_->new_;
  }
  /**
   * Returns the placement new function pointer for individual items.
   */
  constexpr PlacementNew* placementNew() const noexcept {
    return data_->placementNew_;
  }
  /**
   * Returns the typed copy function pointer for individual iterms.
   */
  constexpr Copy* copy() const noexcept {
    return data_->copy_;
  }
  /**
   * Returns the destructor function pointer for individual items.
   */
  constexpr PlacementDelete* placementDelete() const noexcept {
    return data_->placementDelete_;
  }
  constexpr Delete* deleteFn() const noexcept {
    return data_->delete_;
  }
  /**
   * Returns a printable name for the type.
   */
  constexpr c10::string_view name() const noexcept {
    return data_->name_;
  }

  friend constexpr bool operator==(
      const TypeMeta& lhs,
      const TypeMeta& rhs) noexcept;

  template <typename T>
  constexpr bool Match() const noexcept {
    return (id() == TypeIdentifier::Get<T>());
  }

  // Below are static functions that can be called by passing a specific type.

  template <class T>
  static TypeIdentifier Id() noexcept {
    return TypeIdentifier::Get<T>();
  }

  template <class T>
  static constexpr c10::string_view TypeName() noexcept {
    return c10::util::get_fully_qualified_type_name<T>();
  }

  template <class T>
  static constexpr size_t ItemSize() noexcept {
    return sizeof(T);
  }

  /**
   * Returns a TypeMeta object that corresponds to the typename T.
   */
  template <typename T>
  static TypeMeta Make() {
    static constexpr detail::TypeMetaData singleton =
        detail::_makeTypeMetaDataInstance<T>();
    return TypeMeta(&singleton);
  }

 private:
  const detail::TypeMetaData* data_;
};

inline TypeMeta::TypeMeta() noexcept
    : TypeMeta(TypeMeta::Make<detail::_Uninitialized>()) {}

inline constexpr bool operator==(
    const TypeMeta& lhs,
    const TypeMeta& rhs) noexcept {
  return (lhs.id() == rhs.id());
}
inline constexpr bool operator!=(
    const TypeMeta& lhs,
    const TypeMeta& rhs) noexcept {
  return !operator==(lhs, rhs);
}

inline std::ostream& operator<<(
    std::ostream& stream,
    caffe2::TypeMeta typeMeta) {
  return stream << typeMeta.name();
}

// Deprecated. CAFFE_KNOWN_TYPE is not needed anymore.
// TODO Remove all CAFFE_KNOWN_TYPE occurrences
#define CAFFE_KNOWN_TYPE(T)

} // namespace caffe2
