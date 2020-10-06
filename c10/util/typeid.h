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
#include <c10/util/Backtrace.h>
#include <c10/util/C++17.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <c10/util/IdWrapper.h>
#include <c10/util/Type.h>
#include <c10/util/TypeTraits.h>
#include <c10/util/TypeIndex.h>
#include <c10/util/qint32.h>
#include <c10/util/qint8.h>
#include <c10/util/quint8.h>
#include <c10/util/quint4x2.h>
#include <c10/util/BFloat16.h>
#include <c10/util/flat_hash_map.h>

/*
 * TypeIdentifier is a small type containing an id.
 * Types must be registered using CAFFE_KNOWN_TYPE() for them to have a type id.
 * If a type is registered, you can also create an object containing meta data
 * like constructor, destructor, stringified name, ... about the type by calling
 * TypeMeta::Make<T>. This returns a TypeMeta() object, which is basically just
 * a pointer to the type information, so it's cheap to pass around.
 */

// TODO: This file is still in the caffe2 namespace, despite living
// in the ATen directory.  This is because the macro
// CAFFE_KNOWN_TYPE defines a template specialization, which relies
// on the namespace of TypeMeta matching the namespace where the macro is
// called.  This requires us to fix all of the call-sites, which I want to do
// later.  So the namespace is not fixed at the moment.

// Make at::Half a fundamental type.
namespace c10 {
namespace guts {
template <>
struct is_fundamental<at::Half> : std::true_type {};
} // namespace guts
} // namespace c10

namespace caffe2 {

/**
 * A type id is a unique id for a given C++ type.
 * You need to register your types using CAFFE_KNOWN_TYPE(MyType) to be able to
 * use TypeIdentifier with custom types. This is for example used to store the
 * dtype of tensors.
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
  friend class TypeMeta; // TODO Is this friend an issue?
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
    std::enable_if_t<std::is_default_constructible<T>::value>* = nullptr>
inline constexpr TypeMetaData::PlacementNew* _PickPlacementNew() {
  return (c10::guts::is_fundamental<T>::value || std::is_pointer<T>::value)
      ? nullptr
      : &_PlacementNew<T>;
}

template <
    typename T,
    std::enable_if_t<!std::is_default_constructible<T>::value>* = nullptr>
inline constexpr TypeMetaData::PlacementNew* _PickPlacementNew() {
  static_assert(
      !c10::guts::is_fundamental<T>::value && !std::is_pointer<T>::value,
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
    std::enable_if_t<std::is_default_constructible<T>::value>* = nullptr>
inline constexpr TypeMetaData::New* _PickNew() {
  return &_New<T>;
}

template <
    typename T,
    std::enable_if_t<!std::is_default_constructible<T>::value>* = nullptr>
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
    std::enable_if_t<std::is_copy_assignable<T>::value>* = nullptr>
inline constexpr TypeMetaData::Copy* _PickCopy() {
  return (c10::guts::is_fundamental<T>::value || std::is_pointer<T>::value)
      ? nullptr
      : &_Copy<T>;
}

template <
    typename T,
    std::enable_if_t<!std::is_copy_assignable<T>::value>* = nullptr>
inline constexpr TypeMetaData::Copy* _PickCopy() {
  static_assert(
      !c10::guts::is_fundamental<T>::value && !std::is_pointer<T>::value,
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
  return (c10::guts::is_fundamental<T>::value || std::is_pointer<T>::value)
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

template <class T>
inline C10_TYPENAME_CONSTEXPR TypeMetaData _makeTypeMetaDataInstance() {
  C10_HOST_CONSTEXPR_VAR auto typeId = TypeIdentifier::Get<T>();
  C10_TYPENAME_CONSTEXPR auto typeName = c10::util::get_fully_qualified_type_name<T>();

  return {sizeof(T),
          _PickNew<T>(),
          _PickPlacementNew<T>(),
          _PickCopy<T>(),
          _PickPlacementDelete<T>(),
          _PickDelete<T>(),
          typeId,
          typeName};
}

class _Uninitialized final {};

} // namespace detail

/**
 * TypeMeta is a thin class that allows us to store the type of a container such
 * as a blob, or the data type of a tensor, with a unique run-time id. It also
 * stores some additional data such as the item size and the name of the type
 * for run-time inspection.
 */
class C10_API TypeMeta final {
 public:
  using New = detail::TypeMetaData::New;
  using PlacementNew = detail::TypeMetaData::PlacementNew;
  using Copy = detail::TypeMetaData::Copy;
  using PlacementDelete = detail::TypeMetaData::PlacementDelete;
  using Delete = detail::TypeMetaData::Delete;

  /** Create a dummy TypeMeta object. To create a TypeMeta object for a specific
   * type, use TypeMeta::Make<T>().
   */
  TypeMeta() noexcept;

  /**
   * Copy constructor.
   */
  TypeMeta(const TypeMeta& src) noexcept = default;

  /**
   * Assignment operator.
   */
  TypeMeta& operator=(const TypeMeta& src) noexcept = default;

  TypeMeta(TypeMeta&& rhs) noexcept = default;

 private:
  // TypeMeta can only be created by Make, making sure that we do not
  // create incorrectly mixed up TypeMeta objects.
  explicit TypeMeta(const detail::TypeMetaData* data) noexcept
  : data_(data) {
  }

 public:
  /**
   * Returns the type id.
   */
  TypeIdentifier id() const noexcept {
    return data_->id_;
  }
  /**
   * Returns the size of the item.
   */
  size_t itemsize() const noexcept {
    return data_->itemsize_;
  }
  New* newFn() const noexcept {
    return data_->new_;
  }
  /**
   * Returns the placement new function pointer for individual items.
   */
  PlacementNew* placementNew() const noexcept {
    return data_->placementNew_;
  }
  /**
   * Returns the typed copy function pointer for individual iterms.
   */
  Copy* copy() const noexcept {
    return data_->copy_;
  }
  /**
   * Returns the destructor function pointer for individual items.
   */
  PlacementDelete* placementDelete() const noexcept {
    return data_->placementDelete_;
  }
  Delete* deleteFn() const noexcept {
    return data_->delete_;
  }
  /**
   * Returns a printable name for the type.
   */
  c10::string_view name() const noexcept {
    return data_->name_;
  }

  friend bool operator==(
      const TypeMeta& lhs,
      const TypeMeta& rhs) noexcept;

  template <typename T>
  bool Match() const noexcept {
    return (*this == Make<T>());
  }

  // Below are static functions that can be called by passing a specific type.

  template <class T>
  static C10_HOST_CONSTEXPR TypeIdentifier Id() noexcept {
    return TypeIdentifier::Get<T>();
  }

  template <class T>
  static C10_TYPENAME_CONSTEXPR c10::string_view TypeName() noexcept {
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
    // The instance pointed to is declared here, but defined in a .cpp file.
    // We need to silence the compiler warning about using an undefined
    // variable template. '-Wpragmas' and '-Wunknown-warning-option' has to be
    // disabled for compilers that don't know '-Wundefined-var-template' and
    // would error at our attempt to disable it.
#ifndef _MSC_VER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#pragma GCC diagnostic ignored "-Wundefined-var-template"
#endif
    return TypeMeta(_typeMetaDataInstance<T>());
#ifndef _MSC_VER
#pragma GCC diagnostic pop
#endif
  }

 private:
  const detail::TypeMetaData* data_;

  template <class T>
  C10_API static const detail::TypeMetaData* _typeMetaDataInstance() noexcept;
};

template <>
C10_EXPORT const detail::TypeMetaData* TypeMeta::_typeMetaDataInstance<
    detail::_Uninitialized>() noexcept;

inline TypeMeta::TypeMeta() noexcept
    : data_(_typeMetaDataInstance<detail::_Uninitialized>()) {
}

inline bool operator==(
    const TypeMeta& lhs,
    const TypeMeta& rhs) noexcept {
  return (lhs.data_ == rhs.data_);
}
inline bool operator!=(
    const TypeMeta& lhs,
    const TypeMeta& rhs) noexcept {
  return !operator==(lhs, rhs);
}

inline std::ostream& operator<<(
    std::ostream& stream,
    caffe2::TypeMeta typeMeta) {
  return stream << typeMeta.name();
}

/**
 * Register unique id for a type so it can be used in TypeMeta context, e.g. be
 * used as a type for Blob or for Tensor elements.
 *
 * CAFFE_KNOWN_TYPE does explicit instantiation of TypeIdentifier::Get<T>
 * template function and thus needs to be put in a single translation unit (.cpp
 * file) for a given type T. Other translation units that use type T as a type
 * of the caffe2::Blob or element type of caffe2::Tensor need to depend on the
 * translation unit that contains CAFFE_KNOWN_TYPE declaration via regular
 * linkage dependencies.
 *
 * NOTE: the macro needs to be invoked in ::caffe2 namespace
 */
// Implementation note: in MSVC, we will need to prepend the C10_API
// keyword in order to get things compiled properly. in Linux, gcc seems to
// create attribute ignored error for explicit template instantiations, see
//   http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0537r0.html
//   https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51930
// and as a result, we define these two macros slightly differently.
#if defined(_MSC_VER) || defined(__clang__)
#define EXPORT_IF_NOT_GCC C10_EXPORT
#else
#define EXPORT_IF_NOT_GCC
#endif

#define CAFFE_KNOWN_TYPE(T)                                        \
  template <>                                                      \
  EXPORT_IF_NOT_GCC const detail::TypeMetaData*                    \
  TypeMeta::_typeMetaDataInstance<T>() noexcept {                  \
    static C10_TYPENAME_CONSTEXPR detail::TypeMetaData singleton = \
        detail::_makeTypeMetaDataInstance<T>();                    \
    return &singleton;                                             \
  }

} // namespace caffe2
