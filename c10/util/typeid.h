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

#include "c10/macros/Macros.h"
#include "c10/util/Backtrace.h"
#include "c10/util/C++17.h"
#include "c10/util/Exception.h"
#include "c10/util/Half.h"
#include "c10/util/IdWrapper.h"
#include "c10/util/qint8.h"

#include "c10/util/Type.h"

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
namespace std {
template <>
struct is_fundamental<at::Half> : std::true_type {};
} // namespace std

namespace caffe2 {

/**
 * A type id is a unique id for a given C++ type.
 * You need to register your types using CAFFE_KNOWN_TYPE(MyType) to be able to
 * use TypeIdentifier with custom types. This is for example used to store the
 * dtype of tensors.
 */
class C10_API TypeIdentifier final
    : public at::IdWrapper<TypeIdentifier, uint16_t> {
 public:
  static TypeIdentifier createTypeId();

  friend std::ostream& operator<<(std::ostream& stream, TypeIdentifier typeId);
  friend bool operator<(TypeIdentifier lhs, TypeIdentifier rhs);

  // 0 is uint8_t (due to ScalarType BC constraint)
  static constexpr TypeIdentifier uninitialized() {
    return TypeIdentifier(11);
  }

  /**
   * Returns the unique id for the given type T. The id is unique for the type T
   * in the sense that for any two different types, their ids are different; for
   * the same type T, the id remains the same over different calls of the
   * function. However, this is not guaranteed over different runs, as the id
   * is generated during run-time. Do NOT serialize the id for storage.
   */
  template <typename T>
  C10_API static TypeIdentifier Get();

 private:
  constexpr explicit TypeIdentifier(uint16_t id) : IdWrapper(id) {}
  friend class TypeMeta;
};

// Allow usage in std::map / std::set
// TODO Disallow this and rather use std::unordered_map/set everywhere
inline bool operator<(TypeIdentifier lhs, TypeIdentifier rhs) {
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
      const char* name) noexcept
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
  const char* name_;
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
      "Type " + std::string(c10::demangle_type<T>()) +
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
      "Type " + std::string(c10::demangle_type<T>()) +
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
      "Type " + std::string(c10::demangle_type<T>()) +
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

#ifdef __GXX_RTTI
template <class T>
const char* _typeName(const char* literalName) noexcept {
  std::ignore = literalName; // suppress unused warning
  static const std::string name = c10::demangle(typeid(T).name());
  return name.c_str();
}
#else
template <class T>
constexpr const char* _typeName(const char* literalName) noexcept {
  return literalName;
}
#endif

template <class T>
inline TypeMetaData _makeTypeMetaDataInstance(const char* typeName) {
  return {sizeof(T),
          _PickNew<T>(),
          _PickPlacementNew<T>(),
          _PickCopy<T>(),
          _PickPlacementDelete<T>(),
          _PickDelete<T>(),
          TypeIdentifier::Get<T>(),
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
class C10_API TypeMeta {
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
  constexpr const char* name() const noexcept {
    return data_->name_;
  }

  friend bool operator==(const TypeMeta& lhs, const TypeMeta& rhs) noexcept;

  template <typename T>
  constexpr bool Match() const noexcept {
    return (*this == Make<T>());
  }

  // Below are static functions that can be called by passing a specific type.

  template <class T>
  static TypeIdentifier Id() noexcept {
    return TypeIdentifier::Get<T>();
  }

  template <class T>
  static const char* TypeName() noexcept {
    return Make<T>().name();
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
    : data_(_typeMetaDataInstance<detail::_Uninitialized>()) {}

inline bool operator==(const TypeMeta& lhs, const TypeMeta& rhs) noexcept {
  return (lhs.data_ == rhs.data_);
}
inline bool operator!=(const TypeMeta& lhs, const TypeMeta& rhs) noexcept {
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

#define _CAFFE_KNOWN_TYPE_DEFINE_TYPEMETADATA_INSTANCE(T, Counter)   \
  namespace detail {                                                 \
  const TypeMetaData MACRO_CONCAT(_typeMetaDataInstance_, Counter) = \
      _makeTypeMetaDataInstance<T>(_typeName<T>(#T));                \
  }                                                                  \
  template <>                                                        \
  EXPORT_IF_NOT_GCC const detail::TypeMetaData*                      \
  TypeMeta::_typeMetaDataInstance<T>() noexcept {                    \
    return &MACRO_CONCAT(detail::_typeMetaDataInstance_, Counter);   \
  }
#define CAFFE_KNOWN_TYPE(T)                                               \
  template <>                                                             \
  EXPORT_IF_NOT_GCC TypeIdentifier TypeIdentifier::Get<T>() {             \
    static const TypeIdentifier type_id = TypeIdentifier::createTypeId(); \
    return type_id;                                                       \
  }                                                                       \
  _CAFFE_KNOWN_TYPE_DEFINE_TYPEMETADATA_INSTANCE(T, __COUNTER__)

/**
 * CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE is used
 * to preallocate ids for types that are queried very often so that they
 * can be resolved at compile time. Please use CAFFE_KNOWN_TYPE() instead
 * for your own types to allocate dynamic ids for them.
 */
#ifdef _MSC_VER
#define CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(PreallocatedId, T) \
  template <>                                                    \
  inline C10_EXPORT TypeIdentifier TypeIdentifier::Get<T>() {    \
    return TypeIdentifier(PreallocatedId);                       \
  }                                                              \
  namespace detail {                                             \
  C10_API extern const TypeMetaData MACRO_CONCAT(                \
      _typeMetaDataInstance_preallocated_,                       \
      PreallocatedId);                                           \
  }
#define CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(PreallocatedId, T)         \
  namespace detail {                                                    \
  C10_EXPORT const TypeMetaData MACRO_CONCAT(                           \
      _typeMetaDataInstance_preallocated_,                              \
      PreallocatedId) = _makeTypeMetaDataInstance<T>(_typeName<T>(#T)); \
  }                                                                     \
  template <>                                                           \
  C10_EXPORT const detail::TypeMetaData*                                \
  TypeMeta::_typeMetaDataInstance<T>() noexcept {                       \
    return &MACRO_CONCAT(                                               \
        detail::_typeMetaDataInstance_preallocated_, PreallocatedId);   \
  }
#else // _MSC_VER
#define CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(PreallocatedId, T)      \
  template <>                                                         \
  inline C10_EXPORT TypeIdentifier TypeIdentifier::Get<T>() {         \
    return TypeIdentifier(PreallocatedId);                            \
  }                                                                   \
  namespace detail {                                                  \
  C10_EXPORT extern const TypeMetaData MACRO_CONCAT(                  \
      _typeMetaDataInstance_preallocated_,                            \
      PreallocatedId);                                                \
  }                                                                   \
  template <>                                                         \
  inline const detail::TypeMetaData*                                  \
  TypeMeta::_typeMetaDataInstance<T>() noexcept {                     \
    return &MACRO_CONCAT(                                             \
        detail::_typeMetaDataInstance_preallocated_, PreallocatedId); \
  }
#define CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(PreallocatedId, T)         \
  namespace detail {                                                    \
  const TypeMetaData MACRO_CONCAT(                                      \
      _typeMetaDataInstance_preallocated_,                              \
      PreallocatedId) = _makeTypeMetaDataInstance<T>(_typeName<T>(#T)); \
  }
#endif

// Note: we have preallocated the numbers so they line up exactly
// with at::ScalarType's numbering.  All other numbers do not matter.

struct _CaffeHighestPreallocatedTypeId final {};
// TODO static_assert number of declare/define align
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(0, uint8_t)
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(1, int8_t)
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(2, int16_t)
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(3, int)
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(4, int64_t)
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(5, at::Half)
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(6, float)
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(7, double)
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(8, at::ComplexHalf)
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(9, std::complex<float>)
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(10, std::complex<double>)
// 11 = undefined type id
// 12 = Tensor (defined in tensor.h)
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(13, std::string)
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(14, bool)
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(15, uint16_t)
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(16, char)
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(17, std::unique_ptr<std::mutex>)
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(18, std::unique_ptr<std::atomic<bool>>)
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(19, std::vector<int32_t>)
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(20, std::vector<int64_t>)
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(21, std::vector<unsigned long>)
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(22, bool*)
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(23, char*)
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(24, int*)

// For some of the compilers, long is definied separately from int32_t and
// int64_t. As a result we will need to actually define them separately.
// It is recommended that one does NOT use long - use int32_t and int64_t
// explicitly. Explicit long type annotation may go away in the future.
// details: This hack works by defining a _guard_long_unique type, which is
// long iff the compiler has a separate long type and is a dummy type otherwise.
// we then allocate a type id to that _guard_long_unique. If the compiler has a
// separate long type, this allocates a type id for long. Otherwise, it
// allocates a type id for the dummy type, which doesn't matter.
namespace detail {
template <class T>
class _guard_long_unique_dummy final {};
template <class T>
using _guard_long_unique = c10::guts::conditional_t<
    std::is_same<long, int32_t>::value || std::is_same<long, int64_t>::value,
    _guard_long_unique_dummy<T>,
    T>;
} // namespace detail
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(25, detail::_guard_long_unique<long>)
CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(
    26,
    detail::_guard_long_unique<std::vector<long>>)

CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(27, c10::qint8);

CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(28, _CaffeHighestPreallocatedTypeId)
} // namespace caffe2
