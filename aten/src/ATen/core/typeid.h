#pragma once

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <complex>
#ifdef __GXX_RTTI
#include <typeinfo>
#endif

#include <exception>

#include "ATen/core/Error.h"
#include "ATen/core/Backtrace.h"
#include "ATen/core/Macros.h"
#include "ATen/core/Half.h"
#include "ATen/core/IdWrapper.h"

// TODO: This file is still in the caffe2 namespace, despite living
// in the ATen directory.  This is because the macro CAFFE_DECLARE_KNOWN_TYPE
// defines a template specialization, which relies on the namespace of TypeMeta
// matching the namespace where the macro is called.  This requires us to
// fix all of the call-sites, which I want to do later.  So the namespace
// is not fixed at the moment.

// Make at::Half a fundamental type.
namespace std {
template<>
struct is_fundamental<at::Half> : std::true_type {
};
}  // namespace std

namespace caffe2 {

class TypeMeta;

/**
 * A type id is a unique id for a given C++ type.
 * You need to register your types using CAFFE_KNOWN_TYPE(MyType) to be able to
 * use TypeIdentifier with custom types. This is for example used to store the
 * dtype of tensors.
 */
class CAFFE2_API TypeIdentifier final
    : public at::IdWrapper<TypeIdentifier, uint16_t> {
 public:
  static TypeIdentifier createTypeId();

  friend std::ostream& operator<<(
      std::ostream& stream,
      TypeIdentifier typeId);
  friend bool operator<(TypeIdentifier lhs, TypeIdentifier rhs);

  // 0 is uint8_t (due to ScalarType BC constraint)
  static constexpr TypeIdentifier uninitialized() {
    return TypeIdentifier(11);
  }

  const char* name() const noexcept;

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

AT_DEFINE_HASH_FOR_IDWRAPPER(caffe2::TypeIdentifier)

namespace caffe2 {

CAFFE2_API std::unordered_map<TypeIdentifier, std::string>& gTypeNames();
CAFFE2_API std::unordered_set<std::string>& gRegisteredTypeNames();

inline const char* TypeIdentifier::name() const noexcept {
  auto it = gTypeNames().find(*this);
  assert(it != gTypeNames().end());
  return it->second.c_str();
}

CAFFE2_API std::mutex& gTypeRegistrationMutex();

template <typename T>
struct TypeNameRegisterer {
  TypeNameRegisterer(TypeIdentifier id, const std::string& literal_name) {
    std::lock_guard<std::mutex> guard(gTypeRegistrationMutex());
#ifdef __GXX_RTTI
    (void)literal_name;

    std::string name = at::demangle(typeid(T).name());
    // If we are in RTTI mode, we will also use this opportunity to do sanity
    // check if there are duplicated ids registered for the same type. This
    // usually happens when one does not do RTLD_GLOBAL, which is often the
    // case in Python. The way we do the check is to make sure that there are
    // no duplicated names registered - this could be done by checking the
    // uniqueness of names.
    if (gRegisteredTypeNames().count(name)) {
      AT_ERROR("typeid.h: Type name ", name, " was registered twice.  "
               "This should not happen.  Things to check:\n"
               "1. Did you add a new CAFFE_KNOWN_TYPE?  If so, check that "
               "it is not duplicated with an existing CAFFE_KNOWN_TYPE.\n"
               "2. Did you build and install PyTorch and Caffe2 separately? "
               "For example, this would be the case if you ran scripts/onnx/install.sh or "
               "scripts/onnx/install-develop.sh prior to Aug 12, 2018 "
               "(commit 1756daaa7530d).  If so, rebuild using the environment variable "
               " FULL_CAFFE2=1 (if you build latest master, the ONNX scripts are "
               "updated to do this for you.) "
               "For more context, see https://github.com/pytorch/pytorch/issues/10460");
    }
    gRegisteredTypeNames().insert(name);
    gTypeNames()[id] = name;
#else // __GXX_RTTI
    if (literal_name.empty()) {
      gTypeNames()[id] = "(RTTI disabled, cannot show name)";
    } else {
      gTypeNames()[id] = literal_name;
    }
#endif // __GXX_RTTI
  }
};

/**
 * TypeMeta is a thin class that allows us to store the type of a container such
 * as a blob, or the data type of a tensor, with a unique run-time id. It also
 * stores some additional data such as the item size and the name of the type
 * for run-time inspection.
 */
class CAFFE2_API TypeMeta {
 public:
  using PlacementNew = void(void*, size_t);
  using TypedCopy = void(const void*, void*, size_t);
  using TypedDestructor = void(void*, size_t);
  /** Create a dummy TypeMeta object. To create a TypeMeta object for a specific
   * type, use TypeMeta::Make<T>().
   */
  TypeMeta() noexcept
      : id_(TypeIdentifier::uninitialized()),
        itemsize_(0),
        ctor_(nullptr),
        copy_(nullptr),
        dtor_(nullptr) {}

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
  TypeMeta(
      TypeIdentifier i,
      size_t s,
      PlacementNew* ctor,
      TypedCopy* copy,
      TypedDestructor* dtor) noexcept
      : id_(i), itemsize_(s), ctor_(ctor), copy_(copy), dtor_(dtor) {}

  // Mechanism for throwing errors which can't be prevented at compile time
  // due to type erasure. E.g. somebody calling TypeMeta::copy() for
  // non-copiable type. Right now just throws exception but is implemented
  // in .cpp to manage dependencies
  static void _ThrowRuntimeTypeLogicError(const std::string& msg);

 public:
  /**
   * Returns the type id.
   */
  const TypeIdentifier& id() const noexcept {
    return id_;
  }
  /**
   * Returns the size of the item.
   */
  const size_t& itemsize() const noexcept {
    return itemsize_;
  }
  /**
   * Returns the placement new function pointer for individual items.
   */
  PlacementNew* ctor() const noexcept {
    return ctor_;
  }
  /**
   * Returns the typed copy function pointer for individual iterms.
   */
  TypedCopy* copy() const noexcept {
    return copy_;
  }
  /**
   * Returns the destructor function pointer for individual items.
   */
  TypedDestructor* dtor() const noexcept {
    return dtor_;
  }
  /**
   * Returns a printable name for the type.
   */
  const char* name() const noexcept {
    auto it = gTypeNames().find(id_);
    assert(it != gTypeNames().end());
    return it->second.c_str();
  }

  friend bool operator==(const TypeMeta& lhs, const TypeMeta& rhs) noexcept;

  template <typename T>
  bool Match() const noexcept {
    return (id_ == Id<T>());
  }

  // Below are static functions that can be called by passing a specific type.

  /**
   * Returns the unique id for the given type T. The id is unique for the type T
   * in the sense that for any two different types, their id are different; for
   * the same type T, the id remains the same over different calls of the
   * function. However, this is not guaranteed over different runs, as the id
   * is generated during run-time. Do NOT serialize the id for storage.
   */
  template <typename T>
  CAFFE2_API static TypeIdentifier Id();

  /**
   * Returns the item size of the type. This is equivalent to sizeof(T).
   */
  template <typename T>
  static size_t ItemSize() {
    return sizeof(T);
  }

  /**
   * Returns the registered printable name of the type.
   *
   * Works for only the ones registered with CAFFE_KNOWN_TYPE
   */
  template <typename T>
  static const char* TypeName() {
    auto it = gTypeNames().find(Id<T>());
    assert(it != gTypeNames().end());
    return it->second.c_str();
  }

  /**
   * Placement new function for the type.
   */
  template <typename T>
  static void _Ctor(void* ptr, size_t n) {
    T* typed_ptr = static_cast<T*>(ptr);
    for (size_t i = 0; i < n; ++i) {
      new (typed_ptr + i) T;
    }
  }

  template <typename T>
  static void _CtorNotDefault(void* /*ptr*/, size_t /*n*/) {
    _ThrowRuntimeTypeLogicError(
        "Type " + std::string(at::demangle_type<T>()) +
        " is not default-constructible.");
  }

  template <
      typename T,
      typename std::enable_if<std::is_default_constructible<T>::value>::type* =
          nullptr>
  static inline PlacementNew* _PickCtor() {
    return _Ctor<T>;
  }

  template <
      typename T,
      typename std::enable_if<!std::is_default_constructible<T>::value>::type* =
          nullptr>
  static inline PlacementNew* _PickCtor() {
    return _CtorNotDefault<T>;
  }

  /**
   * Typed copy function for classes.
   */
  template <typename T>
  static void _Copy(const void* src, void* dst, size_t n) {
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
  static void _CopyNotAllowed(
      const void* /*src*/,
      void* /*dst*/,
      size_t /*n*/) {
    _ThrowRuntimeTypeLogicError(
        "Type " + std::string(at::demangle_type<T>()) +
        " does not allow assignment.");
  }

  template <
      typename T,
      typename std::enable_if<std::is_copy_assignable<T>::value>::type* =
          nullptr>
  static inline TypedCopy* _PickCopy() {
    return _Copy<T>;
  }

  template <
      typename T,
      typename std::enable_if<!std::is_copy_assignable<T>::value>::type* =
          nullptr>
  static inline TypedCopy* _PickCopy() {
    return _CopyNotAllowed<T>;
  }

  /**
   * Destructor for non-fundamental types.
   */
  template <typename T>
  static void _Dtor(void* ptr, size_t n) {
    T* typed_ptr = static_cast<T*>(ptr);
    for (size_t i = 0; i < n; ++i) {
      typed_ptr[i].~T();
    }
  }

  /**
   * Returns a TypeMeta object that corresponds to the typename T.
   */
  template <typename T>
  static typename std::enable_if<
      std::is_fundamental<T>::value || std::is_pointer<T>::value,
      TypeMeta>::type
  Make() {
    return TypeMeta(Id<T>(), ItemSize<T>(), nullptr, nullptr, nullptr);
  }

  template <typename T>
  static typename std::enable_if<
      !(std::is_fundamental<T>::value || std::is_pointer<T>::value),
      TypeMeta>::type
  Make() {
    return TypeMeta(
        Id<T>(), ItemSize<T>(), _PickCtor<T>(), _PickCopy<T>(), _Dtor<T>);
  }

 private:
  TypeIdentifier id_;
  size_t itemsize_;
  PlacementNew* ctor_;
  TypedCopy* copy_;
  TypedDestructor* dtor_;
};

inline bool operator==(const TypeMeta& lhs, const TypeMeta& rhs) noexcept {
  return (lhs.id_ == rhs.id_);
}
inline bool operator!=(const TypeMeta& lhs, const TypeMeta& rhs) noexcept {
  return !operator==(lhs, rhs);
}

/**
 * Register unique id for a type so it can be used in TypeMeta context, e.g. be
 * used as a type for Blob or for Tensor elements.
 *
 * CAFFE_KNOWN_TYPE does explicit instantiation of TypeMeta::Id<T> template
 * function and thus needs to be put in a single translation unit (.cpp file)
 * for a given type T. Other translation units that use type T as a type of the
 * caffe2::Blob or element type of caffe2::Tensor need to depend on the
 * translation unit that contains CAFFE_KNOWN_TYPE declaration via regular
 * linkage dependencies.
 *
 * NOTE: the macro needs to be invoked in ::caffe2 namespace
 */
// Implementation note: in MSVC, we will need to prepend the CAFFE2_API
// keyword in order to get things compiled properly. in Linux, gcc seems to
// create attribute ignored error for explicit template instantiations, see
//   http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0537r0.html
//   https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51930
// and as a result, we define these two macros slightly differently.
#if defined(_MSC_VER) || defined(__clang__)
#define CAFFE_KNOWN_TYPE(T)                                               \
  template <>                                                             \
  C10_EXPORT TypeIdentifier TypeMeta::Id<T>() {                           \
    static const TypeIdentifier type_id = TypeIdentifier::createTypeId(); \
    static TypeNameRegisterer<T> registerer(type_id, #T);                 \
    return type_id;                                                       \
  }
#else // defined(_MSC_VER) || defined(__clang__)
#define CAFFE_KNOWN_TYPE(T)                                               \
  template <>                                                             \
  TypeIdentifier TypeMeta::Id<T>() {                                      \
    static const TypeIdentifier type_id = TypeIdentifier::createTypeId(); \
    static TypeNameRegisterer<T> registerer(type_id, #T);                 \
    return type_id;                                                       \
  }
#endif // defined(_MSC_VER) || defined(__clang__)

/**
 * CAFFE_DECLARE_KNOWN_TYPE and CAFFE_DEFINE_KNOWN_TYPE are used
 * to preallocate ids for types that are queried very often so that they
 * can be resolved at compile time. Please use CAFFE_KNOWN_TYPE() instead
 * for your own types to allocate dynamic ids for them.
 */
#ifdef _MSC_VER
#define CAFFE_DECLARE_KNOWN_TYPE(PreallocatedId, T)    \
  template <>                                          \
  inline CAFFE2_API TypeIdentifier TypeMeta::Id<T>() { \
    return TypeIdentifier(PreallocatedId);             \
  }
#else // _MSC_VER
#define CAFFE_DECLARE_KNOWN_TYPE(PreallocatedId, T) \
  template <>                                       \
  inline TypeIdentifier TypeMeta::Id<T>() {         \
    return TypeIdentifier(PreallocatedId);          \
  }
#endif

#define CONCAT_IMPL(x, y) x##y
#define MACRO_CONCAT(x, y) CONCAT_IMPL(x, y)

#define CAFFE_DEFINE_KNOWN_TYPE(T)                             \
  namespace {                                                  \
  TypeNameRegisterer<T> MACRO_CONCAT(registerer, __COUNTER__)( \
      TypeMeta::Id<T>(),                                       \
      #T);                                                     \
  }

class Tensor;

// Note: we have preallocated the numbers so they line up exactly
// with at::ScalarType's numbering.  All other numbers do not matter.

struct _CaffeHighestPreallocatedTypeId final {};

CAFFE_DECLARE_KNOWN_TYPE(0, uint8_t)
CAFFE_DECLARE_KNOWN_TYPE(1, int8_t)
CAFFE_DECLARE_KNOWN_TYPE(2, int16_t)
CAFFE_DECLARE_KNOWN_TYPE(3, int)
CAFFE_DECLARE_KNOWN_TYPE(4, int64_t)
CAFFE_DECLARE_KNOWN_TYPE(5, at::Half)
CAFFE_DECLARE_KNOWN_TYPE(6, float)
CAFFE_DECLARE_KNOWN_TYPE(7, double)
CAFFE_DECLARE_KNOWN_TYPE(8, at::ComplexHalf)
CAFFE_DECLARE_KNOWN_TYPE(9, std::complex<float>)
CAFFE_DECLARE_KNOWN_TYPE(10, std::complex<double>)
// 11 = undefined type id

CAFFE_DECLARE_KNOWN_TYPE(12, Tensor)
CAFFE_DECLARE_KNOWN_TYPE(13, std::string)
CAFFE_DECLARE_KNOWN_TYPE(14, bool)
CAFFE_DECLARE_KNOWN_TYPE(15, uint16_t)
CAFFE_DECLARE_KNOWN_TYPE(16, char)
CAFFE_DECLARE_KNOWN_TYPE(17, std::unique_ptr<std::mutex>)
CAFFE_DECLARE_KNOWN_TYPE(18, std::unique_ptr<std::atomic<bool>>)
CAFFE_DECLARE_KNOWN_TYPE(19, std::vector<int32_t>)
CAFFE_DECLARE_KNOWN_TYPE(20, std::vector<int64_t>)
CAFFE_DECLARE_KNOWN_TYPE(21, std::vector<unsigned long>)
CAFFE_DECLARE_KNOWN_TYPE(22, bool*)
CAFFE_DECLARE_KNOWN_TYPE(23, char*)
CAFFE_DECLARE_KNOWN_TYPE(24, int*)

// For some of the compilers, long is definied separately from int32_t and
// int64_t. As a result we will need to actually define them separately.
// It is recommended that one does NOT use long - use int32_t and int64_t
// explicitly. Explicit long type annotation may go away in the future.
#if defined(_MSC_VER) || defined(__APPLE__) || \
    (defined(__ANDROID__) && !defined(__LP64__))
CAFFE_DECLARE_KNOWN_TYPE(25, long)
CAFFE_DECLARE_KNOWN_TYPE(26, std::vector<long>)
#endif 

CAFFE_DECLARE_KNOWN_TYPE(27, _CaffeHighestPreallocatedTypeId)
} // namespace caffe2
