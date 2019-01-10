#ifndef CAFFE2_CORE_TYPEID_H_
#define CAFFE2_CORE_TYPEID_H_

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <type_traits>
#ifdef __GXX_RTTI
#include <set>
#include <typeinfo>
#endif

#include <exception>

#include "caffe2/core/common.h"

namespace caffe2 {

typedef intptr_t CaffeTypeId;
std::map<CaffeTypeId, string>& gTypeNames();
std::set<string>& gRegisteredTypeNames();

// A utility function to demangle a function name.
string Demangle(const char* name);

/**
 * Returns the printable name of the type.
 *
 * Works for all types, not only the ones registered with CAFFE_KNOWN_TYPE
 */
template <typename T>
static const char* DemangleType() {
#ifdef __GXX_RTTI
  static const string name = Demangle(typeid(T).name());
  return name.c_str();
#else // __GXX_RTTI
  return "(RTTI disabled, cannot show name)";
#endif // __GXX_RTTI
}

// A utility function to return an exception string by prepending its exception
// type before its what() content.
string GetExceptionString(const std::exception& e);

std::mutex& gCaffe2TypeRegistrationMutex();

template <typename T>
struct TypeNameRegisterer {
  TypeNameRegisterer(CaffeTypeId id, const string& literal_name) {
    std::lock_guard<std::mutex> guard(gCaffe2TypeRegistrationMutex());
#ifdef __GXX_RTTI
    (void)literal_name;

    string name = Demangle(typeid(T).name());
    // If we are in RTTI mode, we will also use this opportunity to do sanity
    // check if there are duplicated ids registered for the same type. This
    // usually happens when one does not do RTLD_GLOBAL, which is often the
    // case in Python. The way we do the check is to make sure that there are
    // no duplicated names registered - this could be done by checking the
    // uniqueness of names.
    if (gRegisteredTypeNames().count(name)) {
      std::cerr << "Type name " << name
                << " registered twice. This should "
                   "not happen. Do you have duplicated CAFFE_KNOWN_TYPE?"
                << std::endl;
      throw std::runtime_error("TypeNameRegisterer error with type " + name);
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
class TypeMeta {
 public:
  typedef void (*PlacementNew)(void*, size_t);
  typedef void (*TypedCopy)(const void*, void*, size_t);
  typedef void (*TypedDestructor)(void*, size_t);
  /** Create a dummy TypeMeta object. To create a TypeMeta object for a specific
   * type, use TypeMeta::Make<T>().
   */
  TypeMeta()
      : id_(0), itemsize_(0), ctor_(nullptr), copy_(nullptr), dtor_(nullptr) {}

  /**
   * Copy constructor.
   */
  TypeMeta(const TypeMeta& src)
      : id_(src.id_),
        itemsize_(src.itemsize_),
        ctor_(src.ctor_),
        copy_(src.copy_),
        dtor_(src.dtor_) {}
  /**
   * Assignment operator.
   */
  TypeMeta& operator=(const TypeMeta& src) {
    if (this == &src)
      return *this;
    id_ = src.id_;
    itemsize_ = src.itemsize_;
    ctor_ = src.ctor_;
    copy_ = src.copy_;
    dtor_ = src.dtor_;
    return *this;
  }

 private:
  // TypeMeta can only be created by Make, making sure that we do not
  // create incorrectly mixed up TypeMeta objects.
  TypeMeta(
      CaffeTypeId i,
      size_t s,
      PlacementNew ctor,
      TypedCopy copy,
      TypedDestructor dtor)
      : id_(i), itemsize_(s), ctor_(ctor), copy_(copy), dtor_(dtor) {}

 public:
  /**
   * Returns the type id.
   */
  inline const CaffeTypeId& id() const {
    return id_;
  }
  /**
   * Returns the size of the item.
   */
  inline const size_t& itemsize() const {
    return itemsize_;
  }
  /**
   * Returns the placement new function pointer for individual items.
   */
  inline PlacementNew ctor() const {
    return ctor_;
  }
  /**
   * Returns the typed copy function pointer for individual iterms.
   */
  inline TypedCopy copy() const {
    return copy_;
  }
  /**
   * Returns the destructor function pointer for individual items.
   */
  inline TypedDestructor dtor() const {
    return dtor_;
  }
  /**
   * Returns a printable name for the type.
   */
  inline const char* name() const {
    auto it = gTypeNames().find(id_);
    assert(it != gTypeNames().end());
    return it->second.c_str();
  }
  inline bool operator==(const TypeMeta& m) const {
    return (id_ == m.id_);
  }
  inline bool operator!=(const TypeMeta& m) const {
    return (id_ != m.id_);
  }

  template <typename T>
  inline bool Match() const {
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
  CAFFE2_API static CaffeTypeId Id();

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
    for (int i = 0; i < n; ++i) {
      new (typed_ptr + i) T;
    }
  }

  /**
   * Typed copy function for classes.
   */
  template <typename T>
  static void _Copy(const void* src, void* dst, size_t n) {
    const T* typed_src = static_cast<const T*>(src);
    T* typed_dst = static_cast<T*>(dst);
    for (int i = 0; i < n; ++i) {
      typed_dst[i] = typed_src[i];
    }
  }

  /**
   * A placeholder function for types that do not allow assignment.
   */
  template <typename T>
  static void
  _CopyNotAllowed(const void* /*src*/, void* /*dst*/, size_t /*n*/) {
    std::cerr << "Type " << DemangleType<T>() << " does not allow assignment.";
    // This is an error by design, so we will quit loud.
    abort();
  }

  /**
   * Destructor for non-fundamental types.
   */
  template <typename T>
  static void _Dtor(void* ptr, size_t n) {
    T* typed_ptr = static_cast<T*>(ptr);
    for (int i = 0; i < n; ++i) {
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

  template <
      typename T,
      typename std::enable_if<
          !(std::is_fundamental<T>::value || std::is_pointer<T>::value) &&
          std::is_copy_assignable<T>::value>::type* = nullptr>
  static TypeMeta Make() {
    return TypeMeta(Id<T>(), ItemSize<T>(), _Ctor<T>, _Copy<T>, _Dtor<T>);
  }

  template <typename T>
  static TypeMeta Make(
      typename std::enable_if<
          !(std::is_fundamental<T>::value || std::is_pointer<T>::value) &&
          !std::is_copy_assignable<T>::value>::type* = 0) {
    return TypeMeta(
        Id<T>(), ItemSize<T>(), _Ctor<T>, _CopyNotAllowed<T>, _Dtor<T>);
  }

 private:
  CaffeTypeId id_;
  size_t itemsize_;
  PlacementNew ctor_;
  TypedCopy copy_;
  TypedDestructor dtor_;
};

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
// Implementation note: in MSVC, we will need to prepend the CAFFE2_EXPORT
// keyword in order to get things compiled properly. in Linux, gcc seems to
// create attribute ignored error for explicit template instantiations, see
//   http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0537r0.html
//   https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51930
// and as a result, we define these two macros slightly differently.

#ifdef _MSC_VER
#define CAFFE_KNOWN_TYPE(T)                              \
  template <>                                            \
  CAFFE2_EXPORT CaffeTypeId TypeMeta::Id<T>() {          \
    static bool type_id_bit[1];                          \
    static TypeNameRegisterer<T> registerer(             \
        reinterpret_cast<CaffeTypeId>(type_id_bit), #T); \
    return reinterpret_cast<CaffeTypeId>(type_id_bit);   \
  }
#else // _MSC_VER
#define CAFFE_KNOWN_TYPE(T)                              \
  template <>                                            \
  CaffeTypeId TypeMeta::Id<T>() {                        \
    static bool type_id_bit[1];                          \
    static TypeNameRegisterer<T> registerer(             \
        reinterpret_cast<CaffeTypeId>(type_id_bit), #T); \
    return reinterpret_cast<CaffeTypeId>(type_id_bit);   \
  }
#endif

} // namespace caffe2

#endif // CAFFE2_CORE_TYPEID_H_
