#pragma once

#include <cstddef>
#include <sstream>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include <c10/util/intrusive_ptr.h>
#include <c10/util/typeid.h>
#include <c10/macros/Macros.h>

namespace caffe2 {

class Tensor;

/**
 * @brief Blob is a general container that hosts a typed pointer.
 *
 * A Blob hosts a pointer as well as its type, and takes charge of deleting it
 * properly when the blob is deallocated or re-allocated with a new type. A blob
 * could contain anything, although the most common case is to contain a Tensor.
 */
class CAFFE2_API Blob final : public c10::intrusive_ptr_target {
 public:
  /**
   * Initializes an empty Blob.
   */
  Blob() noexcept : meta_(), pointer_(nullptr), has_ownership_(false) {}
  ~Blob() {
    Reset();
  }

  Blob(Blob&& other) noexcept : Blob() {
    swap(other);
  }

  Blob& operator=(Blob&& other) noexcept {
    Blob(std::move(other)).swap(*this);
    return *this;
  }

  /**
   * Checks if the content stored in the blob is of type T.
   */
  template <class T>
  bool IsType() const noexcept {
    return meta_.Match<T>();
  }

  /**
   * Returns the meta info of the blob.
   */
  const TypeMeta& meta() const noexcept {
    return meta_;
  }

  /**
   * Returns a printable typename of the blob.
   */
  const char* TypeName() const noexcept {
    return meta_.name();
  }

  /**
   * @brief Gets the const reference of the stored object. The code checks if
   * the stored object is of the desired type.
   */
  // TODO(jerryzh): add a Get(DeviceType) function?
  template <class T>
  const T& Get() const {
    AT_ASSERTM(
        IsType<T>(),
        "wrong type for the Blob instance. Blob contains ",
        meta_.name(),
        " while caller expects ",
        TypeMeta::TypeName<T>());
    // TODO: after we add Get<Tensor>(DeviceType)
    // and changed all the callsites, we can add
    // a static assert here to enforce T != Tensor
    return *static_cast<const T*>(pointer_);
  }

  const void* GetRaw() const noexcept {
    return pointer_;
  }
  void* GetRaw() noexcept {
    return pointer_;
  }

  /**
   * @brief Gets a mutable pointer to the stored object.
   *
   * If the current object is not of the right type, a new object is created
   * and the old object is freed. Note that type T should have a default
   * constructor. Otherwise, create the object yourself first, and use
   * Reset().
   */
  template <class T>
  T* GetMutable() {
    static_assert(
        std::is_default_constructible<T>::value,
        "GetMutable can't be called with non-default-constructible types. "
        "Try using specialized methods");
    if (IsType<T>()) {
      return static_cast<T*>(pointer_);
    } else {
      // TODO Re-enable logging
      // VLOG(1) << "Create new mutable object " << TypeMeta::TypeName<T>();
      return Reset<T>(new T());
    }
  }

  template <class T>
  T* GetMutableOrNull() {
    if (IsType<T>()) {
      return static_cast<T*>(pointer_);
    } else {
      return nullptr;
    }
  }

  /**
   * Sets the underlying object to the allocated one. The Blob then takes over
   * the ownership of the passed in pointer. If there is already an object in
   * the Blob, the old object is freed.
   *
   * This is used when the underlying class T does not have a default ctor, or
   * complex initializations needs to be done outside the blob.
   */
  template <class T>
  T* Reset(T* allocated) {
    free_();
    meta_ = TypeMeta::Make<T>();
    pointer_ = static_cast<void*>(allocated);
    has_ownership_ = true;
    return allocated;
  }

  /**
   * Sets the underlying object to the allocated one, but does not take over
   * the ownership of the passed in pointer. If there is already an object in
   * the Blob, the old object is freed.
   *
   * Unlike Reset, this does not take over the ownership of the pointer and the
   * caller is responsible for making sure that the lifetime of the allocated
   * blob outlasts the lifetime of any access to this blob, until another Reset
   * call is made or the blob is destructed.
   */
  template <class T>
  typename std::remove_const<T>::type* ShareExternal(
      typename std::remove_const<T>::type* allocated) {
    return static_cast<T*>(ShareExternal(
        static_cast<void*>(allocated),
        TypeMeta::Make<typename std::remove_const<T>::type>()));
  }

  // TODO Remove ShareExternal() and have Blob always own its content
  void* ShareExternal(void* allocated, const TypeMeta& meta) {
    free_();
    meta_ = meta;
    pointer_ = static_cast<void*>(allocated);
    has_ownership_ = false;
    return allocated;
  }

  /**
   * Resets the Blob to an empty one.
   */
  void Reset() {
    free_();
    pointer_ = nullptr;
    meta_ = TypeMeta();
    has_ownership_ = false;
  }

  /**
   * @brief Swaps the underlying storage of two blobs.
   */
  void swap(Blob& rhs) {
    using std::swap;
    swap(meta_, rhs.meta_);
    swap(pointer_, rhs.pointer_);
    swap(has_ownership_, rhs.has_ownership_);
  }

 private:
  void free_() {
    if (has_ownership_) {
      AT_ASSERTM(pointer_ != nullptr, "Can't have ownership of nullptr");
      (*meta_.deleteFn())(pointer_);
    }
  }

  TypeMeta meta_;
  void* pointer_ = nullptr;
  bool has_ownership_ = false;

  C10_DISABLE_COPY_AND_ASSIGN(Blob);
};

inline void swap(Blob& lhs, Blob& rhs) {
  lhs.swap(rhs);
}

inline std::ostream& operator<<(std::ostream& out, const Blob& v) {
  return out << "Blob[" << v.TypeName() << "]";
}

} // namespace caffe2
