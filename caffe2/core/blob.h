#ifndef CAFFE2_CORE_BLOB_H_
#define CAFFE2_CORE_BLOB_H_

#include <cstddef>
#include <sstream>
#include <typeinfo>
#include <type_traits>
#include <vector>
#include "caffe2/core/common.h"

#include <ATen/core/typeid.h>
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

class Tensor;

/**
 * @brief Blob is a general container that hosts a typed pointer.
 *
 * A Blob hosts a pointer as well as its type, and takes charge of deleting it
 * properly when the blob is deallocated or re-allocated with a new type. A blob
 * could contain anything, although the most common case is to contain a Tensor.
 */
class CAFFE2_API Blob final {
 public:
  using DestroyCall = void(void*);

  /**
   * Initializes an empty Blob.
   */
  Blob() noexcept : meta_(), pointer_(nullptr), destroy_(nullptr) {}
  ~Blob() { Reset(); }

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
  inline const TypeMeta& meta() const noexcept { return meta_; }

  /**
   * Returns a printable typename of the blob.
   */
  inline const char* TypeName() const noexcept { return meta_.name(); }

  /**
   * @brief Gets the const reference of the stored object. The code checks if
   * the stored object is of the desired type.
   */
  // TODO(jerryzh): add a Get(DeviceType) function?
  template <class T>
  const T& Get() const {
    CAFFE_ENFORCE(
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
      VLOG(1) << "Create new mutable object " << TypeMeta::TypeName<T>();
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
    if (pointer_ && destroy_) {
      destroy_(pointer_);
    }
    meta_ = TypeMeta::Make<T>();
    pointer_ = static_cast<void*>(allocated);
    destroy_ = &Destroy<T>;
    return allocated;
  }

  inline void*
  Reset(void* allocated, const TypeMeta& meta, DestroyCall* destroy) {
    if (pointer_ && destroy_) {
      destroy_(pointer_);
    }
    meta_ = meta;
    pointer_ = static_cast<void*>(allocated);
    destroy_ = destroy;
    return allocated;
  }

  /**
   * Releases the ownership, if any, this Blob has on the underlying pointer.
   * The user is then responsible for freeing the data if needed
   */
  inline DestroyCall* Release() {
    DestroyCall* d = destroy_;
    destroy_ = nullptr;
    return d;
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

  void* ShareExternal(void* allocated, const TypeMeta& meta) {
    if (pointer_ && destroy_) {
      destroy_(pointer_);
    }
    meta_ = meta;
    pointer_ = static_cast<void*>(allocated);
    destroy_ = nullptr;
    return allocated;
  }

  /**
   * Resets the Blob to an empty one.
   */
  inline void Reset() {
    if (pointer_ && destroy_) {
      destroy_(pointer_);
    }
    pointer_ = nullptr;
    meta_ = TypeMeta();
    destroy_ = nullptr;
  }

  /**
   * @brief Swaps the underlying storage of two blobs.
   */
  void swap(Blob& rhs) {
    using std::swap;
    swap(meta_, rhs.meta_);
    swap(pointer_, rhs.pointer_);
    swap(destroy_, rhs.destroy_);
  }

 private:
  /**
   * @brief A destroy call that is used to properly deconstruct objects.
   */
  template <class T>
  static void Destroy(void* pointer) {
    delete static_cast<T*>(pointer);
  }
  TypeMeta meta_;
  void* pointer_ = nullptr;
  DestroyCall* destroy_ = nullptr;

  C10_DISABLE_COPY_AND_ASSIGN(Blob);
};

inline void swap(Blob& lhs, Blob& rhs) {
  lhs.swap(rhs);
}

inline bool BlobIsTensorType(const Blob& blob, DeviceType device_type) {
  bool is_match = blob.meta().Match<Tensor>();
  if (!is_match) {
    return false;
  }
  const Tensor* tensor = &blob.Get<Tensor>();
  return tensor && tensor->GetDeviceType() == device_type;
}

inline Tensor* BlobGetMutableTensor(Blob* blob, DeviceType device_type) {
  if (blob->IsType<Tensor>()) {
    Tensor* tensor = blob->GetMutable<Tensor>();
    if (tensor->GetDeviceType() == device_type) {
      return tensor;
    }
  }

  // if we're here, then either Blob didn't hold a Tensor
  // or that Tensor had the wrong DeviceType.
  VLOG(1) << "Create new mutable object " << TypeMeta::TypeName<Tensor>()
          << " DeviceType:" << device_type;
  return blob->Reset<Tensor>(new Tensor(device_type));
}

}  // namespace caffe2
#endif  // CAFFE2_CORE_BLOB_H_
