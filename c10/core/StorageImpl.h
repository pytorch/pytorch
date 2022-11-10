#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymInt.h>

#include <c10/util/intrusive_ptr.h>

#include <mutex>

namespace c10 {

namespace detail {

// Safety condition for this is that the mutex must be GUARANTEED not to
// be held when a move occurs.  Storages are moved very rarely (only by
// static runtime
struct UnsafeMovableMutexForStorage {
  std::mutex mutex_;
  // The move constructors just don't actually move the mutex.  Invariant
  // is that moving out of a storage cannot happen concurrently with mutex
  // operations (this is only copy_on_write at the moment).  This would be
  // a read-write race anyway, and NOT ALLOWED.
  //
  // This also means you do NOT need to take out the mutex when moving;
  // there is already a precondition that you're not racing with reads anyway.
  UnsafeMovableMutexForStorage() {}
  UnsafeMovableMutexForStorage(UnsafeMovableMutexForStorage&& a) {}
  UnsafeMovableMutexForStorage& operator=(UnsafeMovableMutexForStorage&&) {
    return *this;
  }
};

} // namespace detail

// A storage represents the underlying backing data buffer for a
// tensor.  This concept was inherited from the original Torch7
// codebase; we'd kind of like to get rid of the concept
// (see https://github.com/pytorch/pytorch/issues/14797) but
// it's hard work and no one has gotten around to doing it.
//
// NB: storage is supposed to uniquely own a data pointer; e.g.,
// two non-null data pointers alias if and only if they are from
// the same storage.  Technically you can violate this invariant
// (e.g., you can create a non-owning StorageImpl with at::from_blob)
// but a lot of things won't work correctly, including:
//
// - An ordinary deleter on such a storage is wrong, because normal deleters
//   assume unique ownership, but if you have two storages at the same data,
//   that implies there is some sort of shared ownership. So your deleter would
//   have to actually be internally doing some sort of refcount thing
// - Deepcopy in Python side relies on storage equality and not data pointer
//   equality; so if there are two separate storages pointing to the same data,
//   the data will actually get duplicated in that case (one data ptr before,
//   two data ptrs after)
// - Version counts won't work correctly, because we do all VC tracking at the
//   level of storages (unless you explicitly disconnect the VC with detach);
//   mutation because data pointers are the same are totally untracked
struct C10_API StorageImpl : public c10::intrusive_ptr_target {
 public:
  struct use_byte_size_t {};

  StorageImpl(
      use_byte_size_t /*use_byte_size*/,
      SymInt size_bytes,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable)
      : data_ptr_(std::move(data_ptr)),
        size_bytes_(std::move(size_bytes)),
        size_bytes_is_symbolic_(size_bytes_.is_symbolic()),
        resizable_(resizable),
        received_cuda_(false),
        allocator_(allocator) {
    if (resizable) {
      TORCH_INTERNAL_ASSERT(
          allocator_, "For resizable storage, allocator must be provided");
    }
  }

  StorageImpl(
      use_byte_size_t /*use_byte_size*/,
      SymInt size_bytes,
      at::Allocator* allocator,
      bool resizable)
      : StorageImpl(
            use_byte_size_t(),
            size_bytes,
            size_bytes.is_symbolic()
                ? allocator->allocate(0)
                : allocator->allocate(size_bytes.as_int_unchecked()),
            allocator,
            resizable) {}

  StorageImpl& operator=(StorageImpl&& other) = default;
  StorageImpl& operator=(const StorageImpl&) = delete;
  StorageImpl() = delete;
  StorageImpl(StorageImpl&& other) = default;
  StorageImpl(const StorageImpl&) = delete;
  ~StorageImpl() override = default;

  void reset() {
    data_ptr_.clear();
    size_bytes_ = 0;
    size_bytes_is_symbolic_ = false;
  }

  template <typename T>
  inline T* data() const {
    return unsafe_data<T>();
  }

  template <typename T>
  inline T* unsafe_data() const {
    return static_cast<T*>(this->data_ptr_.get());
  }

  // Destructor doesn't call release_resources because it's
  // unnecessary; don't forget to change that if needed!
  void release_resources() override {
    data_ptr_.clear();
  }

  size_t nbytes() const {
    TORCH_CHECK(!size_bytes_is_symbolic_);
    return size_bytes_.as_int_unchecked();
  }

  SymInt sym_nbytes() const {
    return size_bytes_;
  }

  // TODO: remove later
  void set_nbytes(size_t size_bytes) {
    size_bytes_ = size_bytes;
    size_bytes_is_symbolic_ = false;
  }

  void set_nbytes(c10::SymInt size_bytes) {
    size_bytes_ = size_bytes;
  }

  bool resizable() const {
    return resizable_;
  };

  at::DataPtr& data_ptr() {
    return data_ptr_;
  };

  const at::DataPtr& data_ptr() const {
    return data_ptr_;
  };

  // Returns the previous data_ptr
  at::DataPtr set_data_ptr(at::DataPtr&& data_ptr) {
    at::DataPtr old_data_ptr(std::move(data_ptr_));
    data_ptr_ = std::move(data_ptr);
    return old_data_ptr;
  };

  void set_data_ptr_noswap(at::DataPtr&& data_ptr) {
    data_ptr_ = std::move(data_ptr);
  }

  // TODO: Return const ptr eventually if possible
  void* data() {
    return data_ptr_.get();
  }

  void* data() const {
    return data_ptr_.get();
  }

  at::DeviceType device_type() const {
    return data_ptr_.device().type();
  }

  at::Allocator* allocator() {
    return allocator_;
  }

  const at::Allocator* allocator() const {
    return allocator_;
  };

  // You generally shouldn't use this method, but it is occasionally
  // useful if you want to override how a tensor will be reallocated,
  // after it was already allocated (and its initial allocator was
  // set)
  void set_allocator(at::Allocator* allocator) {
    allocator_ = allocator;
  }

  Device device() const {
    return data_ptr_.device();
  }

  void set_resizable(bool resizable) {
    if (resizable) {
      // We need an allocator to be resizable
      AT_ASSERT(allocator_);
    }
    resizable_ = resizable;
  }

  /**
   * Can only be called when use_count is 1
   */
  void UniqueStorageShareExternalPointer(
      void* src,
      size_t size_bytes,
      DeleterFnPtr d = nullptr) {
    UniqueStorageShareExternalPointer(
        at::DataPtr(src, src, d, data_ptr_.device()), size_bytes);
  }

  /**
   * Can only be called when use_count is 1
   */
  void UniqueStorageShareExternalPointer(
      at::DataPtr&& data_ptr,
      size_t size_bytes) {
    data_ptr_ = std::move(data_ptr);
    size_bytes_ = size_bytes;
    size_bytes_is_symbolic_ = false;
    allocator_ = nullptr;
    resizable_ = false;
  }

  // This method can be used only after storage construction and cannot be used
  // to modify storage status
  void set_received_cuda(bool received_cuda) {
    received_cuda_ = received_cuda;
  }

  bool received_cuda() {
    return received_cuda_;
  }

  void set_warn_on_write(bool warn_on_write);

  // NB: This should only be retrieved in circumstances where you are
  // logically writing to the Storage; otherwise you may be subject
  // to a read-write race.
  bool warn_on_write() const {
    return warn_on_write_;
  }

  // virtual to deal with FunctionalStorageImpl, sigh
  // NB: This is morally const but we're not const correct and I will
  // need to monkey around data_ptr_ so it was easier to not declare it const
  virtual c10::intrusive_ptr<StorageImpl> copy_on_write();

 private:
  DataPtr data_ptr_;
  SymInt size_bytes_;
  bool size_bytes_is_symbolic_;
  bool resizable_;
  // Identifies that Storage was received from another process and doesn't have
  // local to process cuda memory allocation
  bool received_cuda_;
  // If this storage is written to, raise a warning saying that writes to
  // this storage have behavior that could potentially change in the future.
  // This can be changed on read operations (specifically, when you take out a
  // reshape-induced view on a storage) and so writes to the field must be
  // guarded by copy_on_write_mutex_ (reads to this field need not be guarded,
  // as we only read the field on writes, which would already race with a
  // reshape-induced view read.)
  //
  // This warning is a little too aggressive.  There is no problem if you
  // write to the input/output of a reshape, where the output/input is never
  // used again.  We can cheaply test a sub-case of this, where the
  // output/input is dead, but it's not clear to me it's worth implementing;
  // if the output/input is never used again but still alive, we would still
  // trigger, and we could only detect this situation by delaying the warning
  // to reads (which would require a big pile of machinery at read-side,
  // whereas currently we only need to modify ADInplaceOrView.)
  bool warn_on_write_;
  Allocator* allocator_;
  detail::UnsafeMovableMutexForStorage copy_on_write_mutex_;
};
} // namespace c10
