#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/SymInt.h>
#include <c10/core/impl/PyObjectSlot.h>

#include <c10/util/intrusive_ptr.h>

namespace c10 {

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
        size_bytes_is_heap_allocated_(size_bytes_.is_heap_allocated()),
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
            size_bytes.is_heap_allocated()
                ? allocator->allocate(0)
                : allocator->allocate(size_bytes.as_int_unchecked()),
            allocator,
            resizable) {}

  StorageImpl& operator=(StorageImpl&& other) = delete;
  StorageImpl& operator=(const StorageImpl&) = delete;
  StorageImpl() = delete;
  StorageImpl(StorageImpl&& other) = delete;
  StorageImpl(const StorageImpl&) = delete;
  ~StorageImpl() override = default;

  void reset() {
    data_ptr_.clear();
    size_bytes_ = 0;
    size_bytes_is_heap_allocated_ = false;
  }

  // Destructor doesn't call release_resources because it's
  // unnecessary; don't forget to change that if needed!
  void release_resources() override {
    data_ptr_.clear();
  }

  size_t nbytes() const {
    // OK to do this instead of maybe_as_int as nbytes is guaranteed positive
    TORCH_CHECK(!size_bytes_is_heap_allocated_);
    return size_bytes_.as_int_unchecked();
  }

  SymInt sym_nbytes() const {
    return size_bytes_;
  }

  // TODO: remove later
  void set_nbytes(size_t size_bytes) {
    size_bytes_ = size_bytes;
    size_bytes_is_heap_allocated_ = false;
  }

  void set_nbytes(c10::SymInt size_bytes) {
    size_bytes_ = std::move(size_bytes);
  }

  bool resizable() const {
    return resizable_;
  }

  at::DataPtr& mutable_data_ptr() {
    return data_ptr_;
  }

  const at::DataPtr& data_ptr() const {
    return data_ptr_;
  }

  // Returns the previous data_ptr
  at::DataPtr set_data_ptr(at::DataPtr&& data_ptr) {
    at::DataPtr old_data_ptr(std::move(data_ptr_));
    data_ptr_ = std::move(data_ptr);
    return old_data_ptr;
  }

  void set_data_ptr_noswap(at::DataPtr&& data_ptr) {
    data_ptr_ = std::move(data_ptr);
  }

  const void* data() const {
    return data_ptr_.get();
  }

  void* mutable_data() {
    return data_ptr_.mutable_get();
  }

  at::DeviceType device_type() const {
    return data_ptr_.device().type();
  }

  at::Allocator* allocator() {
    return allocator_;
  }

  const at::Allocator* allocator() const {
    return allocator_;
  }

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
    size_bytes_is_heap_allocated_ = false;
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

  impl::PyObjectSlot* pyobj_slot() {
    return &pyobj_slot_;
  }

  const impl::PyObjectSlot* pyobj_slot() const {
    return &pyobj_slot_;
  }

 private:
  DataPtr data_ptr_;
  SymInt size_bytes_;
  bool size_bytes_is_heap_allocated_;
  bool resizable_;
  // Identifies that Storage was received from another process and doesn't have
  // local to process cuda memory allocation
  bool received_cuda_;
  Allocator* allocator_;
  impl::PyObjectSlot pyobj_slot_;
};

// Declare StorageImpl create function pointer types.
using StorageImplCreateHelper = intrusive_ptr<StorageImpl> (*)(
    StorageImpl::use_byte_size_t,
    SymInt size_bytes,
    Allocator* allocator,
    bool resizable);

C10_API void SetStorageImplCreate(DeviceType t, StorageImplCreateHelper fptr);

C10_API StorageImplCreateHelper GetStorageImplCreate(DeviceType t);

} // namespace c10
