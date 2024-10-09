#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/SymInt.h>
#include <c10/macros/Export.h>
#include <c10/util/Exception.h>
#include <c10/util/ExclusivelyOwned.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/intrusive_ptr.h>
#include <cstddef>
#include <utility>

namespace c10 {

struct IPCHandlePimpl {
  std::string memory_handle;
  std::string ref_counter_handle;
  std::string event_handle;
};

struct Storage;

C10_API bool isSharedStorageAlias(
    const Storage& storage0,
    const Storage& storage1);

struct C10_API Storage {
 public:
  struct use_byte_size_t {};
  struct unsafe_borrow_t {
    explicit unsafe_borrow_t() = default;
  };

  Storage() = default;
  Storage(c10::intrusive_ptr<StorageImpl> ptr)
      : storage_impl_(std::move(ptr)) {}

  // Allocates memory buffer using given allocator and creates a storage with it
  Storage(
      use_byte_size_t /*use_byte_size*/,
      const SymInt& size_bytes,
      Allocator* allocator = nullptr,
      bool resizable = false)
      : storage_impl_(c10::make_intrusive<StorageImpl>(
            StorageImpl::use_byte_size_t(),
            size_bytes,
            allocator,
            resizable)) {}

  // Creates storage with pre-allocated memory buffer. Allocator is given for
  // potential future reallocations, however it can be nullptr if the storage
  // is non-resizable
  Storage(
      use_byte_size_t /*use_byte_size*/,
      size_t size_bytes,
      at::DataPtr data_ptr,
      at::Allocator* allocator = nullptr,
      bool resizable = false)
      : storage_impl_(c10::make_intrusive<StorageImpl>(
            StorageImpl::use_byte_size_t(),
            size_bytes,
            std::move(data_ptr),
            allocator,
            resizable)) {}

 protected:
  explicit Storage(unsafe_borrow_t, const Storage& rhs)
      : storage_impl_(c10::intrusive_ptr<c10::StorageImpl>::reclaim(
            rhs.storage_impl_.get())) {}

  friend MaybeOwnedTraits<Storage>;

 public:
  // Legacy constructor for partially initialized (dtype or memory) storages
  // that can be temporarily created with Caffe2 APIs. See the note on top of
  // TensorImpl.h for details.
  static Storage create_legacy(at::Device device) {
    auto allocator = GetAllocator(device.type());
    return Storage(c10::make_intrusive<StorageImpl>(
        StorageImpl::use_byte_size_t(),
        0,
        allocator->allocate(0), // materialize a non-default Device.
        allocator,
        true));
  }

  // Mimic create_legacy, but without requiring a newly-created StorageImpl.
  void reset_legacy() {
    TORCH_CHECK(resizable() && allocator());
    set_nbytes(0);
    set_data_ptr_noswap(allocator()->allocate(0));
  }

  // TODO: remove later
  void set_nbytes(size_t size_bytes) const {
    storage_impl_->set_nbytes(size_bytes);
  }

  void set_nbytes(c10::SymInt size_bytes) const {
    storage_impl_->set_nbytes(std::move(size_bytes));
  }

  bool resizable() const {
    return storage_impl_->resizable();
  }

  size_t nbytes() const {
    return storage_impl_->nbytes();
  }

  SymInt sym_nbytes() const {
    return storage_impl_->sym_nbytes();
  }
  // get() use here is to get const-correctness

  const void* data() const {
    return storage_impl_->data();
  }

  void* mutable_data() const {
    return storage_impl_->mutable_data();
  }

  at::DataPtr& mutable_data_ptr() const {
    return storage_impl_->mutable_data_ptr();
  }

  const at::DataPtr& data_ptr() const {
    return storage_impl_->data_ptr();
  }

  // Returns the previous data_ptr
  at::DataPtr set_data_ptr(at::DataPtr&& data_ptr) const {
    return storage_impl_->set_data_ptr(std::move(data_ptr));
  }

  void set_data_ptr_noswap(at::DataPtr&& data_ptr) const {
    return storage_impl_->set_data_ptr_noswap(std::move(data_ptr));
  }

  DeviceType device_type() const {
    return storage_impl_->device_type();
  }

  at::Allocator* allocator() const {
    return storage_impl_->allocator();
  }

  at::Device device() const {
    return storage_impl_->device();
  }

  StorageImpl* unsafeReleaseStorageImpl() {
    return storage_impl_.release();
  }

  StorageImpl* unsafeGetStorageImpl() const noexcept {
    return storage_impl_.get();
  }

  c10::weak_intrusive_ptr<StorageImpl> getWeakStorageImpl() const {
    return c10::weak_intrusive_ptr<StorageImpl>(storage_impl_);
  }

  operator bool() const {
    return storage_impl_;
  }

  size_t use_count() const {
    return storage_impl_.use_count();
  }

  inline bool unique() const {
    return storage_impl_.unique();
  }

  bool is_alias_of(const Storage& other) const {
    return (
        storage_impl_ == other.storage_impl_ ||
        isSharedStorageAlias(*this, other));
  }

  void UniqueStorageShareExternalPointer(
      void* src,
      size_t capacity,
      DeleterFnPtr d = nullptr) {
    if (!storage_impl_.unique()) {
      TORCH_CHECK(
          false,
          "UniqueStorageShareExternalPointer can only be called when use_count == 1");
    }
    storage_impl_->UniqueStorageShareExternalPointer(src, capacity, d);
  }

  void UniqueStorageShareExternalPointer(
      at::DataPtr&& data_ptr,
      size_t capacity) {
    if (!storage_impl_.unique()) {
      TORCH_CHECK(
          false,
          "UniqueStorageShareExternalPointer can only be called when use_count == 1");
    }
    storage_impl_->UniqueStorageShareExternalPointer(
        std::move(data_ptr), capacity);
  }

 protected:
  c10::intrusive_ptr<StorageImpl> storage_impl_;
};

template <>
struct MaybeOwnedTraits<c10::Storage> {
  using owned_type = c10::Storage;
  using borrow_type = c10::Storage;

  static borrow_type createBorrow(const owned_type& from) {
    return borrow_type(borrow_type::unsafe_borrow_t{}, from);
  }

  static void assignBorrow(borrow_type& lhs, const borrow_type& rhs) {
    lhs.unsafeReleaseStorageImpl();
    lhs = borrow_type(borrow_type::unsafe_borrow_t{}, rhs);
  }

  static void destroyBorrow(borrow_type& toDestroy) {
    toDestroy.unsafeReleaseStorageImpl(); // "leak" it, but it was already +0.
  }

  static const owned_type& referenceFromBorrow(const borrow_type& borrow) {
    return borrow;
  }

  static const owned_type* pointerFromBorrow(const borrow_type& borrow) {
    return &borrow;
  }

  static bool debugBorrowIsValid(const borrow_type& /*borrow*/) {
    return true;
  }
};

template <>
struct ExclusivelyOwnedTraits<c10::Storage> {
  using repr_type = c10::Storage;
  using pointer_type = c10::Storage*;
  using const_pointer_type = const c10::Storage*;

  static repr_type nullRepr() {
    return c10::Storage();
  }

  template <class... Args>
  static repr_type createInPlace(Args&&... args) {
    return c10::Storage(std::forward<Args>(args)...);
  }

  static repr_type moveToRepr(c10::Storage&& x) {
    return std::move(x);
  }

  static c10::Storage take(c10::Storage& x) {
    return std::move(x);
  }

  static pointer_type getImpl(repr_type& x) {
    return &x;
  }

  static const_pointer_type getImpl(const repr_type& x) {
    return &x;
  }
};

} // namespace c10
