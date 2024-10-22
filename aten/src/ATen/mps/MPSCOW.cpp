//  Copyright © 2024 Apple Inc.

#include <ATen/mps/MPSCOW.h>

#include <ATen/detail/MPSHooksInterface.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSCOWContext.h>
#include <ATen/mps/MPSDevice.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/impl/COWDeleter.h>

namespace at::mps::cow {
namespace {
DispatchKeySet ChangeDispatchKeySetBackend(
    DispatchKeySet original,
    BackendComponent new_backend) {
  BackendComponent old_backend = original.highestBackendKey();

  // following logic TensorImpl::TensorImpl, update the BackendComponent related
  // keys to correspond to device

  // TODO: Autocoast should be a per-backend functionality key, once that change
  // is made this key swap will not be necessary.
  auto new_key_set =
      original - c10::getAutocastRelatedKeySetFromBackend(old_backend);
  new_key_set =
      new_key_set | c10::getAutocastRelatedKeySetFromBackend(new_backend);

  // See note [Removing keys from DispatchKeySet Only Affects Functionality
  // Keys]
  new_key_set = new_key_set.remove_backend(old_backend);
  return new_key_set | DispatchKeySet(new_backend);
}

template <typename T>
static inline bool is_null_or_equal_to(
    const std::optional<T>& test,
    const T& value) {
  if (!test.has_value()) {
    return true;
  }
  return test.value() == value;
}
} // namespace

c10::intrusive_ptr<c10::TensorImpl> lazy_cloned_tensor_for_unified_memory(
    Tensor const& self,
    std::optional<c10::Device> device,
    c10::intrusive_ptr<c10::StorageImpl> lazy_cloned_storage) {
  // For _lazy_clone with device, the tensor itself should be on either MPS or
  // CPU.
  TORCH_CHECK(self.device().is_mps() || self.device().is_cpu());
  // The target device should be either MPS or CPU.
  TORCH_CHECK(
      device == std::nullopt ||
      (device.value().is_mps() || device.value().is_cpu()));

  c10::Device target_device =
      device == std::nullopt ? self.device() : device.value();

  auto device_key_set = self.key_set();

  bool mps_to_cpu = self.device().is_mps() && target_device.is_cpu();
  bool cpu_to_mps = self.device().is_cpu() && target_device.is_mps();

  at::mps::IMPSAllocator* mps_alloc = at::mps::getIMPSAllocator();
  bool is_shared_allocator_and_unified_memory =
      mps_alloc->isSharedStorageSupported() &&
      mps_alloc->isSharedAllocatorUsage();
  // This will not work if our default is not MPS SharedAllocator and the
  // hardware does not have unified memory architecture.
  TORCH_CHECK(is_shared_allocator_and_unified_memory);

  if (mps_to_cpu || cpu_to_mps) {
    // Get the COW context.
    at::DataPtr& data_ptr = lazy_cloned_storage->_mutable_data_ptr_no_checks();
    TORCH_INTERNAL_ASSERT(data_ptr);

    auto* ctx = data_ptr.cast_context<c10::impl::cow::COWDeleterContext>(
        c10::impl::cow::cow_deleter);
    TORCH_INTERNAL_ASSERT(ctx != nullptr);

    at::Allocator* allocator = nullptr;
    BackendComponent backend_component = BackendComponent::InvalidBit;
    void* target_device_data_ptr = nullptr;

    // If the underlying data context is not UnifiedMemoryDataPtrContext
    if (ctx->GetDataDeleter() !=
        c10::impl::cow::unified_memory_data_ptr_ctx_deleter) {
      // Wrap the underlying data context as UnifiedMemoryDataPtrContext.
      ctx->WrapDataPtr(
          mps_to_cpu ? at::mps::cow::WrapMPSToCPU : at::mps::cow::WrapCPUToMPS,
          lazy_cloned_storage->nbytes());
    }
    TORCH_INTERNAL_ASSERT(
        ctx->GetDataDeleter() ==
        c10::impl::cow::unified_memory_data_ptr_ctx_deleter);
    auto* unified_data_ptr_ctx =
        reinterpret_cast<const c10::impl::cow::UnifiedMemoryDataPtrContext*>(
            ctx->GetConstDataPtr());

    if (mps_to_cpu) {
      backend_component = BackendComponent::CPUBit;
      allocator = GetCPUAllocator();

      // CPUTensor.to("mps").to("cpu")
      if (unified_data_ptr_ctx->memory_backed_by_cpu()) {
        target_device_data_ptr = unified_data_ptr_ctx->get_original_data_ctx();
      } else { // MPSTensor.to("cpu")
        target_device_data_ptr = unified_data_ptr_ctx->get_mapped_data_ctx();
      }
    } else {
      TORCH_INTERNAL_ASSERT(cpu_to_mps);

      backend_component = BackendComponent::MPSBit;
      allocator = at::mps::GetMPSAllocator();

      // MPSTensor.to("cpu").to("mps")
      if (!unified_data_ptr_ctx->memory_backed_by_cpu()) {
        target_device_data_ptr = unified_data_ptr_ctx->get_original_data_ctx();
      } else { // CPUTensor.to("mps")
        target_device_data_ptr = unified_data_ptr_ctx->get_mapped_data_ctx();
      }
    }

    TORCH_INTERNAL_ASSERT(allocator != nullptr);
    TORCH_INTERNAL_ASSERT(backend_component != BackendComponent::InvalidBit);
    // target_device_data_ptr might be nullptr when encountering an empty tensor

    // Convert the DispatchKeySet from the original backend to the desired
    // backend
    device_key_set =
        ChangeDispatchKeySetBackend(device_key_set, backend_component);
    if (mps_to_cpu) {
      // TODO(Frank): This felt like an overkill for what's necessary. Maybe
      // there is a way to synchronize only the Tensor event. However, we have
      // to have a synchronization in order to have the MPS content reflects
      // correctly on the CPU.
      at::detail::getMPSHooks().deviceSynchronize();
    }

    // Increment the reference count. The old DataPtr will be deleted and thus
    // the refcount is going to go down. We need to increment the refcount in
    // order to maintain correctness.
    ctx->increment_refcount();

    // Create mapped data pointer with COW Context
    DataPtr dp{
        target_device_data_ptr,
        ctx,
        c10::impl::cow::cow_deleter,
        device.value()};

    lazy_cloned_storage->set_data_ptr_noswap(std::move(dp));
    lazy_cloned_storage->set_allocator(
        allocator); // allocator used for materialization
  }

  auto tensor = c10::make_intrusive<c10::TensorImpl>(
      c10::Storage(std::move(lazy_cloned_storage)),
      device_key_set,
      self.dtype());

  return tensor;
}

bool to_will_cow(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    bool copy,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  auto memory_format = optional_memory_format.value_or(MemoryFormat::Preserve);
  // Base case for dtype, layout, memory_format and whether copy or not.
  if (!(is_null_or_equal_to(dtype, self.dtype().toScalarType()) &&
        is_null_or_equal_to(layout, self.layout()) && !copy &&
        (memory_format == MemoryFormat::Preserve ||
         self.suggest_memory_format() == memory_format))) {
    return false;
  }

  if (!(
          // If tensor itself is not on CPU or MPS, and the target device is not
          // CPU or MPS, we will not perform COW.
          (self.device().is_cpu() || self.device().is_mps()) &&
          // The target device has to be explicitly specified in order to
          // perform COW.
          (device != std::nullopt &&
           (device.value().is_cpu() || device.value().is_mps())))) {
    return false;
  }

  auto* mps_alloc =
      reinterpret_cast<at::mps::IMPSAllocator*>(at::mps::GetMPSAllocator());
  bool is_shared_allocator_and_unified_memory =
      mps_alloc->isSharedStorageSupported() &&
      mps_alloc->isSharedAllocatorUsage();
  // SharedAllocator and unified memory is a must to enable COW for MPS.
  if (!is_shared_allocator_and_unified_memory) {
    return false;
  }

  const auto* storage = self.storage().unsafeGetStorageImpl();
  // If we do not have a storage object, we can't perform COW.
  if (!storage) {
    return false;
  }
  // If the storage allocator is something we do not know how to handle.
  // Eg. tensor created from torch.from_numpy, since we do not own the
  // CPU storage of the numpy array.
  const auto* storage_allocator = storage->allocator();
  if (!(storage_allocator &&
        (storage_allocator == mps_alloc ||
         storage_allocator == GetCPUAllocator()))) {
    return false;
  }

  const c10::DataPtr& data_ptr = storage->data_ptr();
  // If we have an empty tensor, or the data pointer is complicated.
  if (!(data_ptr.get() &&
        (c10::impl::cow::has_simple_data_ptr(*storage) ||
         c10::impl::cow::is_cow_data_ptr(data_ptr)))) {
    return false;
  }

  // Case 1: MPSTensor.to("cpu")
  if (self.device().is_mps() && (device != std::nullopt && device->is_cpu())) {
    return true;
    // Case 2: CPUTensor.to("mps")
  } else if (
      self.device().is_cpu() && (device != std::nullopt && device->is_mps())) {
    return true;
  } else {
    return false;
  }
}

} // namespace at::mps::cow
