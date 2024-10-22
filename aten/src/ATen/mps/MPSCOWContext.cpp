//  Copyright © 2024 Apple Inc.

#include <ATen/mps/MPSCOWContext.h>

#include <ATen/mps/MPSAllocatorInterface.h>
#include <c10/core/impl/COWDeleter.h>

#include <memory>

namespace at::mps::cow {
namespace {
void NoDeleter(void*){};
} // namespace

CPUToMPSDataPtrContext::CPUToMPSDataPtrContext(
    std::unique_ptr<void, DeleterFnPtr> original_data,
    size_t nbytes)
    : original_data_(std::move(original_data)),
      mapped_data_(nullptr, nullptr),
      nbytes_(nbytes) {
  InitializeMappedData();
}

CPUToMPSDataPtrContext::~CPUToMPSDataPtrContext() {
  if (mapped_data_) {
    mapped_data_.reset();
  }
  if (original_data_) {
    original_data_.reset();
  }
}

void CPUToMPSDataPtrContext::InitializeMappedData() {
  at::mps::IMPSAllocator* mps_alloc = at::mps::getIMPSAllocator(true);
  bool is_shared_allocator_and_unified_memory =
      mps_alloc->isSharedStorageSupported() &&
      mps_alloc->isSharedAllocatorUsage();
  if (is_shared_allocator_and_unified_memory) {
    mapped_data_ =
        mps_alloc->registerCPUBackedPtr(original_data_.get(), nbytes_)
            .move_context();
  }
}

std::unique_ptr<void, DeleterFnPtr> CPUToMPSDataPtrContext::
    move_original_data_ctx() {
  return std::move(original_data_);
}

void* CPUToMPSDataPtrContext::get_original_data_ctx() const {
  if (original_data_) {
    return original_data_.get();
  } else {
    return nullptr;
  }
}
void* CPUToMPSDataPtrContext::get_mapped_data_ctx() const {
  if (mapped_data_) {
    return mapped_data_.get();
  } else {
    return nullptr;
  }
}

MPSToCPUDataPtrContext::MPSToCPUDataPtrContext(
    std::unique_ptr<void, DeleterFnPtr> original_data)
    : original_data_(std::move(original_data)), mapped_data_(nullptr, nullptr) {
  InitializeMappedData();
}

MPSToCPUDataPtrContext::~MPSToCPUDataPtrContext() {
  if (mapped_data_) {
    mapped_data_.reset();
  }
  if (original_data_) {
    original_data_.reset();
  }
}

void MPSToCPUDataPtrContext::InitializeMappedData() {
  at::mps::IMPSAllocator* mps_alloc = at::mps::getIMPSAllocator(true);
  bool is_shared_allocator_and_unified_memory =
      mps_alloc->isSharedStorageSupported() &&
      mps_alloc->isSharedAllocatorUsage();
  if (is_shared_allocator_and_unified_memory) {
    mapped_data_ = std::unique_ptr<void, DeleterFnPtr>(
        std::get<0>(
            mps_alloc->unsafeGetSharedBufferPtr(get_original_data_ctx())),
        NoDeleter);
  }
}

std::unique_ptr<void, DeleterFnPtr> MPSToCPUDataPtrContext::
    move_original_data_ctx() {
  return std::move(original_data_);
}

void* MPSToCPUDataPtrContext::get_original_data_ctx() const {
  if (original_data_) {
    return original_data_.get();
  } else {
    return nullptr;
  }
}
void* MPSToCPUDataPtrContext::get_mapped_data_ctx() const {
  if (mapped_data_) {
    return mapped_data_.get();
  } else {
    return nullptr;
  }
}

std::unique_ptr<void, DeleterFnPtr> WrapCPUToMPS(
    std::unique_ptr<void, DeleterFnPtr> data,
    size_t nbytes) {
  auto result = std::unique_ptr<void, DeleterFnPtr>(
      new CPUToMPSDataPtrContext(std::move(data), nbytes),
      c10::impl::cow::unified_memory_data_ptr_ctx_deleter);
  return result;
}

std::unique_ptr<void, DeleterFnPtr> WrapMPSToCPU(
    std::unique_ptr<void, DeleterFnPtr> data,
    size_t nbytes) {
  auto result = std::unique_ptr<void, DeleterFnPtr>(
      new MPSToCPUDataPtrContext(std::move(data)),
      c10::impl::cow::unified_memory_data_ptr_ctx_deleter);
  return result;
}

} // namespace at::mps::cow
