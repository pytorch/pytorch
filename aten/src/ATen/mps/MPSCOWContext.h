// Copyright © 2024 Apple Inc.
// This file is to support the feature for MPS and CPU to share the same
// underlying memory through Copy-on-write context. Please refer to
// UnifiedMemoryDataPtrContext for comments.

#pragma once

#include <c10/core/impl/COWDeleter.h>
#include <c10/util/UniqueVoidPtr.h>

#include <memory>

namespace at::mps::cow {

class C10_API CPUToMPSDataPtrContext
    : public c10::impl::cow::UnifiedMemoryDataPtrContext {
 public:
  explicit CPUToMPSDataPtrContext(
      std::unique_ptr<void, DeleterFnPtr> original_data,
      size_t nbytes);

  ~CPUToMPSDataPtrContext() override;
  void InitializeMappedData() override;

  std::unique_ptr<void, DeleterFnPtr> move_original_data_ctx() override;
  void* get_original_data_ctx() const override;
  void* get_mapped_data_ctx() const override;
  bool memory_backed_by_cpu() const override {
    return true;
  }

 private:
  std::unique_ptr<void, DeleterFnPtr> original_data_;
  std::unique_ptr<void, DeleterFnPtr> mapped_data_;
  size_t nbytes_;
};

class C10_API MPSToCPUDataPtrContext
    : public c10::impl::cow::UnifiedMemoryDataPtrContext {
 public:
  explicit MPSToCPUDataPtrContext(
      std::unique_ptr<void, DeleterFnPtr> original_data);

  ~MPSToCPUDataPtrContext() override;
  void InitializeMappedData() override;

  std::unique_ptr<void, DeleterFnPtr> move_original_data_ctx() override;
  void* get_original_data_ctx() const override;
  void* get_mapped_data_ctx() const override;
  bool memory_backed_by_cpu() const override {
    return false;
  }

 private:
  std::unique_ptr<void, DeleterFnPtr> original_data_;
  std::unique_ptr<void, DeleterFnPtr> mapped_data_;
};

std::unique_ptr<void, DeleterFnPtr> WrapCPUToMPS(
    std::unique_ptr<void, DeleterFnPtr> data,
    size_t nbytes);
std::unique_ptr<void, DeleterFnPtr> WrapMPSToCPU(
    std::unique_ptr<void, DeleterFnPtr> data,
    size_t nbytes);

} // namespace at::mps::cow
