#pragma once
#include <ATen/TypeDefault.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/CUDAGenerator.h>

namespace at {

struct CUDATypeDefault : public TypeDefault {
  CUDATypeDefault(TensorTypeId type_id, bool is_variable, bool is_undefined)
      : TypeDefault(type_id, is_variable, is_undefined) {}

  Allocator* allocator() const override {
    return cuda::getCUDADeviceAllocator();
  }
  Device getDeviceFromPtr(void * data) const {
    return cuda::getDeviceFromPtr(data);
  }
  std::unique_ptr<Generator> generator() const override {
    return std::unique_ptr<Generator>(new CUDAGenerator(&at::globalContext()));
  }
};

} // namespace at
