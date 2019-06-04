#pragma once
#include <ATen/TypeDefault.h>
#include <ATen/cuda/ATenCUDAGeneral.h>

namespace at {

struct AT_CUDA_API CUDATypeDefault : public TypeDefault {
  CUDATypeDefault(TensorTypeId type_id, bool is_variable, bool is_undefined)
      : TypeDefault(type_id, is_variable, is_undefined) {}

  Allocator* allocator() const override;
  Device getDeviceFromPtr(void * data) const override;
  std::unique_ptr<Generator> generator() const;
};

} // namespace at
