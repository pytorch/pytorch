#pragma once
#include <ATen/TypeDefault.h>
#include <ATen/Context.h>
#include <ATen/CPUGenerator.h>

namespace at {

struct CPUTypeDefault : public TypeDefault {
  CPUTypeDefault(TensorTypeId type_id, bool is_variable, bool is_undefined)
      : TypeDefault(type_id, is_variable, is_undefined) {}
  Allocator* allocator() const override {
    return getCPUAllocator();
  }
  Device getDeviceFromPtr(void * data) const {
    return DeviceType::CPU;
  }

  std::unique_ptr<Generator> generator() const override {
    return std::unique_ptr<Generator>(new CPUGenerator(&at::globalContext()));
  }
};

} // namespace at
