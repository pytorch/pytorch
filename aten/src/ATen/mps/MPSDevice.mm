//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSDevice.h>
#include <torch/library.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/MathBitsFallback.h>
#include <ATen/native/MathBitFallThroughLists.h>

namespace at {
namespace mps {

static std::unique_ptr<MPSDevice> mps_device;
static std::once_flag mpsdev_init;

MPSDevice* MPSDevice::getInstance() {
  std::call_once(mpsdev_init, [] {
      mps_device = std::unique_ptr<MPSDevice>(new MPSDevice());
  });
  return mps_device.get();
}

MPSDevice::~MPSDevice() {
  [_mtl_device release];
  _mtl_device = nil;
}

MPSDevice::MPSDevice() {
  NSArray* devices = MTLCopyAllDevices();
  for (unsigned long i = 0 ; i < [devices count] ; i++) {
    id<MTLDevice>  device = devices[i];
    if(![device isLowPower]) { // exclude Intel GPUs
      _mtl_device = device;
      break;
    }
  }
  assert(_mtl_device);
}

at::Allocator* getMPSSharedAllocator();
at::Allocator* GetMPSAllocator(bool useSharedAllocator) {
  return useSharedAllocator ? getMPSSharedAllocator() : GetAllocator(DeviceType::MPS);
}

} // namespace mps

TORCH_LIBRARY_IMPL(aten, MPS, m) {
  m.impl("bitwise_and.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&native::cpu_fallback>());
  m.impl("embedding_renorm_", torch::CppFunction::makeFromBoxedFunction<&native::cpu_fallback>());
  m.impl("linalg_svd", torch::CppFunction::makeFromBoxedFunction<&native::cpu_fallback>());
  m.impl("linalg_svd.U", torch::CppFunction::makeFromBoxedFunction<&native::cpu_fallback>());
  m.impl("repeat_interleave.Tensor", torch::CppFunction::makeFromBoxedFunction<&native::cpu_fallback>());
  m.impl("repeat_interleave.self_Tensor", torch::CppFunction::makeFromBoxedFunction<&native::cpu_fallback>());
  m.impl("repeat_interleave.self_int", torch::CppFunction::makeFromBoxedFunction<&native::cpu_fallback>());
  m.impl("_fft_c2c", torch::CppFunction::makeFromBoxedFunction<&native::cpu_fallback>());
  m.impl("_fft_r2c", torch::CppFunction::makeFromBoxedFunction<&native::cpu_fallback>());
  m.impl("linalg_vector_norm", torch::CppFunction::makeFromBoxedFunction<&native::cpu_fallback>());
  m.impl("sgn.out", torch::CppFunction::makeFromBoxedFunction<&native::cpu_fallback>());
  m.impl("nonzero", torch::CppFunction::makeFromBoxedFunction<&native::cpu_fallback>());
  m.impl("masked_select", torch::CppFunction::makeFromBoxedFunction<&native::cpu_fallback>());
}

} // namespace at
