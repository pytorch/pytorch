//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSDevice.h>
#include <torch/library.h>
#include <ATen/native/CPUFallback.h>

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
  bool allowIntelGPUs = false;
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

TORCH_LIBRARY_IMPL(_, MPS, m) {
  static const char *mps_fallback = getenv("PYTORCH_DISABLE_MPS_FALLBACK");
  if(mps_fallback && std::stoi(mps_fallback) == 1) {
    return;
  }
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&native::cpu_fallback>());
}

} // namespace at
