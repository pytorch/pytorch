#pragma once

#include <ATen/detail/CUDAHooksInterface.h>

#include <ATen/Generator.h>

// TODO: No need to have this whole header, we can just put it all in
// the cpp file

namespace at::cuda::detail {

// Set the callback to initialize Magma, which is set by
// torch_cuda_cu. This indirection is required so magma_init is called
// in the same library where Magma will be used.
TORCH_CUDA_CPP_API void set_magma_init_fn(void (*magma_init_fn)());


// The real implementation of CUDAHooksInterface
struct CUDAHooks : public at::CUDAHooksInterface {
  CUDAHooks(at::CUDAHooksArgs) {}
  void init() const override;
  Device getDeviceFromPtr(void* data) const override;
  bool isPinnedPtr(const void* data) const override;
  const Generator& getDefaultGenerator(
      DeviceIndex device_index = -1) const override;
  Generator getNewGenerator(
      DeviceIndex device_index = -1) const override;
  bool hasCUDA() const override;
  bool hasMAGMA() const override;
  bool hasCuDNN() const override;
  bool hasCuSOLVER() const override;
  bool hasCuBLASLt() const override;
  bool hasROCM() const override;
  const at::cuda::NVRTC& nvrtc() const override;
  DeviceIndex current_device() const override;
  bool hasPrimaryContext(DeviceIndex device_index) const override;
  Allocator* getCUDADeviceAllocator() const override;
  Allocator* getPinnedMemoryAllocator() const override;
  bool compiledWithCuDNN() const override;
  bool compiledWithMIOpen() const override;
  bool supportsDilatedConvolutionWithCuDNN() const override;
  bool supportsDepthwiseConvolutionWithCuDNN() const override;
  bool supportsBFloat16ConvolutionWithCuDNNv8() const override;
  bool hasCUDART() const override;
  long versionCUDART() const override;
  long versionCuDNN() const override;
  std::string showConfig() const override;
  double batchnormMinEpsilonCuDNN() const override;
  int64_t cuFFTGetPlanCacheMaxSize(DeviceIndex device_index) const override;
  void cuFFTSetPlanCacheMaxSize(DeviceIndex device_index, int64_t max_size) const override;
  int64_t cuFFTGetPlanCacheSize(DeviceIndex device_index) const override;
  void cuFFTClearPlanCache(DeviceIndex device_index) const override;
  int getNumGPUs() const override;
  DeviceIndex deviceCount() const override;
  DeviceIndex getCurrentDevice() const override;

#ifdef USE_ROCM
  bool isGPUArch(DeviceIndex device_index, const std::vector<std::string>& archs) const override;
#endif
  void deviceSynchronize(DeviceIndex device_index) const override;
  std::tuple<size_t, size_t, ptrdiff_t, std::string, std::string, std::string, uint64_t, bool>
  StorageShareDevice(const c10::Storage& storage) const override;
  c10::DataPtr StorageNewSharedDevice(c10::DeviceIndex device,
                                      bool event_sync_required,
                                      std::string s_ipc_event_handle,
                                      std::string s_handle,
                                      std::string ref_counter_handle,
                                      ptrdiff_t ref_counter_offset,
                                      ptrdiff_t storage_offset_bytes) const override;
  int64_t getIpcRefCounterFileSize() const override;
};

} // at::cuda::detail
