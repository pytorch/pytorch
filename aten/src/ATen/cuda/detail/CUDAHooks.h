#include <ATen/detail/CUDAHooksInterface.h>

#include <ATen/Generator.h>

// TODO: No need to have this whole header, we can just put it all in
// the cpp file

namespace at { namespace cuda { namespace detail {

// The real implementation of CUDAHooksInterface
struct CUDAHooks : public at::CUDAHooksInterface {
  CUDAHooks(at::CUDAHooksArgs) {}
  std::unique_ptr<THCState, void(*)(THCState*)> initCUDA() const override;
  Device getDeviceFromPtr(void* data) const override;
  Generator* getDefaultCUDAGenerator(DeviceIndex device_index = -1) const override;
  bool hasCUDA() const override;
  bool hasMAGMA() const override;
  bool hasCuDNN() const override;
  const at::cuda::NVRTC& nvrtc() const override;
  int64_t current_device() const override;
  bool hasPrimaryContext(int64_t device_index) const override;
  Allocator* getPinnedMemoryAllocator() const override;
  bool compiledWithCuDNN() const override;
  bool compiledWithMIOpen() const override;
  bool supportsDilatedConvolutionWithCuDNN() const override;
  bool supportsDepthwiseConvolutionWithCuDNN() const override;
  long versionCuDNN() const override;
  std::string showConfig() const override;
  double batchnormMinEpsilonCuDNN() const override;
  int64_t cuFFTGetPlanCacheMaxSize(int64_t device_index) const override;
  void cuFFTSetPlanCacheMaxSize(int64_t device_index, int64_t max_size) const override;
  int64_t cuFFTGetPlanCacheSize(int64_t device_index) const override;
  void cuFFTClearPlanCache(int64_t device_index) const override;
  int getNumGPUs() const override;
};

}}} // at::cuda::detail
