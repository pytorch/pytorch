#include <ATen/detail/CUDAHooksInterface.h>

#include <ATen/Generator.h>
#include <c10/util/Optional.h>

// TODO: No need to have this whole header, we can just put it all in
// the cpp file

namespace at { namespace cuda { namespace detail {

// Set the callback to initialize Magma, which is set by
// torch_cuda_cu. This indirection is required so magma_init is called
// in the same library where Magma will be used.
TORCH_CUDA_CPP_API void set_magma_init_fn(void (*magma_init_fn)());

TORCH_CUDA_CPP_API bool hasPrimaryContext(int64_t device_index);
TORCH_CUDA_CPP_API c10::optional<int64_t> getDeviceIndexWithPrimaryContext();

// The real implementation of CUDAHooksInterface
struct CUDAHooks : public at::CUDAHooksInterface {
  CUDAHooks(at::CUDAHooksArgs) {}
  void initCUDA() const override;
  Device getDeviceFromPtr(void* data) const override;
  bool isPinnedPtr(void* data) const override;
  const Generator& getDefaultCUDAGenerator(DeviceIndex device_index = -1) const override;
  bool hasCUDA() const override;
  bool hasMAGMA() const override;
  bool hasCuDNN() const override;
  bool hasCuSOLVER() const override;
  const at::cuda::NVRTC& nvrtc() const override;
  int64_t current_device() const override;
  bool hasPrimaryContext(int64_t device_index) const override;
  Allocator* getCUDADeviceAllocator() const override;
  Allocator* getPinnedMemoryAllocator() const override;
  bool compiledWithCuDNN() const override;
  bool compiledWithMIOpen() const override;
  bool supportsDilatedConvolutionWithCuDNN() const override;
  bool supportsDepthwiseConvolutionWithCuDNN() const override;
  bool hasCUDART() const override;
  long versionCUDART() const override;
  long versionCuDNN() const override;
  std::string showConfig() const override;
  double batchnormMinEpsilonCuDNN() const override;
  int64_t cuFFTGetPlanCacheMaxSize(int64_t device_index) const override;
  void cuFFTSetPlanCacheMaxSize(int64_t device_index, int64_t max_size) const override;
  int64_t cuFFTGetPlanCacheSize(int64_t device_index) const override;
  void cuFFTClearPlanCache(int64_t device_index) const override;
  int getNumGPUs() const override;
  void deviceSynchronize(int64_t device_index) const override;
};

}}} // at::cuda::detail
