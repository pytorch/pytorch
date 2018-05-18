#include <ATen/detail/CUDAHooksInterface.h>

#include <ATen/Generator.h>

namespace at { namespace cuda { namespace detail {

// The real implementation of CUDAHooksInterface
struct CUDAHooks : public at::CUDAHooksInterface {
  CUDAHooks(at::CUDAHooksArgs) {}
  std::unique_ptr<THCState, void(*)(THCState*)> initCUDA() const override;
  std::unique_ptr<Generator> initCUDAGenerator(Context*) const override;
  bool hasCUDA() const override;
  bool hasCuDNN() const override;
  cudaStream_t getCurrentCUDAStream(THCState*) const override;
  struct cudaDeviceProp* getCurrentDeviceProperties(THCState*) const override;
  struct cudaDeviceProp* getDeviceProperties(THCState*, int device) const override;
  int64_t current_device() const override;
  std::unique_ptr<Allocator> newPinnedMemoryAllocator() const override;
  void registerCUDATypes(Context*) const override;
  bool compiledWithCuDNN() const override;
  bool supportsDilatedConvolutionWithCuDNN() const override;
  long versionCuDNN() const override;
  double batchnormMinEpsilonCuDNN() const override;
  int getNumGPUs() const override;
};

// Sigh, the registry doesn't support namespaces :(
using at::RegistererCUDAHooksRegistry;
using at::CUDAHooksRegistry;

REGISTER_CUDA_HOOKS(CUDAHooks);

}}} // at::cuda::detail
