#pragma once

#include <c10/core/Allocator.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/Registry.h>

#include <cstddef>
#include <functional>
#include <memory>

// Forward-declares at::Context, at::Generator and at::cuda::NVRTC
namespace at {
class Context;
struct Generator;
namespace cuda {
struct NVRTC;
} // namespace cuda
} // namespace at

// NB: Class must live in `at` due to limitations of Registry.h.
namespace at {

#ifdef _MSC_VER
constexpr const char* CUDA_HELP =
  "PyTorch splits its backend into two shared libraries: a CPU library "
  "and a CUDA library; this error has occurred because you are trying "
  "to use some CUDA functionality, but the CUDA library has not been "
  "loaded by the dynamic linker for some reason.  The CUDA library MUST "
  "be loaded, EVEN IF you don't directly use any symbols from the CUDA library! "
  "One common culprit is a lack of -INCLUDE:?warp_size@cuda@at@@YAHXZ "
  "in your link arguments; many dynamic linkers will delete dynamic library "
  "dependencies if you don't depend on any of their symbols.  You can check "
  "if this has occurred by using link on your binary to see if there is a "
  "dependency on *_cuda.dll library.";
#else
constexpr const char* CUDA_HELP =
  "PyTorch splits its backend into two shared libraries: a CPU library "
  "and a CUDA library; this error has occurred because you are trying "
  "to use some CUDA functionality, but the CUDA library has not been "
  "loaded by the dynamic linker for some reason.  The CUDA library MUST "
  "be loaded, EVEN IF you don't directly use any symbols from the CUDA library! "
  "One common culprit is a lack of -Wl,--no-as-needed in your link arguments; many "
  "dynamic linkers will delete dynamic library dependencies if you don't "
  "depend on any of their symbols.  You can check if this has occurred by "
  "using ldd on your binary to see if there is a dependency on *_cuda.so "
  "library.";
#endif

// The CUDAHooksInterface is an omnibus interface for any CUDA functionality
// which we may want to call into from CPU code (and thus must be dynamically
// dispatched, to allow for separate compilation of CUDA code).  How do I
// decide if a function should live in this class?  There are two tests:
//
//  1. Does the *implementation* of this function require linking against
//     CUDA libraries?
//
//  2. Is this function *called* from non-CUDA ATen code?
//
// (2) should filter out many ostensible use-cases, since many times a CUDA
// function provided by ATen is only really ever used by actual CUDA code.
//
// TODO: Consider putting the stub definitions in another class, so that one
// never forgets to implement each virtual function in the real implementation
// in CUDAHooks.  This probably doesn't buy us much though.
struct TORCH_API CUDAHooksInterface {
  // This should never actually be implemented, but it is used to
  // squelch -Werror=non-virtual-dtor
  virtual ~CUDAHooksInterface() = default;

  // Initialize THCState and, transitively, the CUDA state
  virtual void initCUDA() const {
    TORCH_CHECK(false, "Cannot initialize CUDA without ATen_cuda library. ", CUDA_HELP);
  }

  virtual const Generator& getDefaultCUDAGenerator(DeviceIndex device_index = -1) const {
    (void)device_index; // Suppress unused variable warning
    TORCH_CHECK(false, "Cannot get default CUDA generator without ATen_cuda library. ", CUDA_HELP);
  }

  virtual Device getDeviceFromPtr(void* /*data*/) const {
    TORCH_CHECK(false, "Cannot get device of pointer on CUDA without ATen_cuda library. ", CUDA_HELP);
  }

  virtual bool isPinnedPtr(const void* /*data*/) const {
    return false;
  }

  virtual bool hasCUDA() const {
    return false;
  }

  virtual bool hasCUDART() const {
    return false;
  }

  virtual bool hasMAGMA() const {
    return false;
  }

  virtual bool hasCuDNN() const {
    return false;
  }

  virtual bool hasCuSOLVER() const {
    return false;
  }

  virtual bool hasROCM() const {
    return false;
  }

  virtual const at::cuda::NVRTC& nvrtc() const {
    TORCH_CHECK(false, "NVRTC requires CUDA. ", CUDA_HELP);
  }

  virtual bool hasPrimaryContext(int64_t device_index) const {
    TORCH_CHECK(false, "Cannot call hasPrimaryContext(", device_index, ") without ATen_cuda library. ", CUDA_HELP);
  }

  virtual int64_t current_device() const {
    return -1;
  }

  virtual Allocator* getPinnedMemoryAllocator() const {
    TORCH_CHECK(false, "Pinned memory requires CUDA. ", CUDA_HELP);
  }

  virtual Allocator* getCUDADeviceAllocator() const {
    TORCH_CHECK(false, "CUDADeviceAllocator requires CUDA. ", CUDA_HELP);
  }

  virtual bool compiledWithCuDNN() const {
    return false;
  }

  virtual bool compiledWithMIOpen() const {
    return false;
  }

  virtual bool supportsDilatedConvolutionWithCuDNN() const {
    return false;
  }

  virtual bool supportsDepthwiseConvolutionWithCuDNN() const {
    return false;
  }

  virtual bool supportsBFloat16ConvolutionWithCuDNNv8() const {
    return false;
  }

  virtual long versionCuDNN() const {
    TORCH_CHECK(false, "Cannot query cuDNN version without ATen_cuda library. ", CUDA_HELP);
  }

  virtual long versionCUDART() const {
    TORCH_CHECK(false, "Cannot query CUDART version without ATen_cuda library. ", CUDA_HELP);
  }

  virtual std::string showConfig() const {
    TORCH_CHECK(false, "Cannot query detailed CUDA version without ATen_cuda library. ", CUDA_HELP);
  }

  virtual double batchnormMinEpsilonCuDNN() const {
    TORCH_CHECK(false,
        "Cannot query batchnormMinEpsilonCuDNN() without ATen_cuda library. ", CUDA_HELP);
  }

  virtual int64_t cuFFTGetPlanCacheMaxSize(int64_t /*device_index*/) const {
    TORCH_CHECK(false, "Cannot access cuFFT plan cache without ATen_cuda library. ", CUDA_HELP);
  }

  virtual void cuFFTSetPlanCacheMaxSize(int64_t /*device_index*/, int64_t /*max_size*/) const {
    TORCH_CHECK(false, "Cannot access cuFFT plan cache without ATen_cuda library. ", CUDA_HELP);
  }

  virtual int64_t cuFFTGetPlanCacheSize(int64_t /*device_index*/) const {
    TORCH_CHECK(false, "Cannot access cuFFT plan cache without ATen_cuda library. ", CUDA_HELP);
  }

  virtual void cuFFTClearPlanCache(int64_t /*device_index*/) const {
    TORCH_CHECK(false, "Cannot access cuFFT plan cache without ATen_cuda library. ", CUDA_HELP);
  }

  virtual int getNumGPUs() const {
    return 0;
  }

  virtual void deviceSynchronize(int64_t /*device_index*/) const {
    TORCH_CHECK(false, "Cannot synchronize CUDA device without ATen_cuda library. ", CUDA_HELP);
  }
};

// NB: dummy argument to suppress "ISO C++11 requires at least one argument
// for the "..." in a variadic macro"
struct TORCH_API CUDAHooksArgs {};

TORCH_DECLARE_REGISTRY(CUDAHooksRegistry, CUDAHooksInterface, CUDAHooksArgs);
#define REGISTER_CUDA_HOOKS(clsname) \
  C10_REGISTER_CLASS(CUDAHooksRegistry, clsname, clsname)

namespace detail {
TORCH_API const CUDAHooksInterface& getCUDAHooks();
} // namespace detail
} // namespace at
