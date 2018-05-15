#pragma once

#include <ATen/Registry.h>
#include <ATen/Generator.h>
#include <ATen/Error.h>
#include <ATen/Allocator.h>

// Forward declare these CUDA types here to avoid including CUDA headers in
// ATen headers, which would make ATen always require CUDA to build.
struct THCState;
struct CUstream_st;
typedef struct CUstream_st *cudaStream_t;
struct cudaDeviceProp;

namespace at {
  class Context;
}

// NB: Class must live in at due to limitations of Registry.h
namespace at {

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
struct AT_API CUDAHooksInterface {

  // This should never actually be implemented, but it is used to
  // squelch -Werror=non-virtual-dtor
  virtual ~CUDAHooksInterface() {}

  // Initialize THCState and, transitively, the CUDA state
  virtual std::unique_ptr<THCState, void(*)(THCState*)> initCUDA() const {
    AT_ERROR("cannot initialize CUDA without ATen_cuda library");
  }

  virtual std::unique_ptr<Generator> initCUDAGenerator(Context*) const  {
    AT_ERROR("cannot initialize CUDA generator without ATen_cuda library");
  }

  virtual bool hasCUDA() const {
    return false;
  }

  virtual bool hasCuDNN() const {
    return false;
  }

  virtual cudaStream_t getCurrentCUDAStream(THCState*) const {
    AT_ERROR("cannot getCurrentCUDAStream() without ATen_cuda library");
  }

  virtual struct cudaDeviceProp* getCurrentDeviceProperties(THCState*) const {
    AT_ERROR("cannot getCurrentDeviceProperties() without ATen_cuda library");
  }

  virtual struct cudaDeviceProp* getDeviceProperties(THCState*, int device) const {
    AT_ERROR("cannot getDeviceProperties() without ATen_cuda library");
  }

  virtual int64_t current_device() const {
    return -1;
  }

  virtual std::unique_ptr<Allocator> newPinnedMemoryAllocator() const {
    AT_ERROR("pinned memory requires CUDA");
  }

  virtual void registerCUDATypes(Context*) const {
    AT_ERROR("cannot registerCUDATypes() without ATen_cuda library");
  }

  virtual bool compiledWithCuDNN() const {
    return false;
  }

  virtual bool supportsDilatedConvolutionWithCuDNN() const {
    return false;
  }

  virtual long versionCuDNN() const {
    AT_ERROR("cannot query cuDNN version without ATen_cuda library");
  }

  virtual double batchnormMinEpsilonCuDNN() const {
    AT_ERROR("cannot query batchnormMinEpsilonCuDNN() without ATen_cuda library");
  }

  virtual int getNumGPUs() const {
    return 0;
  }

};

// NB: dummy argument to suppress "ISO C++11 requires at least one argument
// for the "..." in a variadic macro"
struct AT_API CUDAHooksArgs {};

AT_DECLARE_REGISTRY(CUDAHooksRegistry, CUDAHooksInterface, CUDAHooksArgs)
#define REGISTER_CUDA_HOOKS(clsname) AT_REGISTER_CLASS(CUDAHooksRegistry, clsname, clsname)

namespace detail {
  AT_API const CUDAHooksInterface& getCUDAHooks();
}

} // namespace at
