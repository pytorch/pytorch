#include <ATen/Registry.h>

#include "ATen/Generator.h"

// Forward declare these CUDA types here to avoid including CUDA headers in
// ATen headers, which would make ATen always require CUDA to build.
struct THCState;
struct CUstream_st;
typedef struct CUstream_st *cudaStream_t;
struct cudaDeviceProp;

namespace at {
  class Context;
}

namespace at { namespace detail {

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
class CUDAHooksInterface {

  // Initialize THCState and, transitively, the CUDA state
  virtual std::unique_ptr<THCState, void(*)(THCState*)> initCUDA() const {
    throw std::runtime_error("cannot initialize CUDA without ATen_cuda library");
  }

  virtual std::unique_ptr<Generator> initCUDAGenerator(Context*) const  {
    throw std::runtime_error("cannot initialize CUDA generator without ATen_cuda library");
  }

};

AT_DECLARE_REGISTRY(CUDAHooksRegistry, CUDAHooksInterface);
AT_DEFINE_REGISTRY(CUDAHooksRegistry, CUDAHooksInterface);
#define REGISTER_CUDA_HOOKS(clsname) AT_REGISTER_CLASS(CUDAHooksRegistry, clsname, clsname)

}} // namespace at::detail
