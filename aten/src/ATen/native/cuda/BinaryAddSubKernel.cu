#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/BinaryOps.h>

// TODO: update to use lazynvrtc
#include <ATen/cuda/detail/LazyNVRTC.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <torch/csrc/jit/resource_guard.h>
#include <sstream>
#include <torch/csrc/jit/frontend/code_template.h>
#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <mutex>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

template<typename scalar_t>
struct AddFunctor {
  AddFunctor(scalar_t a): alpha(a) {}
  __device__ __forceinline__ scalar_t operator() (const scalar_t a, const scalar_t b) const {
    return a + alpha * b;
  }
  private:
    scalar_t alpha;
};
// stringify here?

void add_kernel_cuda(TensorIterator& iter, Scalar alpha_scalar) {
  // stringify here?
  // create template here
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kHalf, kBool, kBFloat16, iter.common_dtype(), "add_cuda/sub_cuda", [&]() {
    // NOTE: we don't need compile-time switching this does at all, so maybe use alternative?
    // Question: is instantiating worthwhile vs. just recompiling?
      // Cons of recompilation: string manipulation done every time
      // Cons of recompilation: need your own code template
      // Cons of instantiation: complicated
    // instantiate dispatched scalar types using the template here
    // this happens at runtime before the call
    // cache whether instantiated or not
    // call
    AddFunctor<scalar_t> f(alpha_scalar.to<scalar_t>());
    gpu_kernel_with_scalars(iter, f);
  });
}

static void sub_kernel_cuda(TensorIterator& iter, Scalar alpha_scalar) {
  add_kernel_cuda(iter, -alpha_scalar);
}

REGISTER_DISPATCH(add_stub, &add_kernel_cuda);
REGISTER_DISPATCH(sub_stub, &sub_kernel_cuda);

} //namespace at
} // namespace native