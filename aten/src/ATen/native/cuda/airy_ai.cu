#include <ATen/native/UnaryOps.h>

#include <limits>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/native/cuda/jit_utils.h>
#include <ATen/NumericUtils.h>
#include <c10/core/Scalar.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/complex.h>

namespace at {
    namespace native {
        namespace {
            const char airy_ai_name[] = "airy_ai_forward";

            void airy_ai_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "airy_ai_cuda", [&]() {
                    jitted_gpu_kernel<airy_ai_name, scalar_t, scalar_t, 1>(iterator, airy_ai_string);
                });
#else
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "airy_ai_cuda", [&]() {
                    gpu_kernel(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
                        return airy_ai_forward(a);
                    });
                });
#endif // AT_USE_JITERATOR()
            }
        }

        REGISTER_DISPATCH(special_airy_ai_stub, &airy_ai_kernel_cuda);
    } // namespace native
} // namespace at

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <algorithm>



#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"

#include "c10/cuda/CUDAMathCompat.h"
namespace at {
namespace fb {

namespace {

template <typename T>
__global__ void PiecewiseLinearKernel(
    int64_t nbounds,
    int64_t nelements,
    int64_t inp_stride,
    const T* inp,
    const T* bounds,
    const T* slopes,
    const T* intercepts,
    T* out) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < nelements) {
    T inp_val = inp[i * inp_stride];
    if (inp_val <= bounds[0]) {
      out[i] = slopes[0] * bounds[0] + intercepts[0];
    } else if (inp_val >= bounds[nbounds - 1]) {
      out[i] =
          slopes[nbounds - 2] * bounds[nbounds - 1] + intercepts[nbounds - 2];
    } else {
      auto low_bound = thrust::lower_bound(
          thrust::device, bounds, bounds + nbounds, inp_val);
      int bounds_idx = low_bound - bounds - 1;
      out[i] = slopes[bounds_idx] * inp_val + intercepts[bounds_idx];
    }
  }
}

template <typename T>
void PiecewiseLinearCUDAImpl(
    const Tensor& inp,
    const Tensor& bounds,
    const Tensor& slopes,
    const Tensor& intercepts,
    Tensor& out) {
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  T* bounds_data = bounds.data_ptr<T>();
  T* slopes_data = slopes.data_ptr<T>();
  T* intercepts_data = intercepts.data_ptr<T>();
  T* inp_data = inp.data_ptr<T>();
  T* out_data = out.data_ptr<T>();
  int64_t nbounds = bounds.size(0);
  int64_t nelements = inp.size(0);
  int64_t inp_stride = inp.stride(0);
  int num_threads = std::min(nelements, (int64_t)512);
  const uint64_t num_blocks =
      cuda::ATenCeilDiv<uint64_t>(nelements, num_threads);
  PiecewiseLinearKernel<<<num_blocks, num_threads, 0, cuda_stream>>>(
      nbounds,
      nelements,
      inp_stride,
      inp_data,
      bounds_data,
      slopes_data,
      intercepts_data,
      out_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace
void PiecewiseLinearCUDA(
    const Tensor& inp,
    const Tensor& bounds,
    const Tensor& slopes,
    const Tensor& intercepts,
    Tensor& out) {
  TORCH_CHECK(
      inp.is_cuda() && bounds.is_cuda() && slopes.is_cuda() &&
          intercepts.is_cuda(),
      "all arguments should be on GPU");
  TORCH_CHECK(
      inp.ndimension() == 2 && bounds.ndimension() == 1 &&
          slopes.ndimension() == 1 && intercepts.ndimension() == 1,
      "input should be 2d, all other arguments should be 1d");
  TORCH_CHECK(
      bounds.is_contiguous() && slopes.is_contiguous() &&
          intercepts.is_contiguous() && out.is_contiguous(),
      "all arguments should be contiguous");
  TORCH_CHECK(inp.size(1) == 1, "input should have 1 value per batch")
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      inp.scalar_type(), "PiecewiseLinearCUDA", [&]() {
        PiecewiseLinearCUDAImpl<scalar_t>(inp, bounds, slopes, intercepts, out);
      });
}
} // namespace fb
} // namespace at

#include <torch/library.h>
#include "ATen/ATen.h"
#include "ATen/core/op_registration/op_registration.h"
#include "c10/core/DispatchKey.h"

namespace at {
namespace fb {
namespace {
class PiecewiseLinearCUDAKernel final : public c10::OperatorKernel {
 public:
  Tensor operator()(
      const Tensor& inp,
      const Tensor& bounds,
      const Tensor& slopes,
      const Tensor& intercepts) const {
    Tensor out = at::native::empty_like(inp);
    PiecewiseLinearCUDA(inp, bounds, slopes, intercepts, out);
    return out;
  }
};

Tensor piecewiseLinearCUDAKernel_op(
    const Tensor& inp,
    const Tensor& bounds,
    const Tensor& slopes,
    const Tensor& intercepts) {
  static const PiecewiseLinearCUDAKernel piecewiseLinearCUDAKernel_{};
  return piecewiseLinearCUDAKernel_(inp, bounds, slopes, intercepts);
}

TORCH_LIBRARY_FRAGMENT(fb, m) {
  m.def(
      "piecewise_linear("
      " Tensor inp,"
      " Tensor bounds,"
      " Tensor slopes,"
      " Tensor intercepts"
      ") -> Tensor");
  m.impl(
      "piecewise_linear",
      torch::dispatch(
          c10::DispatchKey::CUDA, TORCH_FN(piecewiseLinearCUDAKernel_op)));
}

} // namespace
} // namespace fb
} // namespace at
