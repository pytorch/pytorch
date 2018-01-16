#include "ATen/NativeFunctions.h"
#include "ATen/Dispatch.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"

namespace at {
namespace native {

template <typename scalar>
struct WhereOpCUDA {
  static void apply(Tensor& ret, const Tensor& condition, const Tensor& self, const Tensor& other) {
    // yes this name is repetitive, but the CPU version is called CPU_tensor_apply4 and we don't have
    // a CPU namespace or diectory.
    cuda::CUDA_tensor_apply4<scalar, uint8_t, scalar, scalar>(ret, condition, self, other,
      [] __device__ (scalar& ret_val, const uint8_t& cond_val, const scalar &self_val, const scalar &other_val) {
        ret_val = cond_val ? self_val : other_val;
      }
    );
  }
};

Tensor _s_where_cuda(const Tensor& condition, const Tensor& self, const Tensor& other) {
  Tensor ret = self.type().tensor(self.sizes());
  dispatch_all<void, WhereOpCUDA>(ret.type(), "where", ret, condition, self, other);
  return ret;
}

} // at::native
} // at
