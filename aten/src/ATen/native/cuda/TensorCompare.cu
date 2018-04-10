#include "ATen/NativeFunctions.h"
#include "ATen/Dispatch.h"

#include "ATen/cuda/CUDAApplyUtils.cuh"
#include "ATen/cuda/CUDATensorMethods.cuh"
#include "ATen/cuda/CUDATypeConversion.cuh"

namespace {
template <typename scalar_t>
void where_cuda(
    at::Tensor& ret,
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {
  // Yes this name is repetitive, but the CPU version is called
  // CPU_tensor_apply4 and we don't have a CPU namespace or directory.
  at::cuda::CUDA_tensor_apply4<scalar_t, uint8_t, scalar_t, scalar_t>(
      ret,
      condition,
      self,
      other,
      [] __device__(
          scalar_t & ret_val,
          const uint8_t& cond_val,
          const scalar_t& self_val,
          const scalar_t& other_val) {
        ret_val = cond_val ? self_val : other_val;
      });
}
} // namespace

namespace at { namespace native {
Tensor _s_where_cuda(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other) {
  Tensor ret = self.type().tensor(self.sizes());
  AT_DISPATCH_ALL_TYPES_AND_HALF(ret.type(), "where", [&] {
    where_cuda<cuda::type<scalar_t>>(ret, condition, self, other);
  });
  return ret;
}
}} // namespace at::native
