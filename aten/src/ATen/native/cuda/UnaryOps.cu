#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <THC/THCGeneral.h>
#include <THC/THCThrustAllocator.cuh>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>

namespace at { namespace native {

Tensor& bitwise_not_out_cuda(Tensor& out, const Tensor& self) {
  checkBackend("bitwise_not", {out}, Backend::CUDA);
  out.resize_(self.sizes());
  AT_DISPATCH_INTEGRAL_AND_BOOL_TYPES(out.scalar_type(), "bitwise_not_out_cuda", [&]() {
    auto self_data = thrust::device_ptr<scalar_t>(self.data<scalar_t>());
    auto out_data = thrust::device_ptr<scalar_t>(out.data<scalar_t>());

    auto state = globalContext().getTHCState();
    THCThrustAllocator thrust_alloc(state);
    auto policy = thrust::cuda::par(thrust_alloc).on(at::cuda::getCurrentCUDAStream());
    thrust::transform(policy, self_data, self_data + self.numel(),
                      thrust::make_constant_iterator(scalar_t(-1)), out_data,
                      thrust::bit_xor<scalar_t>());
  });

  return out;
}

Tensor& bitwise_not__cuda(Tensor& self) {
  return bitwise_not_out_cuda(self, self);
}

} // native
} // at
