#include "kernel.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

using torch::stable::Tensor;

Tensor mv_tensor_accessor_cuda(Tensor t1, Tensor t2) {
  Tensor res = new_empty(t1, {t1.size(0)});
  THO_DISPATCH_V2(t1.scalar_type(), "my_tensor_accessor_cuda",
                  AT_WRAP(([&]() {
                    auto resa = Accessor_cuda<scalar_t, 1>(reinterpret_cast<scalar_t*>(res.data_ptr()), res.sizes().data(), res.strides().data());
                    auto t1a = Accessor_cuda<scalar_t, 2>(reinterpret_cast<scalar_t*>(t1.data_ptr()), t1.sizes().data(), t1.strides().data());
                    auto t2a = Accessor_cuda<scalar_t, 1>(reinterpret_cast<scalar_t*>(t2.data_ptr()), t2.sizes().data(), t2.strides().data());
                    mv_tensor_accessor_kernel<Accessor_cuda, scalar_t><<<1, 1, 0, 0>>>(resa, t1a, t2a);
                  })),
                  AT_FLOATING_TYPES);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return res;
}

void boxed_mv_tensor_accessor_cuda(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  Tensor t1 = torch::stable::detail::to<Tensor>(stack[0]);
  Tensor t2 = torch::stable::detail::to<Tensor>(stack[1]);
  Tensor res = mv_tensor_accessor_cuda(t1, t2);
  stack[0] = torch::stable::detail::from(res);
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CUDA, m) {
  m.impl("mv_tensor_accessor", &boxed_mv_tensor_accessor_cuda);
}
