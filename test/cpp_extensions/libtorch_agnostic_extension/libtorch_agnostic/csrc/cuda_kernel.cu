#include "kernel.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

using torch::stable::Tensor;

Tensor mv_tensor_accessor_cuda(Tensor m, Tensor v) {
  STD_TORCH_CHECK(m.dim() == 2, "m must be 2D");
  STD_TORCH_CHECK(v.dim() == 1, "v must be 1D");
  STD_TORCH_CHECK(m.size(1) == v.size(0), "m.shape[1] == v.shape[0] must hold");
  STD_TORCH_CHECK(m.scalar_type() == v.scalar_type(), "m and v must have the same dtype");
  STD_TORCH_CHECK(m.device() == v.device(), "m and v must be on the same device");
  Tensor res = new_empty(m, {m.size(0)});
  THO_DISPATCH_V2(m.scalar_type(), "mv_tensor_accessor_cuda",
                  AT_WRAP(([&]() {
                    auto resa = Accessor_cuda<scalar_t, 1>(reinterpret_cast<scalar_t*>(res.data_ptr()), res.sizes().data(), res.strides().data());
                    auto ma = Accessor_cuda<scalar_t, 2>(reinterpret_cast<scalar_t*>(m.data_ptr()), m.sizes().data(), m.strides().data());
                    auto va = Accessor_cuda<scalar_t, 1>(reinterpret_cast<scalar_t*>(v.data_ptr()), v.sizes().data(), v.strides().data());
                    mv_tensor_accessor_kernel<Accessor_cuda, scalar_t><<<1, 1, 0, 0>>>(resa, ma, va);
                  })),
                  AT_FLOATING_TYPES);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return res;
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CUDA, m) {
  m.impl("mv_tensor_accessor", TORCH_BOX(&mv_tensor_accessor_cuda));
}
