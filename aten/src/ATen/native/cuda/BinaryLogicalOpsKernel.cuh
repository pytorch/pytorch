// The content of BinaryLogicalOpsKernel.cuh and Logical*Kernel.cu should have inhabited in BinaryOpsKernel.cu, like its
// CPU counterpart. But doing so will make the compilation of BinaryOpsKernel.cu so long and cause the CI to break.
// These files merely serve as a workaround to reduce the compilation time of BinaryOpsKernel.cu by breaking down
// BinaryOpsKernel.cu.

#pragma once

#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

namespace at { namespace native {

template <typename Op>
void logical_binary_kernel_cuda_impl(TensorIterator& iter, const char* op_name, Op op) {
  AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(1), op_name, [&]() {
    using self_t = scalar_t;
    AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(2), op_name, [&]() {
      using other_t = scalar_t;
      AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(0), op_name, [&]() {
        gpu_kernel(iter, [op]GPU_LAMBDA(self_t a, other_t b) -> scalar_t {
          return static_cast<scalar_t>(op(static_cast<bool>(a), static_cast<bool>(b)));
        });
      });
    });
  });
}

}} // namespace at::native
