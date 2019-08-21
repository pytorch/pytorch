// The content of BinaryLogicalOpsKernel.h and Logical*Kernel.cpp should have inhabited in BinaryOpsKernel.cpp. But
// doing so will make the compilation of BinaryOpsKernel.cpp so long and cause the CI to break. These files merely serve
// as a workaround to reduce the compilation time of BinaryOpsKernel.cpp by breaking down BinaryOpsKernel.cpp to
// multiple files.

#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/cpu/Loops.h>

namespace at { namespace native {

template <typename Op>
static void logical_binary_kernel_impl(TensorIterator& iter, const char* op_name, Op op) {
  AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(1), op_name, [&]() {
    using self_t = scalar_t;
    AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(2), op_name, [&]() {
      using other_t = scalar_t;
      AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(0), op_name, [&]() {
        cpu_kernel(iter, [op](self_t a, other_t b) -> scalar_t {
          return static_cast<scalar_t>(op(static_cast<bool>(a), static_cast<bool>(b)));
        });
      });
    });
  });
}

}} // namespace at::native
