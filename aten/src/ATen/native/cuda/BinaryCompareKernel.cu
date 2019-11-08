#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

void lt_kernel_cuda(TensorIterator& iter) {
  if (iter.common_dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, iter.input_dtype(), "lt_cpu", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
        return a < b;
      });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND(kHalf, iter.common_dtype(), "lt_cpu", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return a < b;
      });
    });
  }
}

void le_kernel_cuda(TensorIterator& iter) {
  if (iter.common_dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, iter.input_dtype(), "le_cpu", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
        return a <= b;
      });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND(kHalf, iter.common_dtype(), "le_cpu", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return a <= b;
      });
    });
  }
}

void gt_kernel_cuda(TensorIterator& iter) {
  if (iter.common_dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, iter.input_dtype(), "gt_cpu", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
        return a > b;
      });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND(kHalf, iter.common_dtype(), "gt_cpu", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return a > b;
      });
    });
  }
}

void ge_kernel_cuda(TensorIterator& iter) {
  if (iter.common_dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, iter.input_dtype(), "ge_cpu", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
        return a >= b;
      });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND(kHalf, iter.common_dtype(), "ge_cpu", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return a >= b;
      });
    });
  }
}

void eq_kernel_cuda(TensorIterator& iter) {
  if (iter.common_dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, iter.input_dtype(), "eq_cpu", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
        return a == b;
      });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND(kHalf, iter.common_dtype(), "eq_cpu", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return a == b;
      });
    });
  }
}

void ne_kernel_cuda(TensorIterator& iter) {
  if (iter.common_dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, iter.input_dtype(), "ne_cpu", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
        return a != b;
      });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND(kHalf, iter.common_dtype(), "ne_cpu", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return a != b;
      });
    });
  }
}

REGISTER_DISPATCH(lt_stub, &lt_kernel_cuda);
REGISTER_DISPATCH(le_stub, &le_kernel_cuda);
REGISTER_DISPATCH(gt_stub, &gt_kernel_cuda);
REGISTER_DISPATCH(ge_stub, &ge_kernel_cuda);
REGISTER_DISPATCH(eq_stub, &eq_kernel_cuda);
REGISTER_DISPATCH(ne_stub, &ne_kernel_cuda);

}} // namespace at::native
