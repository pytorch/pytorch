#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/zmath.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

void lt_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, iter.common_dtype(), "lt_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
      return a < b;
    });
  });
}

void le_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, iter.common_dtype(), "le_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
      return a <= b;
    });
  });
}

void gt_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, iter.common_dtype(), "gt_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
      return a > b;
    });
  });
}

void ge_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, iter.common_dtype(), "ge_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
      return a >= b;
    });
  });
}

void eq_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBool, iter.common_dtype(), "eq_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(thrust_t a, thrust_t b) -> bool {
      return a == b;
    });
  });
}

void ne_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBool, iter.common_dtype(), "ne_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(thrust_t a, thrust_t b) -> bool {
      return a != b;
    });
  });
}

void max_elementwise_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    gpu_kernel(iter, []GPU_LAMBDA(bool a, bool b) -> bool {
      return a || b;
    });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "max_elementwise_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return ::max(a, b);
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "max_elementwise_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return ::max(a, b);
      });
    });
  }
}

void min_elementwise_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    gpu_kernel(iter, []GPU_LAMBDA(bool a, bool b) -> bool {
      return a && b;
    });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "min_elementwise_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return ::min(a, b);
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "min_elementwise_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return ::min(a, b);
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
REGISTER_DISPATCH(max_elementwise_stub, &max_elementwise_kernel_cuda);
REGISTER_DISPATCH(min_elementwise_stub, &min_elementwise_kernel_cuda);

}} // namespace at::native
