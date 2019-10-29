#include <cmath>
#include <iostream>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/cpu/Loops.h>

namespace at { namespace native {
namespace {

using namespace vec256;

void add_kernel(TensorIterator& iter, Scalar alpha_scalar) {
  if (iter.dtype() == ScalarType::Bool) {
      using scalar_t = bool;
      auto alpha = alpha_scalar.to<scalar_t>();
      cpu_kernel(iter,
        [=](scalar_t a, scalar_t b) -> scalar_t { return a + alpha * b; });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, iter.dtype(), "add_cpu/sub_cpu", [&]() {
      auto alpha = alpha_scalar.to<scalar_t>();
      auto alpha_vec = Vec256<scalar_t>(alpha);
      cpu_kernel_vec(iter,
        [=](scalar_t a, scalar_t b) -> scalar_t { return a + alpha * b; },
        [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
          return vec256::fmadd(b, alpha_vec, a);
        });
      });
  }
} 

void atan2_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "atan2_cpu", [&]() {
    cpu_kernel_vec(iter, [=](scalar_t a, scalar_t b) -> scalar_t {
    return std::atan2(a, b);
  },
    [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
      return a.atan2(b);
    });
  });
}

void sub_kernel(TensorIterator& iter, Scalar alpha_scalar) {
  add_kernel(iter, -alpha_scalar);
}

void mul_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    cpu_kernel(iter, [=](bool a, bool b) -> bool { return a && b; });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, iter.dtype(), "mul_cpu", [&]() {
      cpu_kernel_vec(iter,
        [=](scalar_t a, scalar_t b) -> scalar_t { return a * b; },
        [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
          return a * b;
        });
    });
  }
}

void div_kernel(TensorIterator& iter) {
  if (isIntegralType(iter.dtype(), /*includeBool*/ false)) {
    // There's no SIMD integer division, so don't try to vectorize it.
    // TODO: if the divisor is a scalar, rewrite as multiplication by a constant.
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "div_cpu", [&]() {
      cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
        return a / b;
      });
    });
  } else if (isComplexType(iter.dtype())) {
      AT_DISPATCH_COMPLEX_TYPES(iter.dtype(), "div_cpu", [&]() {
        cpu_kernel_vec(iter,
          [=](scalar_t a, scalar_t b) __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
             return a / b;
          },
          [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
            return a / b;
          });
      });
    } else {
    AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, iter.dtype(), "div_cpu", [&]() {
      cpu_kernel_vec(iter,
        [=](scalar_t a, scalar_t b) __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
           return a / b;
        },
        [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
          return a / b;
        });
    });
  }
}

void logical_xor_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, iter.input_dtype(), "logical_xor_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> bool {
          return bool(a) != bool(b);
        });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "logical_xor_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          return static_cast<scalar_t>(bool(a) != bool(b));
        });
    });
  }
}

void lt_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND2(kBool, kBFloat16, iter.input_dtype(), "lt_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> bool {
         return a < b;
       });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND(kBFloat16, iter.dtype(), "lt_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> scalar_t {
         return a < b;
       });
    });
  }
}

void le_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND2(kBool, kBFloat16, iter.input_dtype(), "le_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> bool {
         return a <= b;
       });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND(kBFloat16, iter.dtype(), "le_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> scalar_t {
         return a <= b;
       });
    });
  }
}

void gt_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND2(kBool, kBFloat16, iter.input_dtype(), "gt_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> bool {
         return a > b;
       });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND(kBFloat16, iter.dtype(), "gt_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> scalar_t {
         return a > b;
       });
    });
  }
}

void ge_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND2(kBool, kBFloat16, iter.input_dtype(), "ge_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> bool {
         return a >= b;
       });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND(kBFloat16, iter.dtype(), "ge_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> scalar_t {
         return a >= b;
       });
    });
  }
}

void eq_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBool, kBFloat16, iter.input_dtype(), "eq_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> bool {
         return a == b;
       });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, iter.dtype(), "eq_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> scalar_t {
         return a == b;
       });
    });
  }
}

void ne_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBool, kBFloat16, iter.input_dtype(), "ne_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> bool {
         return a != b;
       });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, iter.dtype(), "ne_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> scalar_t {
         return a != b;
       });
    });
  }
}

void smooth_l1_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND(kBFloat16, iter.dtype(), "smooth_l1_cpu", [&]() {
    cpu_kernel(iter, [=](scalar_t a, scalar_t b) -> scalar_t {
      auto z = std::abs(a - b);
      return z < scalar_t(1.) ? scalar_t(0.5) * z * z : z - scalar_t(0.5);
    });
  });
}

} // anonymous namespace


REGISTER_DISPATCH(add_stub, &add_kernel);
REGISTER_DISPATCH(sub_stub, &sub_kernel);
REGISTER_DISPATCH(mul_stub, &mul_kernel);
REGISTER_DISPATCH(div_stub, &div_kernel);
REGISTER_DISPATCH(atan2_stub, &atan2_kernel);
REGISTER_DISPATCH(logical_xor_stub, &logical_xor_kernel);
REGISTER_DISPATCH(lt_stub, &lt_kernel);
REGISTER_DISPATCH(le_stub, &le_kernel);
REGISTER_DISPATCH(gt_stub, &gt_kernel);
REGISTER_DISPATCH(ge_stub, &ge_kernel);
REGISTER_DISPATCH(eq_stub, &eq_kernel);
REGISTER_DISPATCH(ne_stub, &ne_kernel);
REGISTER_DISPATCH(smooth_l1_stub, &smooth_l1_kernel);

}} // namespace at::native
