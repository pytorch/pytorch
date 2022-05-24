#define TORCH_ASSERT_NO_OPERATORS
#include <limits>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/OpMathType.h>

namespace at { namespace native {

void acos_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      iter.common_dtype(), "acos_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::acos(a);
        });
      });
}

const char asin_name[] = "asin";
void asin_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if(at::isComplexType(common_dtype)) {
#if AT_USE_JITERATOR
  static const auto asin_string = jiterator_stringify(
    template <typename T>
    T asin(T a) {
        return std::asin(a);
    }
  );
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "asin_name", [&]() {
    jitted_gpu_kernel<
        /*name=*/ asin_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 1>(iter, asin_string);
  });
#else
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "asin_name", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        using opmath_t = at::opmath_type<scalar_t>;
        return ::asin(static_cast<opmath_t>(a));
    });
  });
#endif
  } else {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, common_dtype, "asin_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        return ::asin(a);
    });
  });
  }
}

const char atan_name[] = "atan";
void atan_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
#if AT_USE_JITERATOR
  static const auto atan_string = jiterator_stringify(
    template <typename T>
    T atan(T a) {
        return std::atan(a);
    }
  );
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "atan_name", [&]() {
    jitted_gpu_kernel<
        /*name=*/ atan_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 1>(iter, atan_string);
  });
#else
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "atan_name", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        using opmath_t = at::opmath_type<scalar_t>;
        return ::atan(static_cast<opmath_t>(a));
    });
  });
#endif
  } else {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      common_dtype, "atan_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::atan(a);
        });
      });
  }
}

const char sin_name[] = "sin";
void sin_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if(at::isComplexType(common_dtype)) {
#if AT_USE_JITERATOR
  static const auto sin_string = jiterator_stringify(
    template <typename T>
    T sin(T a) {
        return std::sin(a);
    }
  );
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "sin_name", [&]() {
    jitted_gpu_kernel<
        /*name=*/ sin_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 1>(iter, sin_string);
  });
#else
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "sin_name", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        using opmath_t = at::opmath_type<scalar_t>;
        return ::sin(static_cast<opmath_t>(a));
    });
  });
#endif
  } else {
  AT_DISPATCH_FLOATING_TYPES_AND2(
    ScalarType::Half, ScalarType::BFloat16,
    common_dtype, "sin_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::sin(a);
        });
      });
  }
}

void cos_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      iter.common_dtype(), "cos_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::cos(a);
        });
      });
}

void sinh_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      iter.common_dtype(), "sinh_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::sinh(a);
        });
      });
}

void cosh_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      iter.common_dtype(), "cosh_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::cosh(a);
        });
      });
}

void tanh_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      iter.common_dtype(), "tanh_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::tanh(a);
        });
      });
}

void acosh_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      iter.common_dtype(), "acosh_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::acosh(a);
        });
      });
}

void asinh_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      iter.common_dtype(), "asinh_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::asinh(a);
        });
      });
}

void atanh_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      iter.common_dtype(), "atanh_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::atanh(a);
        });
      });
}

const char tan_name[] = "tan";
void tan_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
#if AT_USE_JITERATOR
  static const auto tan_string = jiterator_stringify(
    template <typename T>
    T tan(T a) {
        return std::tan(a);
    }
  );
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "tan_name", [&]() {
    jitted_gpu_kernel<
        /*name=*/ tan_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 1>(iter, tan_string);
  });
#else
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "tan_name", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        using opmath_t = at::opmath_type<scalar_t>;
        return ::tan(static_cast<opmath_t>(a));
    });
  });
#endif
  } else {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      common_dtype, "tan_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::tan(a);
        });
      });
  }
}

REGISTER_DISPATCH(acos_stub, &acos_kernel_cuda);
REGISTER_DISPATCH(acosh_stub, &acosh_kernel_cuda);
REGISTER_DISPATCH(asinh_stub, &asinh_kernel_cuda);
REGISTER_DISPATCH(atanh_stub, &atanh_kernel_cuda);
REGISTER_DISPATCH(asin_stub, &asin_kernel_cuda);
REGISTER_DISPATCH(atan_stub, &atan_kernel_cuda);
REGISTER_DISPATCH(sin_stub, &sin_kernel_cuda);
REGISTER_DISPATCH(cos_stub, &cos_kernel_cuda);
REGISTER_DISPATCH(sinh_stub, &sinh_kernel_cuda);
REGISTER_DISPATCH(cosh_stub, &cosh_kernel_cuda);
REGISTER_DISPATCH(tanh_stub, &tanh_kernel_cuda);
REGISTER_DISPATCH(tan_stub, &tan_kernel_cuda);

}} // namespace at::native
