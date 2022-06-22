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

const char acos_name[] = "acos";
void acos_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if(at::isComplexType(common_dtype)) {
#if AT_USE_JITERATOR
  static const auto acos_string = jiterator_stringify(
    template <typename T>
    T acos(T a) {
        return std::acos(a);
    }
  );
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "acos_name", [&]() {
    jitted_gpu_kernel<
        /*name=*/ acos_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 1>(iter, acos_string);
  });
#else
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "acos_name", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        using opmath_t = at::opmath_type<scalar_t>;
        return ::acos(static_cast<opmath_t>(a));
    });
  });
#endif
  } else {
  AT_DISPATCH_FLOATING_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      common_dtype, "acos_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::acos(a);
        });
      });
  }
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

const char cos_name[] = "cos";
void cos_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if(at::isComplexType(common_dtype)) {
#if AT_USE_JITERATOR
  static const auto cos_string = jiterator_stringify(
    template <typename T>
    T cos(T a) {
        return std::cos(a);
    }
  );
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "cos_name", [&]() {
    jitted_gpu_kernel<
        /*name=*/ cos_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 1>(iter, cos_string);
  });
#else
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "cos_name", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        using opmath_t = at::opmath_type<scalar_t>;
        return ::cos(static_cast<opmath_t>(a));
    });
  });
#endif
  } else {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      common_dtype, "cos_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::cos(a);
        });
      });
  }
}

const char sinh_name[] = "sinh";
void sinh_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if(at::isComplexType(common_dtype)) {
#if AT_USE_JITERATOR
  static const auto sinh_string = jiterator_stringify(
    template <typename T>
    T sinh(T a) {
        return std::sinh(a);
    }
  );
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "sinh_name", [&]() {
    jitted_gpu_kernel<
        /*name=*/ sinh_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 1>(iter, sinh_string);
  });
#else
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "sinh_name", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        using opmath_t = at::opmath_type<scalar_t>;
        return ::sinh(static_cast<opmath_t>(a));
    });
  });
#endif
  } else {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      common_dtype, "sinh_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::sinh(a);
        });
      });
  }
}

const char cosh_name[] = "cosh";
void cosh_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if(at::isComplexType(common_dtype)) {
#if AT_USE_JITERATOR
  static const auto cosh_string = jiterator_stringify(
    template <typename T>
    T cosh(T a) {
        return std::cosh(a);
    }
  );
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "cosh_name", [&]() {
    jitted_gpu_kernel<
        /*name=*/ cosh_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 1>(iter, cosh_string);
  });
#else
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "cosh_name", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        using opmath_t = at::opmath_type<scalar_t>;
        return ::cosh(static_cast<opmath_t>(a));
    });
  });
#endif
  } else {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      common_dtype, "cosh_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::cosh(a);
        });
      });
  }
}

const char tanh_name[] = "tanh";
void tanh_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if(at::isComplexType(common_dtype)) {
#if AT_USE_JITERATOR
  static const auto tanh_string = jiterator_stringify(
    template <typename T>
    T tanh(T a) {
        return std::tanh(a);
    }
  );
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "tanh_name", [&]() {
    jitted_gpu_kernel<
        /*name=*/ tanh_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 1>(iter, tanh_string);
  });
#else
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "tanh_name", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        using opmath_t = at::opmath_type<scalar_t>;
        return ::tanh(static_cast<opmath_t>(a));
    });
  });
#endif
  } else {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      common_dtype, "tanh_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::tanh(a);
        });
      });
  }
}

const char acosh_name[] = "acosh";
void acosh_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if(at::isComplexType(common_dtype)) {
#if AT_USE_JITERATOR
  static const auto acosh_string = jiterator_stringify(
    template <typename T>
    T acosh(T a) {
        return std::acosh(a);
    }
  );
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "acosh_name", [&]() {
    jitted_gpu_kernel<
        /*name=*/ acosh_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 1>(iter, acosh_string);
  });
#else
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "acosh_name", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        using opmath_t = at::opmath_type<scalar_t>;
        return ::acosh(static_cast<opmath_t>(a));
    });
  });
#endif
  } else {
  AT_DISPATCH_FLOATING_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      common_dtype, "acosh_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::acosh(a);
        });
      });
  }
}

const char asinh_name[] = "asinh";
void asinh_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if(at::isComplexType(common_dtype)) {
#if AT_USE_JITERATOR
  static const auto asinh_string = jiterator_stringify(
    template <typename T>
    T asinh(T a) {
        return std::asinh(a);
    }
  );
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "asinh_name", [&]() {
    jitted_gpu_kernel<
        /*name=*/ asinh_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 1>(iter, asinh_string);
  });
#else
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "asinh_name", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        using opmath_t = at::opmath_type<scalar_t>;
        return ::asinh(static_cast<opmath_t>(a));
    });
  });
#endif
  } else {
  AT_DISPATCH_FLOATING_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      common_dtype, "asinh_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::asinh(a);
        });
      });
  }
}

const char atanh_name[] = "atanh";
void atanh_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if(at::isComplexType(common_dtype)) {
#if AT_USE_JITERATOR
  static const auto atanh_string = jiterator_stringify(
    template <typename T>
    T atanh(T a) {
        return std::atanh(a);
    }
  );
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "atanh_name", [&]() {
    jitted_gpu_kernel<
        /*name=*/ atanh_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 1>(iter, atanh_string);
  });
#else
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "atanh_name", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        using opmath_t = at::opmath_type<scalar_t>;
        return ::atanh(static_cast<opmath_t>(a));
    });
  });
#endif
  } else {
  AT_DISPATCH_FLOATING_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      common_dtype, "atanh_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::atanh(a);
        });
      });
  }
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
