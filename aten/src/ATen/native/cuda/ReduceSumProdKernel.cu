#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/jit_macros.h>
#include <ATen/OpMathType.h>

namespace at::native {

template <typename scalar_t, typename acc_t = scalar_t, typename out_t = scalar_t>
struct sum_functor {
  void operator()(TensorIterator& iter) {
#ifdef USE_ROCM
    // Half and BFloat16 can be packed in groups of up to 8 elements and
    // can use *_DWORDX4 instructions to achieve that.
    const bool is_16_bits =
      ( (std::is_same<at::Half, scalar_t>::value) ||
        (std::is_same<at::BFloat16, scalar_t>::value) );
    if (is_16_bits) {
      gpu_reduce_kernel<scalar_t, out_t, /*vt0=*/4, /*input_vec_size=*/8>(
        iter, func_wrapper<out_t>([] GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
          return a + b;
        }));
      return;
    }
#endif
    gpu_reduce_kernel<scalar_t, out_t>(
        iter, func_wrapper<out_t>([] GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
          return a + b;
        }));
  }
};

// jiterated specialization for `complex<Half>`
constexpr char sum_name[] = "sum";
template <>
struct sum_functor<c10::complex<at::Half>> {
// jiterator reduction fails on windows
// Ref: https://github.com/pytorch/pytorch/issues/77305
#if AT_USE_JITERATOR() && !defined(_MSC_VER)
  void operator()(TensorIterator& iter) {
    using scalar_t = c10::complex<at::Half>;
    std::string func = jiterator_stringify(
    arg_t combine(arg_t a, arg_t b) {
      return a + b;
    }
    );
    jitted_gpu_reduce_kernel<sum_name, scalar_t, scalar_t>(
        iter, func, 0.);
  }
#else
  void operator()(TensorIterator& iter) {
    using scalar_t = c10::complex<at::Half>;
    using acc_t = at::opmath_type<scalar_t>;
    gpu_reduce_kernel<scalar_t, scalar_t>(
        iter, func_wrapper<scalar_t>([] GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
          return a + b;
        }), acc_t{0.});
  }
#endif
};

template <typename scalar_t, typename acc_t = scalar_t, typename out_t = scalar_t>
struct nansum_functor {
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter, NanSumOps<acc_t, out_t>{});
  }
};

constexpr char nansum_name[] = "nansum";
template <typename scalar_t>
struct nansum_functor_complex {
#if AT_USE_JITERATOR()
  void operator()(TensorIterator& iter) {
    std::string func = jiterator_stringify(
        arg_t combine(arg_t a, scalar_t b) {
          return a + (std::isnan(b) ? arg_t{0.} : arg_t{b});
        }
    );
    jitted_gpu_reduce_kernel<nansum_name, scalar_t, scalar_t>(
        iter, func, 0.);
  }
#else
  void operator()(TensorIterator& iter) {
    using acc_t = at::opmath_type<scalar_t>;
    gpu_reduce_kernel<scalar_t, scalar_t>(
        iter, NanSumOps<acc_t, acc_t>{});
  }
#endif
};

constexpr char prod_name[] = "prod";
template <typename scalar_t, typename acc_t = scalar_t, typename out_t = scalar_t>
struct prod_functor {
  // jiterator reduction fails on windows
  // Ref: https://github.com/pytorch/pytorch/issues/77305
  #if AT_USE_JITERATOR() && !defined(_MSC_VER)
  void operator()(TensorIterator& iter) {
    std::string func = jiterator_stringify(
    arg_t combine(arg_t a, arg_t b) {
      return a * b;
    }
    );
    jitted_gpu_reduce_kernel<prod_name, scalar_t, out_t>(
        iter, func, 1.);
  }
  #else
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter, func_wrapper<out_t>([] GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
          return a * b;
        }), 1.);
  }
  #endif
};

// Workaround for the error: '*' in boolean context, suggest '&&' instead [-Werror=int-in-bool-context]
template <>
struct prod_functor<bool> {
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<bool, bool>(
        iter, func_wrapper<bool>([] GPU_LAMBDA(bool a, bool b) -> bool {
          return a && b;
        }), 1);
  }
};

// jiterated specialization for `complex<Half>`
template <>
struct prod_functor<c10::complex<at::Half>> {
// jiterator reduction fails on windows
// Ref: https://github.com/pytorch/pytorch/issues/77305
#if AT_USE_JITERATOR() && !defined(_MSC_VER)
  void operator()(TensorIterator& iter) {
    using scalar_t = c10::complex<at::Half>;
    std::string func =
        jiterator_stringify(arg_t combine(arg_t a, arg_t b) { return a * b; });
    jitted_gpu_reduce_kernel<prod_name, scalar_t, scalar_t>(iter, func, 1.);
  }
#else
  void operator()(TensorIterator& iter) {
    using scalar_t = c10::complex<at::Half>;
    using acc_t = at::opmath_type<scalar_t>;
    gpu_reduce_kernel<scalar_t, scalar_t>(
        iter,
        func_wrapper<scalar_t>(
            [] GPU_LAMBDA(acc_t a, acc_t b) -> acc_t { return a * b; }),
        acc_t{1.});
  }
#endif
};

template <typename scalar_t, typename enable = void>
struct xor_sum_functor {
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<scalar_t, uint64_t>(
        iter,
        func_wrapper<uint64_t>(
            [] GPU_LAMBDA(uint64_t a, uint64_t b) -> uint64_t {
              return a ^ b;
            }));
  }
};

template <typename scalar_t>
struct xor_sum_functor<scalar_t, std::enable_if_t<!std::is_integral_v<scalar_t>>> {
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<scalar_t, double>(
        iter,
        // implicitly upcast scalar_t to double
        func_wrapper<double>([] GPU_LAMBDA(double a, double b) -> double {
          union {
            double d;
            uint64_t u;
          } a_converter, b_converter, result_converter;

          a_converter.d = a;
          b_converter.d = b;
          result_converter.u = a_converter.u ^ b_converter.u;
          // return a double, otherwise uint64_t will be cast to double
          // when accumulating and the result will be wrong
          return result_converter.d;
        }));
  }
};

template <typename scalar_t>
struct xor_sum_functor<scalar_t, std::enable_if_t<std::is_same_v<scalar_t, bool>>>  {
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<bool, uint64_t>(
        iter, func_wrapper<uint64_t>([] GPU_LAMBDA(bool a, bool b) -> uint64_t {
          // Bitcast to uint64_t after the XOR operation (using != for booleans)
          return static_cast<uint64_t>(a != b);
        }));
  }
};

// The function `reduce_dispatch` below dispatches to the kernel based
// on the type of `iter`. It takes care of the common logic
// for handling Half-Precision floating types.
// Otherwise the functor `op` is called to dispatch to the kernel
// of relevant type.
//
// Note: Functor `op` should take care of all the types to be supported
//       except for `at::Half` and `at::BFloat16`.
template <
    template <
        typename scalar_t,
        typename acc_t = scalar_t,
        typename out_t = scalar_t>
    typename OpFunctor,
    typename GeneralDispatcher>
static void reduce_dispatch(TensorIterator& iter, GeneralDispatcher op) {
  if (iter.dtype() == kHalf) {
    return OpFunctor<at::Half, float>{}(iter);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return OpFunctor<at::Half, float, float>{}(iter);
  } else if (iter.dtype() == kBFloat16) {
    return OpFunctor<at::BFloat16, float>{}(iter);
  } else if (iter.dtype(1) == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return OpFunctor<at::BFloat16, float, float>{}(iter);
  }
  op(iter);
}

static void sum_kernel_cuda(TensorIterator& iter){
  auto general_dispatcher = [](TensorIterator& iter) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        kBool, kComplexHalf, iter.dtype(), "sum_cuda", [&]() {
          sum_functor<scalar_t>{}(iter);
        });
  };

  reduce_dispatch<sum_functor>(iter, general_dispatcher);
}

static void nansum_kernel_cuda(TensorIterator& iter) {
  auto general_dispatcher = [](TensorIterator& iter) {
    auto dtype = iter.dtype();
    if (at::isComplexType(dtype)) {
        AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "nansum_cuda", [&]() {
          nansum_functor_complex<scalar_t>{}(iter);
        });
    } else {
        AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "nansum_cuda", [&]() {
          nansum_functor<scalar_t>{}(iter);
        });
    }
  };

  reduce_dispatch<nansum_functor>(iter, general_dispatcher);
}

static void prod_kernel_cuda(TensorIterator& iter) {
  auto general_dispatcher = [](TensorIterator& iter) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kComplexHalf, kBool, iter.dtype(), "prod_cuda", [&]() {
      prod_functor<scalar_t>{}(iter);
    });
  };

  reduce_dispatch<prod_functor>(iter, general_dispatcher);
}

static void xor_sum_kernel_cuda(TensorIterator& iter) {
  // Use iter.dtype(1) to dispatch based on the type of the input tensor
  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBFloat16, kBool, iter.dtype(1), "xor_sum_cuda", [&]() {
        xor_sum_functor<scalar_t>{}(iter);
      });
}

REGISTER_DISPATCH(sum_stub, &sum_kernel_cuda)
REGISTER_DISPATCH(nansum_stub, &nansum_kernel_cuda)
REGISTER_DISPATCH(prod_stub, &prod_kernel_cuda)
REGISTER_DISPATCH(xor_sum_stub, &xor_sum_kernel_cuda)

} // namespace at::native
