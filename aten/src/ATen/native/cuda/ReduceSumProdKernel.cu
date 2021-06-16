#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOps.h>

namespace at { namespace native {

template <typename scalar_t, typename acc_t = scalar_t, typename out_t = scalar_t>
struct sum_functor {
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter, func_wrapper<out_t>([] GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
          return a + b;
        }));
  }
};

template <typename scalar_t, typename acc_t = scalar_t, typename out_t = scalar_t>
struct nansum_functor {
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter, NanSumOps<acc_t, out_t>{});
  }
};

template <typename scalar_t, typename acc_t = scalar_t, typename out_t = scalar_t>
struct prod_functor {
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter, func_wrapper<out_t>([] GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
          return a * b;
        }), 1);
  }
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
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(
        ScalarType::Bool, iter.dtype(), "sum_cuda", [&]() {
          sum_functor<scalar_t>{}(iter);
        });
  };

  reduce_dispatch<sum_functor>(iter, general_dispatcher);
}

static void nansum_kernel_cuda(TensorIterator& iter) {
  auto general_dispatcher = [](TensorIterator& iter) {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "nansum_cuda", [&]() {
      nansum_functor<scalar_t>{}(iter);
    });
  };

  reduce_dispatch<nansum_functor>(iter, general_dispatcher);
}

static void prod_kernel_cuda(TensorIterator& iter) {
  auto general_dispatcher = [](TensorIterator& iter) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(ScalarType::Bool, iter.dtype(), "prod_cuda", [&]() {
      prod_functor<scalar_t>{}(iter);
    });
  };

  reduce_dispatch<prod_functor>(iter, general_dispatcher);
}

REGISTER_DISPATCH(sum_stub, &sum_kernel_cuda);
REGISTER_DISPATCH(nansum_stub, &nansum_kernel_cuda);
REGISTER_DISPATCH(prod_stub, &prod_kernel_cuda);

}} // namespace at::native
