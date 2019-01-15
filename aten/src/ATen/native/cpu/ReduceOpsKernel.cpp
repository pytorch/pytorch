#include <numeric>
#include <iterator>
#include <algorithm>

#include <ATen/Dispatch.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/cpu/Reduce.h>
#include <c10/util/Optional.h>

namespace at { namespace native { namespace {

using namespace vec256;

static void sum_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES(iter.type(), "sum", [&] {
    binary_kernel_reduce_vec(
      iter,
      [=](scalar_t a, scalar_t b) -> scalar_t { return a + b; },
      [=](Vec256<scalar_t> a, Vec256<scalar_t> b) { return a + b; });
  });
}

static void mean_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES(iter.type(), "mean", [&] {
    scalar_t factor = scalar_t(iter.num_output_elements()) / iter.numel();
    binary_kernel_reduce(
      iter,
      MeanOps<scalar_t, scalar_t> {factor},
      scalar_t(0)
    );
  });
}

static void std_var_kernel_impl(TensorIterator &iter, bool unbiased, bool take_sqrt) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.type(), "std", [&] {
    binary_kernel_reduce(
      iter,
      WelfordOps<scalar_t, double> { unbiased, take_sqrt },
      WelfordData<double>()
    );
  });
}

static void prod_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES(iter.type(), "prod", [&] {
    binary_kernel_reduce_vec(
      iter,
      [=](scalar_t a, scalar_t b) -> scalar_t { return a * b; },
      [=](Vec256<scalar_t> a, Vec256<scalar_t> b) { return a * b; },
      /*identity=*/1);
  });
}

static void norm_kernel_tensor_iterator_impl(
    TensorIterator& iter,
    Scalar p) {
  float val;
  if (p.isIntegral()) {
    val = p.to<int64_t>();
  } else if (p.isFloatingPoint()) {
    val = p.to<float>();
  } else {
    AT_ERROR("norm_kernel_tensor_iterator_impl expects norm to be integer or float");
  }


  if (val == 0) {
    AT_DISPATCH_FLOATING_TYPES(iter.type(), "norm", [&] {
      binary_kernel_reduce(
        iter,
        NormZeroOps<scalar_t>(),
        scalar_t(0)
      );
    });
  } else if (val == 1) {
    AT_DISPATCH_FLOATING_TYPES(iter.type(), "norm", [&] {
      binary_kernel_reduce(
        iter,
        NormOneOps<scalar_t>(),
        scalar_t(0)
      );
    });
  } else if (val == INFINITY) {
    AT_DISPATCH_FLOATING_TYPES(iter.type(), "norm", [&] {
      binary_kernel_reduce(
        iter,
        AbsMaxOps<scalar_t>(),
        std::numeric_limits<scalar_t>::min()
      );
    });
  } else if (val == -INFINITY) {
    AT_DISPATCH_FLOATING_TYPES(iter.type(), "norm", [&] {
      binary_kernel_reduce(
        iter,
        AbsMinOps<scalar_t>(),
        std::numeric_limits<scalar_t>::max()
      );
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(iter.type(), "norm", [&] {
      binary_kernel_reduce(
        iter,
        NormOps<scalar_t> { scalar_t(val) },
        scalar_t(0)
      );
    });
  }
}

}  // anonymous namespace

REGISTER_DISPATCH(sum_stub, &sum_kernel_impl);
REGISTER_DISPATCH(std_var_stub, &std_var_kernel_impl);
REGISTER_DISPATCH(prod_stub, &prod_kernel_impl);
REGISTER_DISPATCH(mean_stub, &mean_kernel_impl);
REGISTER_DISPATCH(norm_stub, &norm_kernel_tensor_iterator_impl);

}}  // namespace at::native
