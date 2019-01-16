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

}  // anonymous namespace

REGISTER_DISPATCH(sum_stub, &sum_kernel_impl);
REGISTER_DISPATCH(std_var_stub, &std_var_kernel_impl);
REGISTER_DISPATCH(prod_stub, &prod_kernel_impl);
REGISTER_DISPATCH(mean_stub, &mean_kernel_impl);

}}  // namespace at::native
