#include<ATen/native/ReduceAllOps.h>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/cpu/Loops.h>
#include <ATen/native/cpu/zmath.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/cpu/vec256/vec256.h>

namespace at { namespace native {
namespace {

using namespace vec256;

template <typename scalar_t, typename func_t, typename vec_func_t>
inline void reduce_all_impl_vec(
    Tensor& output,
    const Tensor& input,
    const scalar_t ident_v,
    func_t op,
    vec_func_t vop) {
  using Vec = Vec256<scalar_t>;
  const int64_t input_numel = input.numel();
  auto input_data = input.data_ptr<scalar_t>();
  // NOTE: parallel_reduce not support bool type
  scalar_t result = at::parallel_reduce(0, input_numel, internal::GRAIN_SIZE, ident_v, 
    [&](int64_t start, int64_t end, const scalar_t ident) -> scalar_t {
      scalar_t partial_out = vec256::reduce_all<scalar_t>(
        [=](Vec x, Vec y) { return vop(x, y); },
        input_data + start,
        end - start);
      return partial_out;
    }, op);
  output.fill_(result);
}

// For operation not support in avx/avx2
template <typename scalar_t, typename func_t>
inline void reduce_all_impl(
    Tensor& output,
    const Tensor& input,
    const scalar_t ident_v,
    func_t op) {
  const int64_t input_numel = input.numel();
  auto input_data = input.data_ptr<scalar_t>();
  scalar_t result = at::parallel_reduce(0, input_numel, internal::GRAIN_SIZE, ident_v, 
    [&](int64_t start, int64_t end, const scalar_t ident) -> scalar_t {
      scalar_t partial_out = ident;
      for (int64_t i = start; i < end; i++) {
         partial_out = op(partial_out, input_data[i]);
      }
      return partial_out;
    }, op);
  output.fill_(result);
}

static void min_all_kernel_impl(Tensor& result, const Tensor& input) {
  if (input.scalar_type() == ScalarType::Bool) {
    TensorIterator iter = TensorIterator();
    iter.add_input(input);
    iter.build();
    bool result_data  = true;
    cpu_serial_kernel(iter, [&](const bool a) -> void {
      result_data = result_data && a;
    });
    result.fill_(result_data);
  } else if(input.scalar_type() == ScalarType::Long) {
    // for int64_t, vectorized implementation have performance issue,
    // just use scalar path
    reduce_all_impl<int64_t>(result, input, upper_bound<int64_t>(),
      [=](int64_t a, int64_t b) -> int64_t { return min_impl(a, b); });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(input.scalar_type(), "min_all", [&] {
      using Vec = vec256::Vec256<scalar_t>;
      reduce_all_impl_vec<scalar_t>(result, input, upper_bound<scalar_t>(),
        [=] (scalar_t a , scalar_t b) -> scalar_t { return min_impl(a, b); },
        [=](Vec a, Vec b) -> Vec { return minimum(a, b); });
    });
  }
}

static void max_all_kernel_impl(Tensor& result, const Tensor& input) {
  if (input.scalar_type() == ScalarType::Bool) {
    TensorIterator iter = TensorIterator();
    iter.add_input(input);
    iter.build();
    bool result_data  = false;
    cpu_serial_kernel(iter, [&](const bool a) -> void {
      result_data = result_data || a;
    });
    result.fill_(result_data);
  } else if (input.scalar_type() == ScalarType::Long) {
    // for int64_t, vectorized implementation have performance issue,
    // just use scalar path
    reduce_all_impl<int64_t>(result, input, lower_bound<int64_t>(),
      [=](int64_t a, int64_t b) -> int64_t { return max_impl(a, b); });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(input.scalar_type(), "max_all", [&] {
      using Vec = vec256::Vec256<scalar_t>;
      reduce_all_impl_vec<scalar_t>(result, input, lower_bound<scalar_t>(),
        [=] (scalar_t a , scalar_t b) -> scalar_t { return max_impl(a, b); },
        [=](Vec a, Vec b) -> Vec { return maximum(a, b); });
    });
  }
}

} // namespace

REGISTER_DISPATCH(min_all_stub, &min_all_kernel_impl);
REGISTER_DISPATCH(max_all_stub, &max_all_kernel_impl);

}}
