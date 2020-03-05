#include<ATen/native/ReduceAllOps.h>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/zmath.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/cpu/vec256/vec256.h>

namespace at { namespace native {
namespace {

using namespace vec256;

template <typename scalar_t, typename func_t, typename vec_func_t>
inline void reduce_all_impl(
    Tensor& output,
    const Tensor& input,
    scalar_t ident_v,
    func_t op,
    vec_func_t vop) {
  using Vec = Vec256<scalar_t>;
  int64_t input_numel = input.numel();
  auto input_data = input.data_ptr<scalar_t>();
  auto output_data = output.data_ptr<scalar_t>();
  scalar_t result = at::parallel_reduce(0, input_numel, internal::GRAIN_SIZE, ident_v, 
    [&](int64_t start, int64_t end, scalar_t ident) -> scalar_t {
      scalar_t partial_out = vec256::reduce_all<scalar_t>(
        [=](Vec x, Vec y) { return vop(x, y); },
        input_data + start,
        end - start);
      return partial_out;
    }, op);
  output_data[0] = result;
}

static void min_all_kernel_impl(Tensor& result, const Tensor& input) {
  if (input.scalar_type() == ScalarType::Bool) {
    reduce_all_impl<bool>(result, input, true,
      [](bool a, bool b) -> bool { return a && b; },
      [](Vec256<bool> a, Vec256<bool> b) {
        Vec256<bool> c = Vec256<bool>();
        for (int i = 0; i != Vec256<bool>::size(); i++) {
          c[i] = a[i] && b[i];
        }
        return c;
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(input.scalar_type(), "min_all", [&] {
      using Vec = vec256::Vec256<scalar_t>;
      reduce_all_impl<scalar_t>(result, input, upper_bound<scalar_t>(),
        [=] (scalar_t a , scalar_t b) -> scalar_t { return min_impl(a, b); },
        [=](Vec a, Vec b) -> Vec { return minimum(a, b); });
    });
  }
}

static void max_all_kernel_impl(Tensor& result, const Tensor& input) {
  if (input.scalar_type() == ScalarType::Bool) {
    reduce_all_impl<bool>(result, input, false,
      [](bool a, bool b) -> bool { return a || b; },
      [](Vec256<bool> a, Vec256<bool> b) {
        Vec256<bool> c = Vec256<bool>();
        for (int i = 0; i != Vec256<bool>::size(); i++) {
          c[i] = a[i] || b[i];
        }
        return c;
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(input.scalar_type(), "max_all", [&] {
      using Vec = vec256::Vec256<scalar_t>;
      reduce_all_impl<scalar_t>(result, input, lower_bound<scalar_t>(),
        [=] (scalar_t a , scalar_t b) -> scalar_t { return max_impl(a, b); },
        [=](Vec a, Vec b) -> Vec { return maximum(a, b); });
    });
  }
}

} // namespace

REGISTER_DISPATCH(min_all_stub, &min_all_kernel_impl);
REGISTER_DISPATCH(max_all_stub, &max_all_kernel_impl);

}}
