#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/ReduceOpsUtils.h>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/OpMathType.h>

#include <ATen/native/cpu/Loops.h>
#include <ATen/native/cpu/zmath.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>

namespace at::native {
namespace {

using namespace vec;

template <typename scalar_t, typename func_t, typename vec_func_t>
inline void reduce_all_impl_vec(
    Tensor& output,
    const Tensor& input,
    const scalar_t ident_v,
    func_t op,
    vec_func_t vop) {
  using Vec = Vectorized<opmath_type<scalar_t>>;
  const int64_t input_numel = input.numel();
  auto input_data = input.data_ptr<scalar_t>();
  // NOTE: parallel_reduce not support bool type
  scalar_t result = at::parallel_reduce(0, input_numel, internal::GRAIN_SIZE, ident_v,
    [&](int64_t start, int64_t end, const scalar_t /*ident*/) -> scalar_t {
      scalar_t partial_out = vec::reduce_all<scalar_t>(
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
      for (const auto i : c10::irange(start, end)) {
         partial_out = op(partial_out, input_data[i]);
      }
      return partial_out;
    }, op);
  output.fill_(result);
}

static void min_all_kernel_impl(Tensor& result, const Tensor& input) {
  if (input.scalar_type() == ScalarType::Bool) {
    TensorIterator iter = TensorIteratorConfig()
      .add_input(input)
      .build();
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
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(), "min_all", [&] {
      using Vec = Vectorized<opmath_type<scalar_t>>;
      reduce_all_impl_vec<scalar_t>(result, input, upper_bound<scalar_t>(),
        [=] (scalar_t a , scalar_t b) -> scalar_t { return min_impl(a, b); },
        [=](Vec a, Vec b) -> Vec { return minimum(a, b); });
    });
  }
}

static void max_all_kernel_impl(Tensor& result, const Tensor& input) {
  if (input.scalar_type() == ScalarType::Bool) {
    TensorIterator iter = TensorIteratorConfig()
      .add_input(input)
      .build();
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
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(), "max_all", [&] {
      using Vec = Vectorized<opmath_type<scalar_t>>;
      reduce_all_impl_vec<scalar_t>(result, input, lower_bound<scalar_t>(),
        [=] (scalar_t a , scalar_t b) -> scalar_t { return max_impl(a, b); },
        [=](Vec a, Vec b) -> Vec { return maximum(a, b); });
    });
  }
}

// For operation not support in avx/avx2
template <typename scalar_t, typename func_t1, typename func_t2>
inline void reduce_all_impl_two_outputs(
    Tensor& output1,
    Tensor& output2,
    const Tensor& input,
    const std::pair<scalar_t, scalar_t>& ident_v,
    func_t1 reduce_chunk_func,
    func_t2 reduce_acc_func) {
  using scalar_t_pair = std::pair<scalar_t, scalar_t>;
  const int64_t input_numel = input.numel();
  auto input_data = input.data_ptr<scalar_t>();
  scalar_t_pair result = at::parallel_reduce(0, input_numel, internal::GRAIN_SIZE, ident_v,
    [&](int64_t start, int64_t end, const scalar_t_pair& ident) -> scalar_t_pair {
      scalar_t_pair partial_out(ident);
      for (const auto i : c10::irange(start, end)) {
         partial_out = reduce_chunk_func(partial_out, input_data[i]);
      }
      return partial_out;
    },
    reduce_acc_func
  );
  output1.fill_(result.first);
  output2.fill_(result.second);
}

template <typename scalar_t, typename func_t, typename vec_func_t1, typename vec_func_t2>
inline void reduce_all_impl_vec_two_outputs(
    Tensor& output1,
    Tensor& output2,
    const Tensor& input,
    const std::pair<scalar_t, scalar_t>& ident_v,
    func_t reduce_acc_func,
    vec_func_t1 reduce_chunk_func1,
    vec_func_t2 reduce_chunk_func2) {
  using Vec = Vectorized<opmath_type<scalar_t>>;
  using scalar_t_pair = std::pair<scalar_t, scalar_t>;
  const int64_t input_numel = input.numel();
  auto input_data = input.data_ptr<scalar_t>();
  // NOTE: parallel_reduce not support bool type
  std::pair<scalar_t, scalar_t> result = at::parallel_reduce(0, input_numel, internal::GRAIN_SIZE, ident_v,
    [&](int64_t start, int64_t end, const scalar_t_pair& /* ident */) -> scalar_t_pair {
    scalar_t_pair partial_out = vec::reduce2_all<scalar_t>(
        [=](Vec x, Vec y) { return reduce_chunk_func1(x, y); },
        [=](Vec x, Vec y) { return reduce_chunk_func2(x, y); },
        input_data + start,
        end - start);
      return partial_out;
    },
    reduce_acc_func
  );
  output1.fill_(result.first);
  output2.fill_(result.second);
}

static void aminmax_allreduce_kernel(
    const Tensor& input,
    Tensor& min_result,
    Tensor& max_result) {
  if (input.scalar_type() == ScalarType::Bool) {
    TensorIterator iter = TensorIteratorConfig()
      .add_input(input)
      .build();
    bool min_result_data = true;
    bool max_result_data = false;
    cpu_serial_kernel(iter, [&](const bool a) -> void {
      min_result_data = min_result_data && a;
      max_result_data = max_result_data || a;
    });
    min_result.fill_(min_result_data);
    max_result.fill_(max_result_data);
  } else if (input.scalar_type() == ScalarType::Long) {
    // for int64_t, vectorized implementation have performance issue,
    // just use scalar path
    using int64_t_pair = std::pair<int64_t, int64_t>;
    reduce_all_impl_two_outputs<int64_t>(min_result, max_result, input,
      int64_t_pair(upper_bound<int64_t>(), lower_bound<int64_t>()),
      // reduce over chunk
      [=](int64_t_pair a, int64_t b) -> int64_t_pair {
        return int64_t_pair(min_impl(a.first, b), max_impl(a.second, b));
      },
      // combine two inputs
      [=](int64_t_pair a, int64_t_pair b) -> int64_t_pair {
        return int64_t_pair(min_impl(a.first, b.first), max_impl(a.second, b.second));
      }
    );
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, input.scalar_type(), "aminmax_cpu", [&] {
      using Vec = Vectorized<opmath_type<scalar_t>>;
      using scalar_t_pair = std::pair<scalar_t, scalar_t>;
      reduce_all_impl_vec_two_outputs<scalar_t>(
        min_result,
        max_result,
        input,
        scalar_t_pair(upper_bound<scalar_t>(), lower_bound<scalar_t>()),
        [=] (scalar_t_pair a , scalar_t_pair b) -> scalar_t_pair {
          return scalar_t_pair(
            min_impl(a.first, b.first), max_impl(a.second, b.second));
        },
        [=](Vec a, Vec b) -> Vec { return minimum(a, b); },
        [=](Vec a, Vec b) -> Vec { return maximum(a, b); }
      );
    });
  }
}

} // namespace

REGISTER_DISPATCH(min_all_stub, &min_all_kernel_impl);
REGISTER_DISPATCH(max_all_stub, &max_all_kernel_impl);
REGISTER_DISPATCH(aminmax_allreduce_stub, &aminmax_allreduce_kernel);

} // namespace at::native
