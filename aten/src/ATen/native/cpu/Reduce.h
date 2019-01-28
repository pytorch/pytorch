#pragma once

#include <ATen/native/cpu/Loops.h>
#include <ATen/Parallel.h>
#include <c10/util/TypeList.h>

#include <sstream>

namespace at { namespace native { namespace {

using namespace vec256;

// reduction that is contiguous over the input in dim 0
template <typename traits>
static inline bool is_contiguous_reduction(const int64_t* strides) {
  return strides[0] == 0 &&
         strides[1] == sizeof(typename traits::arg2_t);
}

// reduction that is contiguous over the input in dim 1
template <typename traits>
static inline bool is_outer_reduction(const int64_t* strides) {
  return strides[0] == 0 &&
         strides[2] == sizeof(typename traits::result_type) &&
         strides[3] == sizeof(typename traits::arg2_t);
}

template <typename T, typename... Args>
struct all_same : c10::guts::conjunction<
  std::is_same<T, Args>...
> {};

// data_t is the input/output data type.
// acc_t is a type that contains all the necessary data
// to continue reducing.
//
// ops_t is such that &ops_t::reduce, &ops_t::combine, and &ops_t::project exist and satisfy
// the following.
// reduce: (acc_t, data_t) -> acc_t adds one data point to the accumulated value.
// combine: (acc_t, acc_t) -> acc_t combines two accumulated values into one.
// project: acc_t -> data_t finishes the reduction, getting the required output.
//
// Additionally, acc_t must be default-constructible:
// acc_t {} is an identity for combine,
// and project(acc_t {}) is the value of the operation on zero elements.
//
// The point of `combine` is to support parallelization -
// the idea is to one sequence of `reduce` calls per thread of execution,
// and then to combine them at the end with `combine`.
//
// If there is more than one output element,
// our parallelization strategy is to use one thread for each of them,
// which means that `combine` will never be called.
//
// If, on the other hand, there is only one, then we split the input into
// into several pieces, reduce each separately, and then combine them.

template <typename ops_t, typename init_t>
void binary_kernel_reduce(TensorIterator& iter, ops_t ops, init_t init) {
  using rf_t = decltype(&ops_t::reduce);
  using cf_t = decltype(&ops_t::combine);
  using pf_t = decltype(&ops_t::project);
  using r_traits = binary_function_traits<rf_t>;
  using c_traits = binary_function_traits<cf_t>;
  using p_traits = unary_function_traits<pf_t>;
  using acc_t = typename p_traits::arg1_t;
  using data_t = typename p_traits::result_type;
  static_assert(
    all_same<
      acc_t,
      init_t,
      typename r_traits::arg1_t,
      typename r_traits::result_type,
      typename c_traits::arg1_t,
      typename c_traits::arg2_t,
      typename c_traits::result_type>::value,
    "all accumulate types must match");
  static_assert(
    std::is_same<data_t, typename r_traits::arg2_t>::value,
    "all data types must match");
  static_assert(
    std::is_default_constructible<acc_t>::value,
    "the accumulate type must be default-constructible"
  );
  iter.foreach_reduced_elt([&](TensorIterator &sub_iter) {
    auto reduction_body = [&](acc_t acc, int64_t begin, int64_t end) -> acc_t {
      sub_iter.serial_for_each([&acc, &ops, &init](int ntensors, char** data, const int64_t* strides, int64_t size) {
        AT_ASSERT(ntensors == 2);
        char *in = data[1];
        int64_t stride = strides[1];
        for (int64_t i = 0; i < size; ++i) {
          acc = ops.reduce(acc, *(data_t*)in);
          in += stride;
        }
      }, {begin, end});
      return acc;
    };
    acc_t total_acc = init;
    auto numel = sub_iter.numel();
    if (numel < at::internal::GRAIN_SIZE || at::get_max_threads() == 1 || at::in_parallel_region()) {
      total_acc = reduction_body(total_acc, 0, numel);
    } else {
      int max_threads = at::get_max_threads();
      AT_ASSERT(max_threads > 0);
      static_assert(
        !std::is_same<acc_t, bool>::value,
        "Concurrently modifying different references into std::vector<bool> is UB."
      );
      std::vector<acc_t> buffer((unsigned)max_threads, init);
      at::parallel_for(0, numel, internal::GRAIN_SIZE,
        [&](int64_t begin, int64_t end) {
          auto& acc = buffer[at::get_thread_num()];
          acc = reduction_body(acc, begin, end);
        }
      );
      for (int i = 0; i < max_threads; ++i) {
        total_acc = ops.combine(total_acc, buffer[i]);
      }
    }
    char *out = (char *)sub_iter.data_ptr(0);
    *(data_t*)out = ops.project(total_acc);
  });
}

template <typename func_t, typename vec_func_t>
void binary_kernel_reduce_vec(TensorIterator& iter, func_t op, vec_func_t vop, double ident=0) {
  using traits = binary_function_traits<func_t>;
  static_assert(
    all_same<
      typename traits::result_type,
      typename traits::arg1_t,
      typename traits::arg2_t>::value,
    "all types must match");

  iter.output().fill_(ident);
  iter.parallel_reduce([&](int ntensor, char** data, const int64_t* strides, int64_t size0, int64_t size1) {
    int64_t outer_strides[] = { strides[2], strides[3] };
    if (is_contiguous_reduction<traits>(strides)) {
      // input is contiguous in dim 0, output is reduced in dim 0
      UNARY_OUTER_LOOP(data, outer_strides, size1, [&] {
        vectorized_inner_reduction(data, size0, op, vop);
      });
    } else if (is_outer_reduction<traits>(strides)) {
      // input and output are contiguous in dim 1
      int64_t inner_stride = strides[1]; // stride of input in dim 0
      vectorized_outer_reduction(data, inner_stride, size0, size1, op, vop);
    } else {
      UNARY_OUTER_LOOP(data, outer_strides, size1, [&] {
        char* ptrs[3] = { data[0], data[0], data[1] };
        int64_t inner_strides[3] = { strides[0], strides[0], strides[1] };
        binary_loop(ptrs, inner_strides, 0, size0, op);
      });
    }
  });
}

}}}  // namespace at::native::<anonymous>
