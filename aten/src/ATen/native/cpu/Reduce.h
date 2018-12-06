#pragma once

#include <ATen/native/cpu/Loops.h>
#include <ATen/Parallel.h>

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

template <typename func_t, typename vec_func_t>
void binary_kernel_reduce_vec(TensorIterator& iter, func_t op, vec_func_t vop, double ident=0) {
  using traits = binary_function_traits<func_t>;
  static_assert(
    std::is_same<typename traits::result_type, typename traits::arg1_t>::value,
    "all types must match");
  static_assert(
    std::is_same<typename traits::result_type, typename traits::arg2_t>::value,
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
