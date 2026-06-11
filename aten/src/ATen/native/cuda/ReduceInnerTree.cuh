#pragma once

namespace at::native {

template <typename acc_t, int vec_size, typename BinaryOp>
__device__ __forceinline__ acc_t linear_reduce(acc_t (&vals)[vec_size], BinaryOp op) {
  #pragma unroll
  for (int i = 1; i < vec_size; i++) {
    vals[0] = op(vals[0], vals[i]);
  }
  return vals[0];
}

template <typename acc_t, int vec_size, typename BinaryOp>
__device__ __forceinline__ acc_t inner_tree_reduce(acc_t (&vals)[vec_size], BinaryOp op) {
  #pragma unroll
  for (int stride = 1; stride < vec_size; stride *= 2) {
    #pragma unroll
    for (int i = 0; i + stride < vec_size; i += stride * 2) {
      vals[i] = op(vals[i], vals[i + stride]);
    }
  }
  return vals[0];
}

template <int max_depth, typename acc_t, typename BinaryOp>
__device__ __forceinline__ void streaming_inner_tree_reduce_step(
    acc_t* tree_accs, int& top, int load, acc_t carry, BinaryOp op) {
  int merges = __ffs(load + 1) - 1;
  #pragma unroll
  for (int m = 0; m < max_depth; m++) {
    if (m >= merges) break;
    carry = op(tree_accs[--top], carry);
  }
  tree_accs[top++] = carry;
}

struct AddOp {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

template <typename acc_t, int vec_size>
__device__ __forceinline__ acc_t linear_sum(acc_t (&vals)[vec_size]) {
  return linear_reduce(vals, AddOp{});
}

template <typename acc_t, int vec_size>
__device__ __forceinline__ acc_t inner_tree_sum(acc_t (&vals)[vec_size]) {
  return inner_tree_reduce(vals, AddOp{});
}

template <int max_depth, typename acc_t>
__device__ __forceinline__ void streaming_inner_tree_step(
    acc_t* tree_accs, int& top, int load, acc_t carry) {
  streaming_inner_tree_reduce_step<max_depth>(tree_accs, top, load, carry, AddOp{});
}

} // namespace at::native
