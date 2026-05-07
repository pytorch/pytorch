#include <ATen/native/mps/kernels/ScatterGather.h>
#include <c10/metal/atomic.h>
#include <c10/metal/indexing.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

// ── scatter with "set" mode (last write wins, matches CUDA semantics) ────────

template <typename T, typename IT>
kernel void scatter_set(
    device T* self [[buffer(0)]],
    device IT* index [[buffer(1)]],
    device T* src [[buffer(2)]],
    constant ScatterGatherParams<>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  uint32_t tid_ = tid;
  long src_offset = 0;
  long self_offset = 0;
  long index_offset = 0;

  for (int32_t d = params.ndim - 1; d >= 0; d--) {
    uint32_t size = params.index_sizes[d];
    uint32_t dim_idx = tid_ % size;

    src_offset += dim_idx * params.src_strides[d];
    index_offset += dim_idx * params.index_strides[d];

    if (d != params.dim) {
      self_offset += dim_idx * params.self_strides[d];
    }

    tid_ /= size;
  }

  uint32_t idx_val = static_cast<uint32_t>(index[index_offset]);
  self_offset += idx_val * params.self_strides[params.dim];

  self[self_offset] = src[src_offset];
}

// ── scatter_add (atomic add — fast path for the most common GNN op) ──────────

template <typename T, typename IT>
kernel void scatter_add(
    device AtomicType_t<T>* self [[buffer(0)]],
    device IT* index [[buffer(1)]],
    device T* src [[buffer(2)]],
    constant ScatterGatherParams<>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  uint32_t tid_ = tid;
  long src_offset = 0;
  long self_offset = 0;
  long index_offset = 0;

  for (int32_t d = params.ndim - 1; d >= 0; d--) {
    uint32_t size = params.index_sizes[d];
    uint32_t dim_idx = tid_ % size;

    src_offset += dim_idx * params.src_strides[d];
    index_offset += dim_idx * params.index_strides[d];

    if (d != params.dim) {
      self_offset += dim_idx * params.self_strides[d];
    }

    tid_ /= size;
  }

  uint32_t idx_val = static_cast<uint32_t>(index[index_offset]);
  self_offset += idx_val * params.self_strides[params.dim];

  AtomicType<T>::atomic_add(self, self_offset, src[src_offset]);
}

// ── scatter_reduce (generic reduce via atomic CAS) ───────────────────────────

struct ScatterReduceOp {
  template <typename T>
  static T prod(T a, T b) { return c10::metal::mul(a, b); }
  template <typename T>
  static T amin(T a, T b) { return min(a, b); }
  template <typename T>
  static T amax(T a, T b) { return max(a, b); }
};

template <typename T, typename IT, T (*ReduceOp)(T, T)>
kernel void scatter_reduce(
    device AtomicType_t<T>* self [[buffer(0)]],
    device IT* index [[buffer(1)]],
    device T* src [[buffer(2)]],
    constant ScatterGatherParams<>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  uint32_t tid_ = tid;
  long src_offset = 0;
  long self_offset = 0;
  long index_offset = 0;

  for (int32_t d = params.ndim - 1; d >= 0; d--) {
    uint32_t size = params.index_sizes[d];
    uint32_t dim_idx = tid_ % size;

    src_offset += dim_idx * params.src_strides[d];
    index_offset += dim_idx * params.index_strides[d];

    if (d != params.dim) {
      self_offset += dim_idx * params.self_strides[d];
    }

    tid_ /= size;
  }

  uint32_t idx_val = static_cast<uint32_t>(index[index_offset]);
  self_offset += idx_val * params.self_strides[params.dim];

  AtomicType<T>::atomic_binary_op(self, self_offset, src[src_offset], ReduceOp);
}

// ── gather (read-only, no atomics) ──────────────────────────────────────────

template <typename T, typename IT>
kernel void gather_kernel(
    device T* output [[buffer(0)]],
    device T* self [[buffer(1)]],
    device IT* index [[buffer(2)]],
    constant ScatterGatherParams<>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  uint32_t tid_ = tid;
  long output_offset = 0;
  long self_offset = 0;
  long index_offset = 0;

  for (int32_t d = params.ndim - 1; d >= 0; d--) {
    uint32_t size = params.index_sizes[d];
    uint32_t dim_idx = tid_ % size;

    output_offset += dim_idx * params.src_strides[d];
    index_offset += dim_idx * params.index_strides[d];

    if (d != params.dim) {
      self_offset += dim_idx * params.self_strides[d];
    }

    tid_ /= size;
  }

  uint32_t idx_val = static_cast<uint32_t>(index[index_offset]);
  self_offset += idx_val * params.self_strides[params.dim];

  output[output_offset] = self[self_offset];
}

// ── Instantiation ────────────────────────────────────────────────────────────

#define REGISTER_SCATTER_SET(T, IT)                                        \
  template [[host_name("scatter_set_" #T "_" #IT)]]                        \
  kernel void scatter_set<T, IT>(                                          \
      device T * self [[buffer(0)]],                                       \
      device IT * index [[buffer(1)]],                                     \
      device T * src [[buffer(2)]],                                        \
      constant ScatterGatherParams<> & params [[buffer(3)]],               \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_SCATTER_ADD(T, IT)                                        \
  template [[host_name("scatter_add_" #T "_" #IT)]]                        \
  kernel void scatter_add<T, IT>(                                          \
      device AtomicType_t<T> * self [[buffer(0)]],                         \
      device IT * index [[buffer(1)]],                                     \
      device T * src [[buffer(2)]],                                        \
      constant ScatterGatherParams<> & params [[buffer(3)]],               \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_SCATTER_REDUCE(ReduceOp, T, IT)                           \
  template [[host_name("scatter_" #ReduceOp "_" #T "_" #IT)]]             \
  kernel void scatter_reduce<T, IT, ScatterReduceOp::ReduceOp<T>>(        \
      device AtomicType_t<T> * self [[buffer(0)]],                         \
      device IT * index [[buffer(1)]],                                     \
      device T * src [[buffer(2)]],                                        \
      constant ScatterGatherParams<> & params [[buffer(3)]],               \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_GATHER(T, IT)                                             \
  template [[host_name("gather_" #T "_" #IT)]]                            \
  kernel void gather_kernel<T, IT>(                                        \
      device T * output [[buffer(0)]],                                     \
      device T * self [[buffer(1)]],                                       \
      device IT * index [[buffer(2)]],                                     \
      constant ScatterGatherParams<> & params [[buffer(3)]],               \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_ALL_OPS(T, IT)          \
  REGISTER_SCATTER_SET(T, IT)            \
  REGISTER_SCATTER_ADD(T, IT)            \
  REGISTER_SCATTER_REDUCE(prod, T, IT)   \
  REGISTER_SCATTER_REDUCE(amax, T, IT)   \
  REGISTER_SCATTER_REDUCE(amin, T, IT)   \
  REGISTER_GATHER(T, IT)

#define REGISTER_ALL_INDEX_TYPES(T) \
  REGISTER_ALL_OPS(T, int)          \
  REGISTER_ALL_OPS(T, long)

REGISTER_ALL_INDEX_TYPES(float);
REGISTER_ALL_INDEX_TYPES(half);
REGISTER_ALL_INDEX_TYPES(bfloat);
REGISTER_ALL_INDEX_TYPES(int);
REGISTER_ALL_INDEX_TYPES(short);
REGISTER_ALL_INDEX_TYPES(char);
REGISTER_ALL_INDEX_TYPES(uchar);
REGISTER_ALL_INDEX_TYPES(long);
