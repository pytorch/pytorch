#include <c10/metal/atomic.h>
#include <c10/metal/error.h>
#include <c10/metal/indexing.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

// Metal kernels implementing PyTorch's scatter with reduce='set'.
//
// All three kernels accept a `tid_offset` baked into the linear thread
// index. The host chunks dispatch into <=UINT_MAX-thread launches so that
// scatter works on tensors with > 2^32 index elements (Metal's
// `[[thread_position_in_grid]]` is at most a `uint`).

// Strided generic version for non-contiguous tensors.
template <typename T, typename index_t>
kernel void scatter_set(
    device T* output [[buffer(0)]],
    constant T* src [[buffer(1)]],
    constant index_t* index [[buffer(2)]],
    constant long* index_sizes [[buffer(3)]],
    constant long* output_strides [[buffer(4)]],
    constant long* src_strides [[buffer(5)]],
    constant long* index_strides [[buffer(6)]],
    constant uint3& ndim_dim [[buffer(7)]],
    constant long& dim_size [[buffer(8)]],
    constant long& tid_offset [[buffer(9)]],
    device ErrorMessages* error_buf [[buffer(10)]],
    uint thread_index [[thread_position_in_grid]]) {
  const uint ndim = ndim_dim.x;
  const uint dim = ndim_dim.y;
  const long tid = long(thread_index) + tid_offset;

  ::metal::array<long, max_ndim> pos;
  pos_from_thread_index<long>(tid, &pos[0], index_sizes, ndim);

  const long index_offs = offset_from_coord<long>(&pos[0], index_strides, ndim);
  long idx = long(index[index_offs]);
  if (idx < 0 || idx >= dim_size) {
    TORCH_REPORT_ERROR(
        error_buf,
        "scatter: index ",
        idx,
        " is out of bounds for dimension ",
        long(dim),
        " with size ",
        dim_size);
    return;
  }

  const long src_offs = offset_from_coord<long>(&pos[0], src_strides, ndim);
  pos[dim] = idx;
  const long out_offs = offset_from_coord<long>(&pos[0], output_strides, ndim);
  output[out_offs] = src[src_offs];
}

// Fast path: contiguous output/src/index, src and index share shape,
// shape matches output outside `dim`. Indexing collapses to one div + one
// mod per thread. inner_size = prod(output.sizes()[dim+1:]) precomputed
// on the host.
template <typename T, typename index_t>
kernel void scatter_set_dense(
    device T* output [[buffer(0)]],
    constant T* src [[buffer(1)]],
    constant index_t* index [[buffer(2)]],
    constant long& inner_size [[buffer(3)]],
    constant long& index_dim_size [[buffer(4)]],
    constant long& output_dim_size [[buffer(5)]],
    constant long& tid_offset [[buffer(6)]],
    device ErrorMessages* error_buf [[buffer(7)]],
    uint thread_index [[thread_position_in_grid]]) {
  const long tid = long(thread_index) + tid_offset;
  long idx = long(index[tid]);
  if (idx < 0 || idx >= output_dim_size) {
    TORCH_REPORT_ERROR(
        error_buf,
        "scatter: index ",
        idx,
        " is out of bounds for dimension with size ",
        output_dim_size);
    return;
  }
  const long inner = tid % inner_size;
  const long outer = tid / (inner_size * index_dim_size);
  const long out_offset =
      outer * (inner_size * output_dim_size) + idx * inner_size + inner;
  output[out_offset] = src[tid];
}

// Same as `scatter_set_dense` but the source is a single scalar value,
// avoiding the per-thread src read and the host-side `at::empty + fill_`
// the legacy MPSGraph path needed for `scatter.value`.
template <typename T, typename index_t>
kernel void scatter_set_dense_value(
    device T* output [[buffer(0)]],
    constant T& value [[buffer(1)]],
    constant index_t* index [[buffer(2)]],
    constant long& inner_size [[buffer(3)]],
    constant long& index_dim_size [[buffer(4)]],
    constant long& output_dim_size [[buffer(5)]],
    constant long& tid_offset [[buffer(6)]],
    device ErrorMessages* error_buf [[buffer(7)]],
    uint thread_index [[thread_position_in_grid]]) {
  const long tid = long(thread_index) + tid_offset;
  long idx = long(index[tid]);
  if (idx < 0 || idx >= output_dim_size) {
    TORCH_REPORT_ERROR(
        error_buf,
        "scatter: index ",
        idx,
        " is out of bounds for dimension with size ",
        output_dim_size);
    return;
  }
  const long inner = tid % inner_size;
  const long outer = tid / (inner_size * index_dim_size);
  const long out_offset =
      outer * (inner_size * output_dim_size) + idx * inner_size + inner;
  output[out_offset] = value;
}

// Reduce modes for the atomic scatter kernels below. Selected at host-side
// pipeline lookup via the kernel name suffix.
enum class ScatterReduceOp { Add, Prod, Amin, Amax };

template <typename T>
inline T scatter_op_prod(T a, T b) {
  return c10::metal::mul(a, b);
}
template <typename T>
inline T scatter_op_amin(T a, T b) {
  return ::c10::metal::min(a, b);
}
template <typename T>
inline T scatter_op_amax(T a, T b) {
  return ::c10::metal::max(a, b);
}

// Tag-dispatch the atomic operation so we only instantiate the AtomicType
// member that actually exists for a given T (e.g. AtomicType<long> has only
// atomic_add, not atomic_binary_op).
template <typename T, ScatterReduceOp Op>
struct ScatterAtomicApply;

template <typename T>
struct ScatterAtomicApply<T, ScatterReduceOp::Add> {
  static inline void apply(device T* output, long out_offset, T src_val) {
    AtomicType<T>::atomic_add(
        reinterpret_cast<device AtomicType_t<T>*>(output), out_offset, src_val);
  }
};

template <typename T>
struct ScatterAtomicApply<T, ScatterReduceOp::Prod> {
  static inline void apply(device T* output, long out_offset, T src_val) {
    AtomicType<T>::atomic_binary_op(
        reinterpret_cast<device AtomicType_t<T>*>(output),
        out_offset,
        src_val,
        scatter_op_prod<T>);
  }
};

template <typename T>
struct ScatterAtomicApply<T, ScatterReduceOp::Amin> {
  static inline void apply(device T* output, long out_offset, T src_val) {
    AtomicType<T>::atomic_binary_op(
        reinterpret_cast<device AtomicType_t<T>*>(output),
        out_offset,
        src_val,
        scatter_op_amin<T>);
  }
};

template <typename T>
struct ScatterAtomicApply<T, ScatterReduceOp::Amax> {
  static inline void apply(device T* output, long out_offset, T src_val) {
    AtomicType<T>::atomic_binary_op(
        reinterpret_cast<device AtomicType_t<T>*>(output),
        out_offset,
        src_val,
        scatter_op_amax<T>);
  }
};

// Signed int64 amin/amax via Metal's atomic_min/max on ulong (Metal 3.1+).
// The output buffer is pre-encoded (sign bit XOR'd, done by the host
// scatter_signbit_xor_long kernel) so signed order matches unsigned order;
// per-thread we encode the src value the same way.
template <>
struct ScatterAtomicApply<long, ScatterReduceOp::Amin> {
  static inline void apply(device long* output, long out_offset, long src_val) {
    const ulong encoded = ulong(src_val) ^ (1ul << 63);
    ::metal::atomic_min_explicit(
        reinterpret_cast<device ::metal::atomic<ulong>*>(output) + out_offset,
        encoded,
        ::metal::memory_order_relaxed);
  }
};
template <>
struct ScatterAtomicApply<long, ScatterReduceOp::Amax> {
  static inline void apply(device long* output, long out_offset, long src_val) {
    const ulong encoded = ulong(src_val) ^ (1ul << 63);
    ::metal::atomic_max_explicit(
        reinterpret_cast<device ::metal::atomic<ulong>*>(output) + out_offset,
        encoded,
        ::metal::memory_order_relaxed);
  }
};

// Toggle the sign bit of every element. Used as the pre- and post-pass that
// brackets a signed int64 amin/amax scatter (the encoding is its own inverse).
kernel void scatter_signbit_xor_long(
    device ulong* data [[buffer(0)]],
    constant long& tid_offset [[buffer(1)]],
    uint thread_index [[thread_position_in_grid]]) {
  const long tid = long(thread_index) + tid_offset;
  data[tid] ^= (1ul << 63);
}

// Dense atomic scatter for one of {add, prod, amin, amax}. Shape requirements
// match scatter_set_dense.
template <typename T, typename index_t, ScatterReduceOp Op>
kernel void scatter_reduce_dense(
    device T* output [[buffer(0)]],
    constant T* src [[buffer(1)]],
    constant index_t* index [[buffer(2)]],
    constant long& inner_size [[buffer(3)]],
    constant long& index_dim_size [[buffer(4)]],
    constant long& output_dim_size [[buffer(5)]],
    constant long& tid_offset [[buffer(6)]],
    device ErrorMessages* error_buf [[buffer(7)]],
    uint thread_index [[thread_position_in_grid]]) {
  const long tid = long(thread_index) + tid_offset;
  long idx = long(index[tid]);
  if (idx < 0 || idx >= output_dim_size) {
    TORCH_REPORT_ERROR(
        error_buf,
        "scatter: index ",
        idx,
        " is out of bounds for dimension with size ",
        output_dim_size);
    return;
  }
  const long inner = tid % inner_size;
  const long outer = tid / (inner_size * index_dim_size);
  const long out_offset =
      outer * (inner_size * output_dim_size) + idx * inner_size + inner;
  ScatterAtomicApply<T, Op>::apply(output, out_offset, src[tid]);
}

// Strided atomic scatter. Generic for non-contiguous output/src/index.
template <typename T, typename index_t, ScatterReduceOp Op>
kernel void scatter_reduce_strided(
    device T* output [[buffer(0)]],
    constant T* src [[buffer(1)]],
    constant index_t* index [[buffer(2)]],
    constant long* index_sizes [[buffer(3)]],
    constant long* output_strides [[buffer(4)]],
    constant long* src_strides [[buffer(5)]],
    constant long* index_strides [[buffer(6)]],
    constant uint3& ndim_dim [[buffer(7)]],
    constant long& dim_size [[buffer(8)]],
    constant long& tid_offset [[buffer(9)]],
    device ErrorMessages* error_buf [[buffer(10)]],
    uint thread_index [[thread_position_in_grid]]) {
  const uint ndim = ndim_dim.x;
  const uint dim = ndim_dim.y;
  const long tid = long(thread_index) + tid_offset;

  ::metal::array<long, max_ndim> pos;
  pos_from_thread_index<long>(tid, &pos[0], index_sizes, ndim);

  const long index_offs = offset_from_coord<long>(&pos[0], index_strides, ndim);
  long idx = long(index[index_offs]);
  if (idx < 0 || idx >= dim_size) {
    TORCH_REPORT_ERROR(
        error_buf,
        "scatter: index ",
        idx,
        " is out of bounds for dimension ",
        long(dim),
        " with size ",
        dim_size);
    return;
  }

  const long src_offs = offset_from_coord<long>(&pos[0], src_strides, ndim);
  pos[dim] = idx;
  const long out_offs = offset_from_coord<long>(&pos[0], output_strides, ndim);
  ScatterAtomicApply<T, Op>::apply(output, out_offs, src[src_offs]);
}

// gather: for each coord in `index` (= output) shape, read input at
// (coord with coord[dim] replaced by index[coord]) and write to output[coord].
// One write per output element -> no atomics needed.

// Dense fast path: output and index contiguous, index.shape == output.shape,
// input.shape == output.shape outside dim (no slicing).
template <typename T, typename index_t>
kernel void gather_dense(
    device T* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant index_t* index [[buffer(2)]],
    constant long& inner_size [[buffer(3)]],
    constant long& output_dim_size [[buffer(4)]],
    constant long& input_dim_size [[buffer(5)]],
    constant long& tid_offset [[buffer(6)]],
    device ErrorMessages* error_buf [[buffer(7)]],
    uint thread_index [[thread_position_in_grid]]) {
  const long tid = long(thread_index) + tid_offset;
  long idx = long(index[tid]);
  if (idx < 0 || idx >= input_dim_size) {
    TORCH_REPORT_ERROR(
        error_buf,
        "gather: index ",
        idx,
        " is out of bounds for dimension with size ",
        input_dim_size);
    output[tid] = T(0);
    return;
  }
  const long inner = tid % inner_size;
  const long outer = tid / (inner_size * output_dim_size);
  const long input_offset =
      outer * (inner_size * input_dim_size) + idx * inner_size + inner;
  output[tid] = input[input_offset];
}

// Strided generic gather. Iterates over `index` (= output) shape per thread.
template <typename T, typename index_t>
kernel void gather_strided(
    device T* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant index_t* index [[buffer(2)]],
    constant long* sizes [[buffer(3)]],
    constant long* output_strides [[buffer(4)]],
    constant long* input_strides [[buffer(5)]],
    constant long* index_strides [[buffer(6)]],
    constant uint3& ndim_dim [[buffer(7)]],
    constant long& input_dim_size [[buffer(8)]],
    constant long& tid_offset [[buffer(9)]],
    device ErrorMessages* error_buf [[buffer(10)]],
    uint thread_index [[thread_position_in_grid]]) {
  const uint ndim = ndim_dim.x;
  const uint dim = ndim_dim.y;
  const long tid = long(thread_index) + tid_offset;

  ::metal::array<long, max_ndim> pos;
  pos_from_thread_index<long>(tid, &pos[0], sizes, ndim);

  const long out_offs = offset_from_coord<long>(&pos[0], output_strides, ndim);
  const long index_offs = offset_from_coord<long>(&pos[0], index_strides, ndim);
  long idx = long(index[index_offs]);
  if (idx < 0 || idx >= input_dim_size) {
    TORCH_REPORT_ERROR(
        error_buf,
        "gather: index ",
        idx,
        " is out of bounds for dimension ",
        long(dim),
        " with size ",
        input_dim_size);
    output[out_offs] = T(0);
    return;
  }
  pos[dim] = idx;
  const long input_offs = offset_from_coord<long>(&pos[0], input_strides, ndim);
  output[out_offs] = input[input_offs];
}

#define REGISTER_SCATTER_REDUCE_VARIANT(DTYPE, IDXTYPE, OP, OP_ENUM)          \
  template                                                                    \
      [[host_name("scatter_" #OP "_dense_" #DTYPE "_" #IDXTYPE)]] kernel void \
      scatter_reduce_dense<DTYPE, IDXTYPE, ScatterReduceOp::OP_ENUM>(         \
          device DTYPE * output [[buffer(0)]],                                \
          constant DTYPE * src [[buffer(1)]],                                 \
          constant IDXTYPE * index [[buffer(2)]],                             \
          constant long& inner_size [[buffer(3)]],                            \
          constant long& index_dim_size [[buffer(4)]],                        \
          constant long& output_dim_size [[buffer(5)]],                       \
          constant long& tid_offset [[buffer(6)]],                            \
          device ErrorMessages* error_buf [[buffer(7)]],                      \
          uint thread_index [[thread_position_in_grid]]);                     \
  template [[host_name("scatter_" #OP "_strided_" #DTYPE                      \
                       "_" #IDXTYPE)]] kernel void                            \
  scatter_reduce_strided<DTYPE, IDXTYPE, ScatterReduceOp::OP_ENUM>(           \
      device DTYPE * output [[buffer(0)]],                                    \
      constant DTYPE * src [[buffer(1)]],                                     \
      constant IDXTYPE * index [[buffer(2)]],                                 \
      constant long* index_sizes [[buffer(3)]],                               \
      constant long* output_strides [[buffer(4)]],                            \
      constant long* src_strides [[buffer(5)]],                               \
      constant long* index_strides [[buffer(6)]],                             \
      constant uint3& ndim_dim [[buffer(7)]],                                 \
      constant long& dim_size [[buffer(8)]],                                  \
      constant long& tid_offset [[buffer(9)]],                                \
      device ErrorMessages* error_buf [[buffer(10)]],                         \
      uint thread_index [[thread_position_in_grid]])

#define REGISTER_SCATTER_SET_OP(DTYPE, IDXTYPE)                                \
  template [[host_name("scatter_set_" #DTYPE "_" #IDXTYPE)]] kernel void       \
  scatter_set<DTYPE, IDXTYPE>(                                                 \
      device DTYPE * output [[buffer(0)]],                                     \
      constant DTYPE * src [[buffer(1)]],                                      \
      constant IDXTYPE * index [[buffer(2)]],                                  \
      constant long* index_sizes [[buffer(3)]],                                \
      constant long* output_strides [[buffer(4)]],                             \
      constant long* src_strides [[buffer(5)]],                                \
      constant long* index_strides [[buffer(6)]],                              \
      constant uint3& ndim_dim [[buffer(7)]],                                  \
      constant long& dim_size [[buffer(8)]],                                   \
      constant long& tid_offset [[buffer(9)]],                                 \
      device ErrorMessages* error_buf [[buffer(10)]],                          \
      uint thread_index [[thread_position_in_grid]]);                          \
  template [[host_name("scatter_set_dense_" #DTYPE "_" #IDXTYPE)]] kernel void \
  scatter_set_dense<DTYPE, IDXTYPE>(                                           \
      device DTYPE * output [[buffer(0)]],                                     \
      constant DTYPE * src [[buffer(1)]],                                      \
      constant IDXTYPE * index [[buffer(2)]],                                  \
      constant long& inner_size [[buffer(3)]],                                 \
      constant long& index_dim_size [[buffer(4)]],                             \
      constant long& output_dim_size [[buffer(5)]],                            \
      constant long& tid_offset [[buffer(6)]],                                 \
      device ErrorMessages* error_buf [[buffer(7)]],                           \
      uint thread_index [[thread_position_in_grid]]);                          \
  template [[host_name("scatter_set_dense_value_" #DTYPE                       \
                       "_" #IDXTYPE)]] kernel void                             \
  scatter_set_dense_value<DTYPE, IDXTYPE>(                                     \
      device DTYPE * output [[buffer(0)]],                                     \
      constant DTYPE & value [[buffer(1)]],                                    \
      constant IDXTYPE * index [[buffer(2)]],                                  \
      constant long& inner_size [[buffer(3)]],                                 \
      constant long& index_dim_size [[buffer(4)]],                             \
      constant long& output_dim_size [[buffer(5)]],                            \
      constant long& tid_offset [[buffer(6)]],                                 \
      device ErrorMessages* error_buf [[buffer(7)]],                           \
      uint thread_index [[thread_position_in_grid]])

#define REGISTER_SCATTER_SET_DTYPE(DTYPE) \
  REGISTER_SCATTER_SET_OP(DTYPE, long);   \
  REGISTER_SCATTER_SET_OP(DTYPE, int)

#define REGISTER_GATHER_OP(DTYPE, IDXTYPE)                                  \
  template [[host_name("gather_dense_" #DTYPE "_" #IDXTYPE)]] kernel void   \
  gather_dense<DTYPE, IDXTYPE>(                                             \
      device DTYPE * output [[buffer(0)]],                                  \
      constant DTYPE * input [[buffer(1)]],                                 \
      constant IDXTYPE * index [[buffer(2)]],                               \
      constant long& inner_size [[buffer(3)]],                              \
      constant long& output_dim_size [[buffer(4)]],                         \
      constant long& input_dim_size [[buffer(5)]],                          \
      constant long& tid_offset [[buffer(6)]],                              \
      device ErrorMessages* error_buf [[buffer(7)]],                        \
      uint thread_index [[thread_position_in_grid]]);                       \
  template [[host_name("gather_strided_" #DTYPE "_" #IDXTYPE)]] kernel void \
  gather_strided<DTYPE, IDXTYPE>(                                           \
      device DTYPE * output [[buffer(0)]],                                  \
      constant DTYPE * input [[buffer(1)]],                                 \
      constant IDXTYPE * index [[buffer(2)]],                               \
      constant long* sizes [[buffer(3)]],                                   \
      constant long* output_strides [[buffer(4)]],                          \
      constant long* input_strides [[buffer(5)]],                           \
      constant long* index_strides [[buffer(6)]],                           \
      constant uint3& ndim_dim [[buffer(7)]],                               \
      constant long& input_dim_size [[buffer(8)]],                          \
      constant long& tid_offset [[buffer(9)]],                              \
      device ErrorMessages* error_buf [[buffer(10)]],                       \
      uint thread_index [[thread_position_in_grid]])

#define REGISTER_GATHER_DTYPE(DTYPE) \
  REGISTER_GATHER_OP(DTYPE, long);   \
  REGISTER_GATHER_OP(DTYPE, int)

#define REGISTER_SCATTER_REDUCE_FOR_OP(DTYPE, OP, OP_ENUM)   \
  REGISTER_SCATTER_REDUCE_VARIANT(DTYPE, long, OP, OP_ENUM); \
  REGISTER_SCATTER_REDUCE_VARIANT(DTYPE, int, OP, OP_ENUM)

// add: all numeric dtypes (atomic_add covers float/half/bfloat/int/short/
// char/uchar/bool/long via the two-word trick, plus float2/half2 for complex).
#define REGISTER_SCATTER_ADD_DTYPE(DTYPE) \
  REGISTER_SCATTER_REDUCE_FOR_OP(DTYPE, add, Add)

// prod/amin/amax use atomic_binary_op (CAS loop). Restricted to dtypes for
// which AtomicType<T> provides atomic_binary_op: float, int, half, bfloat,
// short, char, uchar, bool. amin/amax for complex is undefined and rejected
// at the meta-func level. long has its own specialization for amin/amax (via
// Metal's 64-bit atomic_min/max on ulong with sign-flip encoding) but no
// equivalent for prod.
#define REGISTER_SCATTER_PROD_MIN_MAX_DTYPE(DTYPE)   \
  REGISTER_SCATTER_REDUCE_FOR_OP(DTYPE, prod, Prod); \
  REGISTER_SCATTER_REDUCE_FOR_OP(DTYPE, amin, Amin); \
  REGISTER_SCATTER_REDUCE_FOR_OP(DTYPE, amax, Amax)

// long: amin/amax only (no atomic prod for 64-bit).
#define REGISTER_SCATTER_AMIN_AMAX_DTYPE(DTYPE)      \
  REGISTER_SCATTER_REDUCE_FOR_OP(DTYPE, amin, Amin); \
  REGISTER_SCATTER_REDUCE_FOR_OP(DTYPE, amax, Amax)

// Dtypes that support all of {gather, set, add, prod, amin, amax}: float, int,
// half, bfloat, short, char, uchar, bool -- AtomicType provides both atomic_add
// and atomic_binary_op for these.
#define REGISTER_SCATTER_OPS_FULL(DTYPE) \
  REGISTER_GATHER_DTYPE(DTYPE);          \
  REGISTER_SCATTER_SET_DTYPE(DTYPE);     \
  REGISTER_SCATTER_ADD_DTYPE(DTYPE);     \
  REGISTER_SCATTER_PROD_MIN_MAX_DTYPE(DTYPE)

// Dtypes that support {gather, set, add} only: long (atomic_add is eventually
// consistent, no true 64-bit CAS for atomic_binary_op), float2/half2
// (complex; amin/amax undefined).
#define REGISTER_SCATTER_OPS_SET_ADD(DTYPE) \
  REGISTER_GATHER_DTYPE(DTYPE);             \
  REGISTER_SCATTER_SET_DTYPE(DTYPE);        \
  REGISTER_SCATTER_ADD_DTYPE(DTYPE)

REGISTER_SCATTER_OPS_FULL(float);
REGISTER_SCATTER_OPS_FULL(half);
REGISTER_SCATTER_OPS_FULL(bfloat);
REGISTER_SCATTER_OPS_FULL(int);
REGISTER_SCATTER_OPS_FULL(short);
REGISTER_SCATTER_OPS_FULL(char);
REGISTER_SCATTER_OPS_FULL(uchar);
REGISTER_SCATTER_OPS_FULL(bool);

REGISTER_SCATTER_OPS_SET_ADD(long);
REGISTER_SCATTER_AMIN_AMAX_DTYPE(long);
REGISTER_SCATTER_OPS_SET_ADD(float2);
REGISTER_SCATTER_OPS_SET_ADD(half2);
