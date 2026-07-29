#include <ATen/native/mps/kernels/Indexing.h>
#include <c10/metal/atomic.h>
#include <c10/metal/error.h>
#include <c10/metal/indexing.h>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

struct IndexAB {
  constant int64_t* indexArray;
};

uint3 index_get_offsets(
    constant int64_t* sizes,
    constant int64_t* output_strides,
    constant int64_t* input_strides,
    constant int64_t* indices_strides,
    uint ndim,
    uint thread_index) {
  uint pos[max_ndim];
  pos_from_thread_index(thread_index, pos, sizes, ndim);
  const auto output_offs = offset_from_coord(pos, output_strides, ndim);
  const auto input_offs = offset_from_coord(pos, input_strides, ndim);
  const auto indices_offs =
      offset_from_coord(pos, indices_strides, ndim) / sizeof(int64_t);
  return uint3(output_offs, input_offs, indices_offs);
}

template <typename OffsetT>
OffsetT index_apply_indices(
    uint2 offs,
    constant IndexAB* indices,
    constant int64_t* sizes,
    constant int64_t* strides,
    uint num_indices,
    thread bool& error,
    device ErrorMessages* error_buf) {
  OffsetT rc = offs.x;
  for (uint i = 0; i < num_indices; i++) {
    auto idx = indices[i].indexArray[offs.y];
    if (idx < -sizes[i] || idx >= sizes[i]) {
      TORCH_REPORT_ERROR(
          error_buf,
          "index ",
          idx,
          " is out of bounds for dimension ",
          i,
          " with size ",
          sizes[i]);
      error = true;
      break;
    }
    if (idx < 0) {
      idx += sizes[i];
    }
    rc += idx * strides[i];
  }
  return rc;
}

template <typename T, typename OffsetT = ulong>
kernel void index_select(
    device T* output,
    constant T* input,
    constant IndexAB* indices,
    constant int64_t* sizes,
    constant int64_t* output_strides,
    constant int64_t* input_strides,
    constant int64_t* indices_strides,
    constant int64_t* index_sizes,
    constant int64_t* index_strides,
    constant uint4& ndim_nindices_numel,
    device ErrorMessages* error_buffer,
    uint thread_index [[thread_position_in_grid]]) {
  const auto ndim = ndim_nindices_numel.x;
  const auto num_indices = ndim_nindices_numel.y;
  const auto offs = index_get_offsets(
      sizes,
      output_strides,
      input_strides,
      indices_strides,
      ndim,
      thread_index);
  bool error = false;
  auto input_offs = index_apply_indices<OffsetT>(
      offs.yz,
      indices,
      index_sizes,
      index_strides,
      num_indices,
      error,
      error_buffer);
  if (error) {
    output[offs.x / sizeof(T)] = 0;
    return;
  }
  output[offs.x / sizeof(T)] = input[input_offs / sizeof(T)];
}

template <typename T, typename OffsetT = ulong>
inline void index_put_impl(
    device T* output,
    constant T* input,
    constant IndexAB* indices,
    constant int64_t* sizes,
    constant int64_t* output_strides,
    constant int64_t* input_strides,
    constant int64_t* indices_strides,
    constant int64_t* index_sizes,
    constant int64_t* index_strides,
    constant uint4& ndim_nindices_numel,
    device ErrorMessages* error_buffer,
    uint thread_index) {
  bool error = false;
  const auto ndim = ndim_nindices_numel.x;
  const auto num_indices = ndim_nindices_numel.y;
  const auto offs = index_get_offsets(
      sizes,
      output_strides,
      input_strides,
      indices_strides,
      ndim,
      thread_index);
  auto output_offs = index_apply_indices<OffsetT>(
      offs.xz,
      indices,
      index_sizes,
      index_strides,
      num_indices,
      error,
      error_buffer);
  if (error) {
    return;
  }
  output[output_offs / sizeof(T)] = input[offs.y / sizeof(T)];
}

template <typename T, typename OffsetT = ulong>
kernel void index_put(
    device T* output,
    constant T* input,
    constant IndexAB* indices,
    constant int64_t* sizes,
    constant int64_t* output_strides,
    constant int64_t* input_strides,
    constant int64_t* indices_strides,
    constant int64_t* index_sizes,
    constant int64_t* index_strides,
    constant uint4& ndim_nindices_numel,
    device ErrorMessages* error_buffer,
    uint thread_index [[thread_position_in_grid]]) {
  index_put_impl(
      output,
      input,
      indices,
      sizes,
      output_strides,
      input_strides,
      indices_strides,
      index_sizes,
      index_strides,
      ndim_nindices_numel,
      error_buffer,
      thread_index);
}

template <typename T, typename OffsetT = ulong>
kernel void index_put_serial(
    device T* output,
    constant T* input,
    constant IndexAB* indices,
    constant int64_t* sizes,
    constant int64_t* output_strides,
    constant int64_t* input_strides,
    constant int64_t* indices_strides,
    constant int64_t* index_sizes,
    constant int64_t* index_strides,
    constant uint4& ndim_nindices_numel,
    device ErrorMessages* error_buffer,
    uint thread_index [[thread_position_in_grid]]) {
  (void)thread_index; // Suppress unused variable warning
  for (uint idx = 0; idx < ndim_nindices_numel.z; ++idx) {
    index_put_impl(
        output,
        input,
        indices,
        sizes,
        output_strides,
        input_strides,
        indices_strides,
        index_sizes,
        index_strides,
        ndim_nindices_numel,
        error_buffer,
        idx);
  }
}

template <typename T, typename OffsetT = ulong>
kernel void index_put_accumulate(
    device T* output,
    constant T* input,
    constant IndexAB* indices,
    constant int64_t* sizes,
    constant int64_t* output_strides,
    constant int64_t* input_strides,
    constant int64_t* indices_strides,
    constant int64_t* index_sizes,
    constant int64_t* index_strides,
    constant uint4& ndim_nindices_numel,
    device ErrorMessages* error_buffer,
    uint thread_index [[thread_position_in_grid]]) {
  const auto ndim = ndim_nindices_numel.x;
  const auto num_indices = ndim_nindices_numel.y;
  const auto offs = index_get_offsets(
      sizes,
      output_strides,
      input_strides,
      indices_strides,
      ndim,
      thread_index);
  bool error = false;
  auto output_offs = index_apply_indices<OffsetT>(
      offs.xz,
      indices,
      index_sizes,
      index_strides,
      num_indices,
      error,
      error_buffer);
  if (error) {
    return;
  }
  AtomicType<T>::atomic_add(
      reinterpret_cast<device AtomicType_t<T>*>(output),
      output_offs / sizeof(T),
      input[offs.y / sizeof(T)]);
}

#define REGISTER_INDEX_OP(OP_NAME, SUFFIX, DTYPE)                   \
  template [[host_name("index_" #OP_NAME "_" #SUFFIX)]] kernel void \
      index_##OP_NAME<DTYPE>(                                       \
          device DTYPE * output,                                    \
          constant DTYPE * input,                                   \
          constant IndexAB * indices,                               \
          constant int64_t* sizes,                                  \
          constant int64_t* output_strides,                         \
          constant int64_t* input_strides,                          \
          constant int64_t* indices_strides,                        \
          constant int64_t* index_sizes,                            \
          constant int64_t* index_strides,                          \
          constant uint4& ndim_nindices_numel,                      \
          device ErrorMessages* error_buffer,                       \
          uint thread_index [[thread_position_in_grid]])

#define REGISTER_INDEX_OP_ALL_DTYPES(OP_NAME) \
  REGISTER_INDEX_OP(OP_NAME, 8bit, char);     \
  REGISTER_INDEX_OP(OP_NAME, 16bit, short);   \
  REGISTER_INDEX_OP(OP_NAME, 32bit, int);     \
  REGISTER_INDEX_OP(OP_NAME, 64bit, long)

REGISTER_INDEX_OP_ALL_DTYPES(select);
REGISTER_INDEX_OP_ALL_DTYPES(put);
REGISTER_INDEX_OP_ALL_DTYPES(put_serial);

REGISTER_INDEX_OP(put_accumulate, float, float);
REGISTER_INDEX_OP(put_accumulate, half, half);
REGISTER_INDEX_OP(put_accumulate, bfloat, bfloat);
REGISTER_INDEX_OP(put_accumulate, long, long);
REGISTER_INDEX_OP(put_accumulate, int, int);
REGISTER_INDEX_OP(put_accumulate, short, short);
REGISTER_INDEX_OP(put_accumulate, char, char);
REGISTER_INDEX_OP(put_accumulate, uchar, uchar);
REGISTER_INDEX_OP(put_accumulate, bool, bool);
REGISTER_INDEX_OP(put_accumulate, float2, float2);
REGISTER_INDEX_OP(put_accumulate, half2, half2);

struct IndexReduceOp {
  template <typename T>
  static T prod(T a, T b) {
    return c10::metal::mul(a, b);
  }

  template <typename T>
  static T mean(T a, T b) {
    return a + b;
  }

  template <typename T>
  static T amin(T a, T b) {
    return min(a, b);
  }

  template <typename T>
  static T amax(T a, T b) {
    return max(a, b);
  }
};

template <typename T, typename IT, T (*ReduceOp)(T, T)>
kernel void index_reduce(
    device AtomicType_t<T>* self [[buffer(0)]],
    device IT* index [[buffer(1)]],
    device T* source [[buffer(2)]],
    constant IndexReduceParams<>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  uint32_t tid_ = tid;
  long source_offset = 0;
  long self_offset = 0;

  for (int32_t dim = params.ndim - 1; dim >= 0; dim--) {
    auto source_size = params.source_sizes[dim];
    auto dim_idx = tid_ % source_size;

    source_offset += dim_idx * params.source_strides[dim];

    if (dim == params.reduce_dim) {
      uint32_t self_dim_idx =
          static_cast<uint32_t>(index[dim_idx * params.index_stride]);
      self_offset += self_dim_idx * params.self_strides[dim];
    } else {
      self_offset += dim_idx * params.self_strides[dim];
    }

    tid_ /= source_size;
  }

  T source_elem = source[source_offset];

  AtomicType<T>::atomic_binary_op(self, self_offset, source_elem, ReduceOp);
}

#define REGISTER_INDEX_REDUCE_OP(ReduceOp, T, IT)                  \
  template [[host_name("index_reduce_" #ReduceOp "_" #T "_" #IT)]] \
  kernel void index_reduce<T, IT, IndexReduceOp::ReduceOp<T>>(     \
      device AtomicType_t<T> * self [[buffer(0)]],                 \
      device IT * index [[buffer(1)]],                             \
      device T * source [[buffer(2)]],                             \
      constant IndexReduceParams<> & params [[buffer(3)]],         \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_INDEX_REDUCE_OP_ALL_REDUCE_TYPES(T, IT) \
  REGISTER_INDEX_REDUCE_OP(amax, T, IT);                 \
  REGISTER_INDEX_REDUCE_OP(mean, T, IT);                 \
  REGISTER_INDEX_REDUCE_OP(amin, T, IT);                 \
  REGISTER_INDEX_REDUCE_OP(prod, T, IT);

#define REGISTER_INDEX_REDUCE_OP_ALL_INDEX_TYPES(T)  \
  REGISTER_INDEX_REDUCE_OP_ALL_REDUCE_TYPES(T, int); \
  REGISTER_INDEX_REDUCE_OP_ALL_REDUCE_TYPES(T, long);

REGISTER_INDEX_REDUCE_OP_ALL_INDEX_TYPES(float);
REGISTER_INDEX_REDUCE_OP_ALL_INDEX_TYPES(half);
REGISTER_INDEX_REDUCE_OP_ALL_INDEX_TYPES(bfloat);
REGISTER_INDEX_REDUCE_OP_ALL_INDEX_TYPES(int);
REGISTER_INDEX_REDUCE_OP_ALL_INDEX_TYPES(short);
REGISTER_INDEX_REDUCE_OP_ALL_INDEX_TYPES(char);
REGISTER_INDEX_REDUCE_OP_ALL_INDEX_TYPES(uchar);

// index_add is index_reduce with a sum reduction plus an alpha scale on the
// source. Accumulation happens directly in the output dtype via atomic add,
// which (unlike MPSGraph scatter) natively covers long and complex types.
template <typename T, typename IT>
kernel void index_add(
    device AtomicType_t<T>* self [[buffer(0)]],
    device IT* index [[buffer(1)]],
    device T* source [[buffer(2)]],
    constant IndexReduceParams<>& params [[buffer(3)]],
    constant T& alpha [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  uint32_t tid_ = tid;
  long source_offset = 0;
  long self_offset = 0;

  for (int32_t dim = params.ndim - 1; dim >= 0; dim--) {
    auto source_size = params.source_sizes[dim];
    auto dim_idx = tid_ % source_size;

    source_offset += dim_idx * params.source_strides[dim];

    if (dim == params.reduce_dim) {
      // Clamp keeps the access in bounds; out-of-range indices are reported
      // separately by the index_check_bounds pass, which invalidates the
      // result.
      long idx = clamp(
          long(index[dim_idx * params.index_stride]),
          0L,
          long(params.self_sizes[dim]) - 1);
      self_offset += static_cast<uint32_t>(idx) * params.self_strides[dim];
    } else {
      self_offset += dim_idx * params.self_strides[dim];
    }

    tid_ /= source_size;
  }

  AtomicType<T>::atomic_add(
      self, self_offset, mul(alpha, source[source_offset]));
}

#define REGISTER_INDEX_ADD_OP(T, IT)                          \
  template [[host_name("index_add_" #T "_" #IT)]] kernel void \
  index_add<T, IT>(                                           \
      device AtomicType_t<T> * self [[buffer(0)]],            \
      device IT * index [[buffer(1)]],                        \
      device T * source [[buffer(2)]],                        \
      constant IndexReduceParams<> & params [[buffer(3)]],    \
      constant T & alpha [[buffer(4)]],                       \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_INDEX_ADD_OP_ALL_INDEX_TYPES(T) \
  REGISTER_INDEX_ADD_OP(T, int);                 \
  REGISTER_INDEX_ADD_OP(T, long);

REGISTER_INDEX_ADD_OP_ALL_INDEX_TYPES(float);
REGISTER_INDEX_ADD_OP_ALL_INDEX_TYPES(half);
REGISTER_INDEX_ADD_OP_ALL_INDEX_TYPES(bfloat);
REGISTER_INDEX_ADD_OP_ALL_INDEX_TYPES(long);
REGISTER_INDEX_ADD_OP_ALL_INDEX_TYPES(int);
REGISTER_INDEX_ADD_OP_ALL_INDEX_TYPES(short);
REGISTER_INDEX_ADD_OP_ALL_INDEX_TYPES(char);
REGISTER_INDEX_ADD_OP_ALL_INDEX_TYPES(uchar);
REGISTER_INDEX_ADD_OP_ALL_INDEX_TYPES(bool);
REGISTER_INDEX_ADD_OP_ALL_INDEX_TYPES(float2);
REGISTER_INDEX_ADD_OP_ALL_INDEX_TYPES(half2);

// Dim-based index_select gather: output[..., j, ...] = input[..., index[j],
// ...] along reduce_dim. One thread per output element; templated by element
// bit-size (T) so a single set of kernels covers every dtype, complex included.
// Validate index values once (one thread per index) before a gather/scatter
// kernel runs, so the hot kernel can clamp instead of branch-and-report on
// every element. Reports the first out-of-bounds index via the stream error
// buffer.
template <typename IT>
kernel void index_check_bounds(
    device IT* index [[buffer(0)]],
    constant uint& index_stride [[buffer(1)]],
    constant long& dim_size [[buffer(2)]],
    device ErrorMessages* error_buf [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  long idx = index[tid * index_stride];
  if (idx < 0 || idx >= dim_size) {
    TORCH_REPORT_ERROR(
        error_buf,
        "index ",
        idx,
        " is out of bounds for dimension with size ",
        dim_size);
  }
}

template [[host_name("index_check_bounds_int")]] kernel void index_check_bounds<
    int>(
    device int*,
    constant uint&,
    constant long&,
    device ErrorMessages*,
    uint);
template [[host_name("index_check_bounds_long")]] kernel void index_check_bounds<
    long>(
    device long*,
    constant uint&,
    constant long&,
    device ErrorMessages*,
    uint);

template <typename T, typename IT>
kernel void index_select_dim(
    device T* output [[buffer(0)]],
    device IT* index [[buffer(1)]],
    device T* input [[buffer(2)]],
    constant IndexReduceParams<>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  uint32_t tid_ = tid;
  long output_offset = 0;
  long input_offset = 0;

  // params.source_* describe the output (iterated), params.self_* the input.
  for (int32_t dim = params.ndim - 1; dim >= 0; dim--) {
    auto output_size = params.source_sizes[dim];
    auto dim_idx = tid_ % output_size;

    output_offset += dim_idx * params.source_strides[dim];

    if (dim == params.reduce_dim) {
      // Clamp keeps the access in bounds; index_check_bounds reports any
      // out-of-range index and invalidates the result.
      long idx = clamp(
          long(index[dim_idx * params.index_stride]),
          0L,
          long(params.self_sizes[dim]) - 1);
      input_offset += static_cast<uint32_t>(idx) * params.self_strides[dim];
    } else {
      input_offset += dim_idx * params.self_strides[dim];
    }

    tid_ /= output_size;
  }

  output[output_offset] = input[input_offset];
}

#define REGISTER_INDEX_SELECT_DIM_OP(SUFFIX, T, IT)                       \
  template [[host_name("index_select_dim_" #SUFFIX "_" #IT)]] kernel void \
  index_select_dim<T, IT>(                                                \
      device T * output [[buffer(0)]],                                    \
      device IT * index [[buffer(1)]],                                    \
      device T * input [[buffer(2)]],                                     \
      constant IndexReduceParams<> & params [[buffer(3)]],                \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_INDEX_SELECT_DIM_OP_ALL_SIZES(IT) \
  REGISTER_INDEX_SELECT_DIM_OP(8bit, char, IT);    \
  REGISTER_INDEX_SELECT_DIM_OP(16bit, short, IT);  \
  REGISTER_INDEX_SELECT_DIM_OP(32bit, int, IT);    \
  REGISTER_INDEX_SELECT_DIM_OP(64bit, long, IT);

REGISTER_INDEX_SELECT_DIM_OP_ALL_SIZES(int);
REGISTER_INDEX_SELECT_DIM_OP_ALL_SIZES(long);

// Fast path for contiguous index_select: tensors are [outer, dim, inner]; a 3D
// grid (inner, out_dim, outer) drops the per-element coordinate decomposition.
template <typename T, typename IT>
kernel void index_select_dim_dense(
    device T* output [[buffer(0)]],
    device IT* index [[buffer(1)]],
    device T* input [[buffer(2)]],
    constant IndexSelectParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]) {
  // Clamp keeps the read in bounds; index_check_bounds reports any out-of-range
  // index and invalidates the result.
  long in_row = clamp(
      long(index[tid.y * params.index_stride]),
      0L,
      long(params.in_dim_size) - 1);
  long out_off =
      (static_cast<long>(tid.z) * params.out_dim_size + tid.y) * params.inner +
      tid.x;
  long in_off =
      (static_cast<long>(tid.z) * params.in_dim_size + in_row) * params.inner +
      tid.x;
  output[out_off] = input[in_off];
}

#define REGISTER_INDEX_SELECT_DIM_DENSE_OP(SUFFIX, T, IT)                  \
  template                                                                 \
      [[host_name("index_select_dim_dense_" #SUFFIX "_" #IT)]] kernel void \
      index_select_dim_dense<T, IT>(                                       \
          device T * output [[buffer(0)]],                                 \
          device IT * index [[buffer(1)]],                                 \
          device T * input [[buffer(2)]],                                  \
          constant IndexSelectParams & params [[buffer(3)]],               \
          uint3 tid [[thread_position_in_grid]]);

#define REGISTER_INDEX_SELECT_DIM_DENSE_OP_ALL_SIZES(IT) \
  REGISTER_INDEX_SELECT_DIM_DENSE_OP(8bit, char, IT);    \
  REGISTER_INDEX_SELECT_DIM_DENSE_OP(16bit, short, IT);  \
  REGISTER_INDEX_SELECT_DIM_DENSE_OP(32bit, int, IT);    \
  REGISTER_INDEX_SELECT_DIM_DENSE_OP(64bit, long, IT);

REGISTER_INDEX_SELECT_DIM_DENSE_OP_ALL_SIZES(int);
REGISTER_INDEX_SELECT_DIM_DENSE_OP_ALL_SIZES(long);

template <typename StridesT, typename DataT>
kernel void kernel_index_offsets(
    constant StridesT* strides [[buffer(0)]],
    device DataT* data_offsets [[buffer(1)]],
    constant uint* iter_shape [[buffer(2)]],
    constant uint& num_dimensions [[buffer(3)]],
    uint thread_index [[thread_position_in_grid]]) {
  data_offsets[thread_index] = 0;
  uint32_t idx = thread_index;
  for (uint32_t dim = 0; dim < num_dimensions; dim++) {
    uint32_t remainder = idx % iter_shape[dim];
    idx /= iter_shape[dim];

    data_offsets[thread_index] += remainder * DataT(strides[dim]);
  }
}

template [[host_name("kernel_index_offsets_32")]] kernel void
kernel_index_offsets<packed_uint3, uint3>(
    constant packed_uint3* strides [[buffer(0)]],
    device uint3* data_offsets [[buffer(1)]],
    constant uint* iter_shape [[buffer(2)]],
    constant uint& num_dimensions [[buffer(3)]],
    uint thread_index [[thread_position_in_grid]]);

template [[host_name("kernel_index_offsets_64")]] kernel void
kernel_index_offsets<packed_uint3, ulong3>(
    constant packed_uint3* strides [[buffer(0)]],
    device ulong3* data_offsets [[buffer(1)]],
    constant uint* iter_shape [[buffer(2)]],
    constant uint& num_dimensions [[buffer(3)]],
    uint thread_index [[thread_position_in_grid]]);

template <typename T>
kernel void masked_fill_scalar_dense(
    device T* input,
    constant bool* mask,
    constant T& val,
    uint thread_index [[thread_position_in_grid]]) {
  if (mask[thread_index]) {
    input[thread_index] = val;
  }
}

template <typename T>
kernel void masked_fill_scalar_broadcast(
    device T* input,
    constant bool* mask,
    constant T& val,
    constant uint& mask_numel,
    uint thread_index [[thread_position_in_grid]]) {
  if (mask[thread_index % mask_numel]) {
    input[thread_index] = val;
  }
}

template <typename T>
kernel void masked_fill_scalar_strided(
    device T* input,
    constant bool* mask,
    constant T& val,
    constant long* sizes,
    constant long* input_strides,
    constant long* mask_strides,
    device uint& ndim,
    uint thread_index [[thread_position_in_grid]]) {
  int pos[max_ndim];
  pos_from_thread_index(int(thread_index), pos, sizes, ndim);
  if (mask[offset_from_coord(pos, mask_strides, ndim)]) {
    input[offset_from_coord(pos, input_strides, ndim)] = val;
  }
}

template <typename T, typename index_t, typename OffsetT>
kernel void index_copy_dense(
    device T* output,
    constant T* source,
    constant index_t* indices,
    constant long& dim_size,
    constant long& inner,
    constant long& indices_numel,
    uint3 gid [[thread_position_in_grid]]) {
  OffsetT after = gid.x;
  OffsetT i = gid.y;
  OffsetT before = gid.z;
  OffsetT idx = indices[i];
  output[(before * OffsetT(dim_size) + idx) * OffsetT(inner) + after] =
      source[(before * OffsetT(indices_numel) + i) * OffsetT(inner) + after];
}

template <typename T, typename index_t, typename OffsetT>
kernel void index_copy_strided(
    device T* output,
    constant T* source,
    constant index_t* indices,
    constant long& dim_size,
    constant long& dim_out_stride, // result.stride(dim)
    constant long& dim_source_stride, // source.stride(dim)
    constant long* slice_sizes, // result sizes with dim removed
    constant long* slice_out_strides, // result strides with dim removed
    constant long* slice_source_strides, // source strides with dim removed
    constant uint& slice_ndim, // ndim - 1
    constant long& slice_numel,
    constant long& indices_stride, // index.stride(0)
    uint thread_index [[thread_position_in_grid]]) {
  OffsetT j = OffsetT(thread_index) % OffsetT(slice_numel);
  OffsetT i = OffsetT(thread_index) / OffsetT(slice_numel);
  OffsetT idx = indices[i * OffsetT(indices_stride)];
  if (idx < 0) {
    idx += OffsetT(dim_size);
  }
  OffsetT slice_pos[max_ndim];
  pos_from_thread_index(j, slice_pos, slice_sizes, slice_ndim);
  OffsetT out_offset =
      offset_from_coord(slice_pos, slice_out_strides, slice_ndim);
  OffsetT src_offset =
      offset_from_coord(slice_pos, slice_source_strides, slice_ndim);
  output[out_offset + idx * OffsetT(dim_out_stride)] =
      source[src_offset + i * OffsetT(dim_source_stride)];
}

// Scatter-based index_fill: each thread writes exactly one element.
// Launches indices_numel * slice_numel threads, where
// slice_numel = numel / self.size(dim).

// Dense (contiguous output): offset computed analytically.
template <typename T>
kernel void index_fill_dense(
    device T* output,
    constant long* indices,
    constant T& fill_val,
    constant long& dim_size,
    constant long&
        dim_stride, // output.stride(dim); trailing product for contiguous
    constant long& slice_numel,
    uint thread_index [[thread_position_in_grid]]) {
  long j =
      thread_index % slice_numel; // position within the slice (non-dim dims)
  long i = thread_index / slice_numel; // which index
  long idx = indices[i];
  if (idx < 0) {
    idx += dim_size;
  }
  long before = j / dim_stride;
  long after = j % dim_stride;
  output[before * (dim_size * dim_stride) + idx * dim_stride + after] =
      fill_val;
}

// Strided (non-contiguous output or index): offset computed via slice strides.
template <typename T>
kernel void index_fill_strided(
    device T* output,
    constant long* indices,
    constant T& fill_val,
    constant long& dim_size,
    constant long& dim_out_stride, // output.stride(dim)
    constant long*
        slice_sizes, // output sizes with dim removed (length = ndim-1)
    constant long*
        slice_out_strides, // output strides with dim removed (length = ndim-1)
    constant uint& slice_ndim, // ndim - 1
    constant long& slice_numel,
    constant long& indices_stride, // index.stride(0)
    uint thread_index [[thread_position_in_grid]]) {
  long j = thread_index % slice_numel;
  long i = thread_index / slice_numel;
  long idx = indices[i * indices_stride];
  if (idx < 0) {
    idx += dim_size;
  }
  int slice_pos[max_ndim];
  pos_from_thread_index(int(j), slice_pos, slice_sizes, slice_ndim);
  long out_offset = offset_from_coord(slice_pos, slice_out_strides, slice_ndim);
  output[out_offset + idx * dim_out_stride] = fill_val;
}

// Two-pass mask-based index_fill for large index counts.
// Avoids write conflicts from duplicate indices and improves cache locality.
//
// Pass 0 (zero_mask): clear the mask buffer in the same compute encoder to
//   avoid a blit/compute encoder switch from using at::zeros().
// Pass 1 (set_mask): mark which dim-positions to fill.
// Pass 2 (from_mask): fill all elements whose dim-coordinate is marked.

kernel void index_fill_zero_mask(
    device bool* mask,
    uint i [[thread_position_in_grid]]) {
  mask[i] = false;
}

kernel void index_fill_set_mask(
    device bool* mask,
    constant long* indices,
    constant long& dim_size,
    constant long& indices_stride,
    uint thread_index [[thread_position_in_grid]]) {
  long idx = indices[thread_index * indices_stride];
  if (idx < 0)
    idx += dim_size;
  mask[idx] = true;
}

// Dense: fills output elements whose dim-coordinate is marked in mask.
// Uses a 3D thread grid (inner, dim, outer) to avoid any integer division:
//   x = inner index  (0..inner_size-1 where inner_size = stride(dim))
//   y = dim coord    (0..dim_size-1)           <-- no division needed
//   z = outer index  (0..outer_size-1)
// Output offset = (outer * dim_size + dim) * inner_size + inner.
template <typename T>
kernel void index_fill_dense_from_mask(
    device T* output,
    constant bool* mask,
    constant T& fill_val,
    constant uint& dim_size,
    constant uint& inner_size,
    uint3 thread_pos [[thread_position_in_grid]]) {
  uint inner_idx = thread_pos.x;
  uint dim_idx = thread_pos.y;
  uint outer_idx = thread_pos.z;
  if (mask[dim_idx]) {
    output[(outer_idx * dim_size + dim_idx) * inner_size + inner_idx] =
        fill_val;
  }
}

// Strided: thread i fills its element if its dim-coordinate is marked.
template <typename T>
kernel void index_fill_strided_from_mask(
    device T* output,
    constant bool* mask,
    constant T& fill_val,
    constant long* sizes,
    constant long* out_strides,
    constant uint& dim,
    constant uint& ndim,
    uint thread_index [[thread_position_in_grid]]) {
  int pos[max_ndim];
  pos_from_thread_index(int(thread_index), pos, sizes, ndim);
  if (mask[pos[dim]]) {
    long out_offset = offset_from_coord(pos, out_strides, ndim);
    output[out_offset] = fill_val;
  }
}

#define INSTANTIATE_INDEX_FILL(T)                  \
  template [[host_name("index_fill_dense_" #T)]]   \
  kernel void index_fill_dense<T>(                 \
      device T*,                                   \
      constant long*,                              \
      constant T&,                                 \
      constant long&,                              \
      constant long&,                              \
      constant long&,                              \
      uint);                                       \
  template [[host_name("index_fill_strided_" #T)]] \
  kernel void index_fill_strided<T>(               \
      device T*,                                   \
      constant long*,                              \
      constant T&,                                 \
      constant long&,                              \
      constant long&,                              \
      constant long*,                              \
      constant long*,                              \
      constant uint&,                              \
      constant long&,                              \
      constant long&,                              \
      uint);

#define INSTANTIATE_INDEX_COPY_W(T, index_t, OffsetT, OBITS)               \
  template [[host_name("index_copy_dense_" #T "_" #index_t "_" #OBITS)]]   \
  kernel void index_copy_dense<T, index_t, OffsetT>(                       \
      device T*,                                                           \
      constant T*,                                                         \
      constant index_t*,                                                   \
      constant long&,                                                      \
      constant long&,                                                      \
      constant long&,                                                      \
      uint3);                                                              \
                                                                           \
  template [[host_name("index_copy_strided_" #T "_" #index_t "_" #OBITS)]] \
  kernel void index_copy_strided<T, index_t, OffsetT>(                     \
      device T*,                                                           \
      constant T*,                                                         \
      constant index_t*,                                                   \
      constant long&,                                                      \
      constant long&,                                                      \
      constant long&,                                                      \
      constant long*,                                                      \
      constant long*,                                                      \
      constant long*,                                                      \
      constant uint&,                                                      \
      constant long&,                                                      \
      constant long&,                                                      \
      uint);

#define INSTANTIATE_INDEX_COPY(T, index_t)      \
  INSTANTIATE_INDEX_COPY_W(T, index_t, int, 32) \
  INSTANTIATE_INDEX_COPY_W(T, index_t, long, 64)

#define REGISTER_MASKED_FILL_SCALAR(SIZE, DTYPE)                            \
  template [[host_name("masked_fill_scalar_strided_" #SIZE)]] kernel void   \
  masked_fill_scalar_strided<DTYPE>(                                        \
      device DTYPE*,                                                        \
      constant bool*,                                                       \
      constant DTYPE&,                                                      \
      constant long*,                                                       \
      constant long*,                                                       \
      constant long*,                                                       \
      device uint&,                                                         \
      uint);                                                                \
  template [[host_name("masked_fill_scalar_dense_" #SIZE)]] kernel void     \
  masked_fill_scalar_dense<DTYPE>(                                          \
      device DTYPE*, constant bool*, constant DTYPE&, uint);                \
  template [[host_name("masked_fill_scalar_broadcast_" #SIZE)]] kernel void \
  masked_fill_scalar_broadcast<DTYPE>(                                      \
      device DTYPE*, constant bool*, constant DTYPE&, constant uint&, uint)

REGISTER_MASKED_FILL_SCALAR(64bit, long);
REGISTER_MASKED_FILL_SCALAR(32bit, int);
REGISTER_MASKED_FILL_SCALAR(16bit, short);
REGISTER_MASKED_FILL_SCALAR(8bit, char);
INSTANTIATE_INDEX_COPY(float, int);
INSTANTIATE_INDEX_COPY(float, long);
INSTANTIATE_INDEX_COPY(bool, int);
INSTANTIATE_INDEX_COPY(bool, long);
INSTANTIATE_INDEX_COPY(half, int);
INSTANTIATE_INDEX_COPY(half, long);
INSTANTIATE_INDEX_COPY(int, int);
INSTANTIATE_INDEX_COPY(int, long);
INSTANTIATE_INDEX_COPY(long, int);
INSTANTIATE_INDEX_COPY(long, long);
INSTANTIATE_INDEX_COPY(short, int);
INSTANTIATE_INDEX_COPY(short, long);
INSTANTIATE_INDEX_COPY(char, int);
INSTANTIATE_INDEX_COPY(char, long);
INSTANTIATE_INDEX_COPY(uchar, int);
INSTANTIATE_INDEX_COPY(uchar, long);

INSTANTIATE_INDEX_COPY(bfloat, int);
INSTANTIATE_INDEX_COPY(bfloat, long);
INSTANTIATE_INDEX_COPY(float2, int);
INSTANTIATE_INDEX_COPY(float2, long);
INSTANTIATE_INDEX_COPY(half2, int);
INSTANTIATE_INDEX_COPY(half2, long);

INSTANTIATE_INDEX_FILL(float);
INSTANTIATE_INDEX_FILL(bool);
INSTANTIATE_INDEX_FILL(half);
INSTANTIATE_INDEX_FILL(bfloat);
INSTANTIATE_INDEX_FILL(int);
INSTANTIATE_INDEX_FILL(long);
INSTANTIATE_INDEX_FILL(short);
INSTANTIATE_INDEX_FILL(char);
INSTANTIATE_INDEX_FILL(uchar);
INSTANTIATE_INDEX_FILL(float2);
INSTANTIATE_INDEX_FILL(half2);

#define INSTANTIATE_INDEX_FILL_FROM_MASK(T)                  \
  template [[host_name("index_fill_dense_from_mask_" #T)]]   \
  kernel void index_fill_dense_from_mask<T>(                 \
      device T*,                                             \
      constant bool*,                                        \
      constant T&,                                           \
      constant uint&,                                        \
      constant uint&,                                        \
      uint3);                                                \
  template [[host_name("index_fill_strided_from_mask_" #T)]] \
  kernel void index_fill_strided_from_mask<T>(               \
      device T*,                                             \
      constant bool*,                                        \
      constant T&,                                           \
      constant long*,                                        \
      constant long*,                                        \
      constant uint&,                                        \
      constant uint&,                                        \
      uint);

INSTANTIATE_INDEX_FILL_FROM_MASK(float)
INSTANTIATE_INDEX_FILL_FROM_MASK(half)
INSTANTIATE_INDEX_FILL_FROM_MASK(bfloat)
INSTANTIATE_INDEX_FILL_FROM_MASK(int)
INSTANTIATE_INDEX_FILL_FROM_MASK(long)
INSTANTIATE_INDEX_FILL_FROM_MASK(short)
INSTANTIATE_INDEX_FILL_FROM_MASK(char)
INSTANTIATE_INDEX_FILL_FROM_MASK(uchar)
INSTANTIATE_INDEX_FILL_FROM_MASK(bool)
INSTANTIATE_INDEX_FILL_FROM_MASK(float2)
INSTANTIATE_INDEX_FILL_FROM_MASK(half2)

// Nonzero kernel implementation using prefix-sum + scatter approach.
//
// Step 1 (count_nonzero_prefix_sum): Each threadgroup computes an exclusive
// prefix sum of the nonzero flags over its chunk. Per-threadgroup totals are
// written to block_sums.
//
// Step 2 (prefix_sum_blocks_uint): A single threadgroup computes the exclusive
// prefix sum of block_sums → block_offsets and writes the total nonzero count
// to a 1-element buffer. The host reads back only that single int, then
// allocates the output tensor.
//
// Step 3 (scatter_nonzero_indices): Each thread with a nonzero element writes
// its multi-dimensional indices into the output at the position determined by
// block_offsets[tgid] + prefix[tid].

template <typename T, enable_if_t<!is_complex_v<T>, bool> = true>
inline bool is_nonzero(T val) {
  return val != T(0);
}

template <typename T, enable_if_t<is_complex_v<T>, bool> = true>
inline bool is_nonzero(T val) {
  return val.x != 0 || val.y != 0;
}

// Count-side width: per-threadgroup block sums and the intra-block prefix
// values are bounded by the threadgroup size (<= 1024), so they stay uint32.
// The cumulative quantities (block_offsets, the total nonzero count, and the
// scatter output position) can only exceed 2^32 when the tensor itself has
// more than 2^32 elements, so there are two block-scan kernels: the default
// prefix_sum_blocks_uint keeps the fast parallel simd_shuffle scan (uint32
// accum, used whenever numel <= 2^32, i.e. the common case), and
// prefix_sum_blocks_ulong does the scan in 64-bit for tensors above 2^32
// elements. Both write int64 block_offsets and total, so scatter stays uniform.
// (Metal 3.x cannot simd_shuffle 64-bit integers, so the i64 path scans in
// threadgroup memory.) The input flat index is a separate concern, widened via
// the index_t template parameter (see scatter_nonzero_indices).
template <typename T, typename index_t>
[[max_total_threads_per_threadgroup(1024)]]
kernel void count_nonzero_prefix_sum(
    const device T* input [[buffer(0)]],
    device uint* prefix [[buffer(1)]],
    device uint* block_sums [[buffer(2)]],
    constant ulong& flat_base [[buffer(3)]],
    constant uint& block_base [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  uint num_simds = (tgsize + simdgroup_size - 1) / simdgroup_size;

  // Chunked dispatch: tid/tgid are chunk-local; flat_base/block_base map them
  // to global element and block indices so tensors with more than 2^32 elements
  // (which exceed Metal's 32-bit thread_position_in_grid) are handled across
  // multiple dispatches over a single global prefix-sum.
  index_t gid = static_cast<index_t>(flat_base) + tid;

  uint flag = is_nonzero(input[gid]) ? 1u : 0u;

  // Inclusive prefix sum within SIMD group using shuffle
  uint val = flag;
  for (uint offset = 1; offset < simdgroup_size; offset <<= 1) {
    uint other = simd_shuffle_and_fill_up(val, 0u, static_cast<ushort>(offset));
    val += other;
  }

  // The last lane in each simd group writes its total.
  // For full groups this is lane 31; for the last (partial) group we compute
  // which lane is actually last.
  threadgroup uint simdgroup_totals[32];
  bool is_last_lane_in_simd;
  if (simd_group_id < num_simds - 1) {
    is_last_lane_in_simd = (simd_lane_id == simdgroup_size - 1);
  } else {
    uint lanes_in_last = tgsize - simd_group_id * simdgroup_size;
    is_last_lane_in_simd = (simd_lane_id == lanes_in_last - 1);
  }
  if (is_last_lane_in_simd) {
    simdgroup_totals[simd_group_id] = val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // First simd group computes exclusive prefix sum of simd group totals
  threadgroup uint simdgroup_offsets[32];
  if (simd_group_id == 0) {
    uint sg_val =
        (simd_lane_id < num_simds) ? simdgroup_totals[simd_lane_id] : 0u;
    for (uint offset = 1; offset < simdgroup_size; offset <<= 1) {
      uint other =
          simd_shuffle_and_fill_up(sg_val, 0u, static_cast<ushort>(offset));
      sg_val += other;
    }
    uint exclusive =
        simd_shuffle_and_fill_up(sg_val, 0u, static_cast<ushort>(1));
    simdgroup_offsets[simd_lane_id] = exclusive;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint exclusive_val = val - flag + simdgroup_offsets[simd_group_id];

  prefix[gid] = exclusive_val;

  if (lid == tgsize - 1) {
    block_sums[block_base + tgid] =
        simdgroup_offsets[num_simds - 1] + simdgroup_totals[num_simds - 1];
  }
}

// Step 2: exclusive prefix sum of block_sums -> block_offsets, plus the grand
// total. Runs in a single threadgroup. Fast path for numel <= 2^32: the
// per-block sums and their running total both fit in uint32, so this keeps the
// parallel simd_shuffle scan. block_offsets/total are still int64-typed so the
// scatter kernel is identical across paths.
[[max_total_threads_per_threadgroup(1024)]]
kernel void prefix_sum_blocks_uint(
    const device uint* block_sums [[buffer(0)]],
    device long* block_offsets [[buffer(1)]],
    device long* total_nonzero [[buffer(2)]],
    constant uint& num_blocks [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  uint num_simds = (tgsize + simdgroup_size - 1) / simdgroup_size;

  // Each thread handles a contiguous chunk of blocks
  uint chunk_size = (num_blocks + tgsize - 1) / tgsize;
  uint start = lid * chunk_size;
  uint end = min(start + chunk_size, num_blocks);

  // Serial sum over this thread's chunk
  uint chunk_total = 0u;
  for (uint i = start; i < end; i++) {
    chunk_total += block_sums[i];
  }

  // Parallel inclusive prefix sum of chunk_totals across threads
  uint val = chunk_total;
  for (uint offset = 1; offset < simdgroup_size; offset <<= 1) {
    uint other = simd_shuffle_and_fill_up(val, 0u, static_cast<ushort>(offset));
    val += other;
  }

  threadgroup uint simdgroup_totals[32];
  bool is_last_lane_in_simd;
  if (simd_group_id < num_simds - 1) {
    is_last_lane_in_simd = (simd_lane_id == simdgroup_size - 1);
  } else {
    uint lanes_in_last = tgsize - simd_group_id * simdgroup_size;
    is_last_lane_in_simd = (simd_lane_id == lanes_in_last - 1);
  }
  if (is_last_lane_in_simd) {
    simdgroup_totals[simd_group_id] = val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup uint simdgroup_offsets[32];
  if (simd_group_id == 0) {
    uint sg_val =
        (simd_lane_id < num_simds) ? simdgroup_totals[simd_lane_id] : 0u;
    for (uint offset = 1; offset < simdgroup_size; offset <<= 1) {
      uint other =
          simd_shuffle_and_fill_up(sg_val, 0u, static_cast<ushort>(offset));
      sg_val += other;
    }
    uint exclusive =
        simd_shuffle_and_fill_up(sg_val, 0u, static_cast<ushort>(1));
    simdgroup_offsets[simd_lane_id] = exclusive;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // This thread's exclusive offset = inclusive_scan - chunk_total +
  // simdgroup_offset
  uint thread_offset = val - chunk_total + simdgroup_offsets[simd_group_id];

  // Write block_offsets for this thread's chunk using a serial exclusive scan
  uint running = thread_offset;
  for (uint i = start; i < end; i++) {
    block_offsets[i] = static_cast<long>(running);
    running += block_sums[i];
  }

  if (lid == tgsize - 1) {
    *total_nonzero = static_cast<long>(
        simdgroup_offsets[num_simds - 1] + simdgroup_totals[num_simds - 1]);
  }
}

// 64-bit variant for tensors with more than 2^32 elements, where the running
// count can exceed uint32. Metal 3.x cannot simd_shuffle 64-bit integers, so
// the cross-thread scan uses a threadgroup-memory Hillis-Steele scan in 64-bit
// instead of the uint path's simd_shuffle scan. This path only runs for
// >2^32-element inputs (>34GB output regime); it never affects normal tensors,
// which take prefix_sum_blocks_uint.
[[max_total_threads_per_threadgroup(1024)]]
kernel void prefix_sum_blocks_ulong(
    const device uint* block_sums [[buffer(0)]],
    device long* block_offsets [[buffer(1)]],
    device long* total_nonzero [[buffer(2)]],
    constant uint& num_blocks [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]) {
  // Each thread handles a contiguous chunk of blocks.
  uint chunk_size = (num_blocks + tgsize - 1) / tgsize;
  uint start = lid * chunk_size;
  uint end = min(start + chunk_size, num_blocks);

  // Serial sum over this thread's chunk. Each block_sum is <= tgsize (<=1024)
  // and a chunk spans at most ceil(num_blocks / tgsize) blocks, so this fits
  // in uint32 for any input that fits in device memory.
  uint chunk_total = 0u;
  for (uint i = start; i < end; i++) {
    chunk_total += block_sums[i];
  }

  // Parallel inclusive prefix sum of the per-thread chunk totals, accumulated
  // in 64-bit (the running count can exceed 2^32 for a large dense input). Uses
  // a Hillis-Steele scan in threadgroup memory: read the neighbor's partial
  // sum, barrier, then add, so no simd_shuffle (unavailable for 64-bit) is
  // needed. offset < tgsize covers all tgsize (<=1024) lanes.
  threadgroup ulong tg_scan[1024];
  tg_scan[lid] = static_cast<ulong>(chunk_total);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint offset = 1; offset < tgsize; offset <<= 1) {
    ulong add = (lid >= offset) ? tg_scan[lid - offset] : 0ul;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tg_scan[lid] += add;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lid == tgsize - 1) {
    *total_nonzero = static_cast<long>(tg_scan[lid]);
  }

  // Each thread writes the exclusive block offsets for its own chunk in 64-bit.
  // tg_scan[lid] is the inclusive prefix, so the exclusive base is it minus
  // this thread's own chunk_total.
  ulong running = tg_scan[lid] - static_cast<ulong>(chunk_total);
  for (uint i = start; i < end; i++) {
    block_offsets[i] = static_cast<long>(running);
    running += block_sums[i];
  }
}

// Scatter the multi-dimensional indices of nonzero elements.
// Output layout: out[position * ndim + d] = index along dimension d.
// The output position and its address arithmetic are 64-bit: the nonzero count
// (hence pos) can exceed UINT32_MAX for a large dense input, and pos * ndim can
// exceed it even when pos and ndim individually fit in uint32.
template <typename T, typename index_t>
[[max_total_threads_per_threadgroup(1024)]]
kernel void scatter_nonzero_indices(
    const device T* input [[buffer(0)]],
    const device uint* prefix [[buffer(1)]],
    device int64_t* output [[buffer(2)]],
    constant int& ndim [[buffer(3)]],
    constant int64_t* sizes [[buffer(4)]],
    constant long* block_offsets [[buffer(5)]],
    constant long& max_entries [[buffer(6)]],
    constant ulong& flat_base [[buffer(7)]],
    constant uint& block_base [[buffer(8)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]]) {
  // Chunked dispatch: map chunk-local tid/tgid to the global element/block.
  ulong gid = flat_base + tid;
  if (!is_nonzero(input[gid]))
    return;

  ulong pos =
      static_cast<ulong>(block_offsets[block_base + tgid]) + prefix[gid];
  if (pos >= static_cast<ulong>(max_entries))
    return;

  // index_t is uint for tensors that fit 32-bit index math and ulong otherwise;
  // it keeps the per-dimension div/mod below in fast 32-bit math on the input
  // flat index. The output base is always 64-bit (count-side, see above).
  index_t flat = gid;
  ulong out_base = pos * static_cast<ulong>(ndim);
  for (int d = ndim - 1; d >= 0; d--) {
    index_t dim_size = static_cast<index_t>(sizes[d]);
    output[out_base + d] = static_cast<int64_t>(flat % dim_size);
    flat /= dim_size;
  }
}

// IDX (uint/ulong) is the flat-index width; it also names the registered
// kernel. The buffers themselves are IDX-independent (prefix/block_sums stay
// uint32, block_offsets/output stay int64), so only the template arg varies.
#define REGISTER_NONZERO_KERNELS_FOR_IDX(DTYPE, IDX)          \
  template [[host_name("count_nonzero_prefix_sum_" #DTYPE     \
                       "_" #IDX)]] [[kernel]] void            \
  count_nonzero_prefix_sum<DTYPE, IDX>(                       \
      const device DTYPE* input [[buffer(0)]],                \
      device uint* prefix [[buffer(1)]],                      \
      device uint* block_sums [[buffer(2)]],                  \
      constant ulong& flat_base [[buffer(3)]],                \
      constant uint& block_base [[buffer(4)]],                \
      uint tid [[thread_position_in_grid]],                   \
      uint lid [[thread_position_in_threadgroup]],            \
      uint tgsize [[threads_per_threadgroup]],                \
      uint tgid [[threadgroup_position_in_grid]],             \
      uint simd_lane_id [[thread_index_in_simdgroup]],        \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]); \
                                                              \
  template [[host_name("scatter_nonzero_indices_" #DTYPE      \
                       "_" #IDX)]] [[kernel]] void            \
  scatter_nonzero_indices<DTYPE, IDX>(                        \
      const device DTYPE* input [[buffer(0)]],                \
      const device uint* prefix [[buffer(1)]],                \
      device int64_t* output [[buffer(2)]],                   \
      constant int& ndim [[buffer(3)]],                       \
      constant int64_t* sizes [[buffer(4)]],                  \
      constant long* block_offsets [[buffer(5)]],             \
      constant long& max_entries [[buffer(6)]],               \
      constant ulong& flat_base [[buffer(7)]],                \
      constant uint& block_base [[buffer(8)]],                \
      uint tid [[thread_position_in_grid]],                   \
      uint tgid [[threadgroup_position_in_grid]])

#define REGISTER_NONZERO_KERNELS(DTYPE)          \
  REGISTER_NONZERO_KERNELS_FOR_IDX(DTYPE, uint); \
  REGISTER_NONZERO_KERNELS_FOR_IDX(DTYPE, ulong)

REGISTER_NONZERO_KERNELS(float);
REGISTER_NONZERO_KERNELS(half);
REGISTER_NONZERO_KERNELS(bfloat);
REGISTER_NONZERO_KERNELS(long);
REGISTER_NONZERO_KERNELS(int);
REGISTER_NONZERO_KERNELS(short);
REGISTER_NONZERO_KERNELS(char);
REGISTER_NONZERO_KERNELS(uchar);
REGISTER_NONZERO_KERNELS(bool);
REGISTER_NONZERO_KERNELS(float2);
REGISTER_NONZERO_KERNELS(half2);
