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

template <typename T, typename index_t>
kernel void index_copy_dense(
    device T* output,
    constant T* input,
    constant T* source,
    constant index_t* indices,
    constant uint& dim,
    constant long* sizes,
    constant uint& ndim,
    constant uint& indices_numel,
    uint thread_index [[thread_position_in_grid]]) {
  // first copy input to output
  output[thread_index] = input[thread_index];

  // calculate pos in the tensor using a signed counter
  long pos[max_ndim];
  long linear_idx = thread_index;
  for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
    pos[i] = linear_idx % sizes[i];
    linear_idx /= sizes[i];
  }

  // check if this position's dim coordinate is in the indices
  long dim_pos = pos[dim];

  // search through indices to see if current dim pos should be updated
  for (uint i = 0; i < indices_numel; i++) {
    if (indices[i] == dim_pos) {
      // this position should be updated from source
      // calculate source offset where the source tensor has the same shape
      // except along dim where it has size = indices_numel
      long source_offset = 0;
      long stride = 1;
      for (int j = static_cast<int>(ndim) - 1; j >= 0; --j) {
        if (j == static_cast<int>(dim)) {
          // for the indexed dimension, use position i
          source_offset += i * stride;
          stride *= indices_numel;
        } else {
          // for other dimensions use the same position
          source_offset += pos[j] * stride;
          stride *= sizes[j];
        }
      }

      output[thread_index] = source[source_offset];
      break;
    }
  }
}

template <typename T, typename index_t>
kernel void index_copy_strided(
    device T* output,
    constant T* input,
    constant T* source,
    constant index_t* indices,
    constant uint& dim,
    constant long* sizes,
    constant uint& ndim,
    constant uint& indices_numel,
    constant long* input_strides,
    constant long* output_strides,
    constant long* source_strides,
    constant long& indices_stride,
    uint thread_index [[thread_position_in_grid]]) {
  int pos[max_ndim];
  pos_from_thread_index(int(thread_index), pos, sizes, ndim);

  // compute offsets for the output and input tensors
  long output_offset = offset_from_coord(pos, output_strides, ndim);
  long input_offset = offset_from_coord(pos, input_strides, ndim);

  output[output_offset] = input[input_offset];

  // save the original coordinate along the dim we're updating
  int orig_dim = pos[dim];

  // find the last index in the indices array that equals this coordinate
  int last_matching_index = -1;
  for (uint i = 0; i < indices_numel; i++) {
    if (indices[i * indices_stride] == orig_dim) {
      last_matching_index = int(i);
    }
  }

  // if a matching index was found, use it to update the output
  if (last_matching_index != -1) {
    pos[dim] = last_matching_index;
    long source_offset = offset_from_coord(pos, source_strides, ndim);
    output[output_offset] = source[source_offset];
  }
}

#define INSTANTIATE_INDEX_COPY(T, index_t)                      \
  template [[host_name("index_copy_dense_" #T "_" #index_t)]]   \
  kernel void index_copy_dense<T, index_t>(                     \
      device T*,                                                \
      constant T*,                                              \
      constant T*,                                              \
      constant index_t*,                                        \
      constant uint&,                                           \
      constant long*,                                           \
      constant uint&,                                           \
      constant uint&,                                           \
      uint);                                                    \
                                                                \
  template [[host_name("index_copy_strided_" #T "_" #index_t)]] \
  kernel void index_copy_strided<T, index_t>(                   \
      device T*,                                                \
      constant T*,                                              \
      constant T*,                                              \
      constant index_t*,                                        \
      constant uint&,                                           \
      constant long*,                                           \
      constant uint&,                                           \
      constant uint&,                                           \
      constant long*,                                           \
      constant long*,                                           \
      constant long*,                                           \
      constant long&,                                           \
      uint);

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
