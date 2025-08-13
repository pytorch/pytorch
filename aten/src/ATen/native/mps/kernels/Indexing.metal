#include <c10/metal/atomic.h>
#include <c10/metal/indexing.h>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

struct IndexAB {
  constant int64_t* indexArray;
};

template <typename T, typename OffsetsT>
kernel void index_select(
    constant IndexAB* indexAB [[buffer(0)]],
    constant void* indexSizes [[buffer(1)]],
    constant void* indexStrides [[buffer(2)]],
    constant OffsetsT* offsets [[buffer(3)]],
    constant void* inputData [[buffer(4)]],
    device void* outputData [[buffer(5)]],
    constant uint32_t& num_indices [[buffer(6)]],
    uint thread_index [[thread_position_in_grid]]) {
  constant int64_t* index_sizes = (constant int64_t*)indexSizes;
  constant int64_t* index_strides = (constant int64_t*)indexStrides;
  int64_t offset = 0;
  for (uint32_t i = 0; i < num_indices; i++) {
    constant int64_t* indexArray = indexAB[i].indexArray;
    int64_t index = indexArray[offsets[thread_index].z / sizeof(int64_t)];
    if (index < 0) {
      index += index_sizes[i];
    }
    offset += index * index_strides[i];
  }
  device T* out =
      (device T*)((device char*)outputData + offsets[thread_index].x);
  constant T* in = (constant T*)((constant char*)inputData +
                                 offsets[thread_index].y + offset);
  *out = *in;
}

template <typename T, typename OffsetsT>
void index_put_impl(
    constant IndexAB* indexAB,
    constant int64_t* index_sizes,
    constant int64_t* index_strides,
    constant OffsetsT* offsets,
    constant void* inputData,
    device void* outputData,
    constant uint32_t& num_indices,
    uint thread_index) {
  int64_t offset = 0;
  for (uint32_t i = 0; i < num_indices; i++) {
    constant int64_t* indexArray = indexAB[i].indexArray;
    int64_t index = indexArray[offsets[thread_index].z / sizeof(int64_t)];

    if (index < 0) {
      index += index_sizes[i];
    }
    offset += index * index_strides[i];
  }
  device T* out =
      (device T*)((device char*)outputData + offsets[thread_index].x + offset);
  constant T* in =
      (constant T*)((constant char*)inputData + offsets[thread_index].y);
  *out = *in;
}

template <typename T, typename OffsetsT>
kernel void index_put_serial(
    constant IndexAB* indexAB [[buffer(0)]],
    constant void* indexSizes [[buffer(1)]],
    constant void* indexStrides [[buffer(2)]],
    constant OffsetsT* offsets [[buffer(3)]],
    constant void* inputData [[buffer(4)]],
    device void* outputData [[buffer(5)]],
    constant uint32_t& num_indices [[buffer(6)]],
    constant uint* numIters [[buffer(7)]]) {
  constant int64_t* index_sizes = (constant int64_t*)indexSizes;
  constant int64_t* index_strides = (constant int64_t*)indexStrides;

  for (uint iter_i = 0; iter_i < *numIters; iter_i++) {
    index_put_impl<T>(
        indexAB,
        index_sizes,
        index_strides,
        offsets,
        inputData,
        outputData,
        num_indices,
        iter_i);
  }
}

template <typename T, typename OffsetsT>
kernel void index_put(
    constant IndexAB* indexAB [[buffer(0)]],
    constant void* indexSizes [[buffer(1)]],
    constant void* indexStrides [[buffer(2)]],
    constant OffsetsT* offsets [[buffer(3)]],
    constant void* inputData [[buffer(4)]],
    device void* outputData [[buffer(5)]],
    constant uint32_t& num_indices [[buffer(6)]],
    uint thread_index [[thread_position_in_grid]]) {
  constant int64_t* index_sizes = (constant int64_t*)indexSizes;
  constant int64_t* index_strides = (constant int64_t*)indexStrides;
  index_put_impl<T>(
      indexAB,
      index_sizes,
      index_strides,
      offsets,
      inputData,
      outputData,
      num_indices,
      thread_index);
}

#define REGISTER_INDEX_OP(                                     \
    DTYPE_SIZE, IDX_SIZE, DTYPE, INDEX_OP_TYPE, IDX_DTYPE)     \
  template [[host_name("index_" #INDEX_OP_TYPE "_" #DTYPE_SIZE \
                       "_" #IDX_SIZE)]] kernel void            \
      index_##INDEX_OP_TYPE<DTYPE, IDX_DTYPE>(                 \
          constant IndexAB * indexAB [[buffer(0)]],            \
          constant void* indexSizes [[buffer(1)]],             \
          constant void* indexStrides [[buffer(2)]],           \
          constant IDX_DTYPE* offsets [[buffer(3)]],           \
          constant void* inputData [[buffer(4)]],              \
          device void* outputData [[buffer(5)]],               \
          constant uint32_t& num_indices [[buffer(6)]],        \
          uint thread_index [[thread_position_in_grid]])

#define REGISTER_INDEX_OP_ALL_DTYPES(INDEX_OP_TYPE)              \
  REGISTER_INDEX_OP(8bit, idx32, char, INDEX_OP_TYPE, uint3);    \
  REGISTER_INDEX_OP(8bit, idx64, char, INDEX_OP_TYPE, ulong3);   \
  REGISTER_INDEX_OP(16bit, idx32, short, INDEX_OP_TYPE, uint3);  \
  REGISTER_INDEX_OP(16bit, idx64, short, INDEX_OP_TYPE, ulong3); \
  REGISTER_INDEX_OP(32bit, idx32, int, INDEX_OP_TYPE, uint3);    \
  REGISTER_INDEX_OP(32bit, idx64, int, INDEX_OP_TYPE, ulong3);   \
  REGISTER_INDEX_OP(64bit, idx32, long, INDEX_OP_TYPE, uint3);   \
  REGISTER_INDEX_OP(64bit, idx64, long, INDEX_OP_TYPE, ulong3);

REGISTER_INDEX_OP_ALL_DTYPES(select);
REGISTER_INDEX_OP_ALL_DTYPES(put);

#define REGISTER_SINGLE_THREADED_INDEX_OP(                     \
    DTYPE_SIZE, IDX_SIZE, DTYPE, INDEX_OP_TYPE, IDX_DTYPE)     \
  template [[host_name("index_" #INDEX_OP_TYPE "_" #DTYPE_SIZE \
                       "_" #IDX_SIZE)]] kernel void            \
      index_##INDEX_OP_TYPE<DTYPE, IDX_DTYPE>(                 \
          constant IndexAB * indexAB [[buffer(0)]],            \
          constant void* indexSizes [[buffer(1)]],             \
          constant void* indexStrides [[buffer(2)]],           \
          constant IDX_DTYPE* offsets [[buffer(3)]],           \
          constant void* inputData [[buffer(4)]],              \
          device void* outputData [[buffer(5)]],               \
          constant uint32_t& num_indices [[buffer(6)]],        \
          constant uint* numIters [[buffer(7)]])

#define REGISTER_SINGLE_THREADED_INDEX_OP_ALL_DTYPES(INDEX_OP_TYPE)            \
  REGISTER_SINGLE_THREADED_INDEX_OP(8bit, idx32, char, INDEX_OP_TYPE, uint3);  \
  REGISTER_SINGLE_THREADED_INDEX_OP(8bit, idx64, char, INDEX_OP_TYPE, ulong3); \
  REGISTER_SINGLE_THREADED_INDEX_OP(                                           \
      16bit, idx32, short, INDEX_OP_TYPE, uint3);                              \
  REGISTER_SINGLE_THREADED_INDEX_OP(                                           \
      16bit, idx64, short, INDEX_OP_TYPE, ulong3);                             \
  REGISTER_SINGLE_THREADED_INDEX_OP(32bit, idx32, int, INDEX_OP_TYPE, uint3);  \
  REGISTER_SINGLE_THREADED_INDEX_OP(32bit, idx64, int, INDEX_OP_TYPE, ulong3); \
  REGISTER_SINGLE_THREADED_INDEX_OP(64bit, idx32, long, INDEX_OP_TYPE, uint3); \
  REGISTER_SINGLE_THREADED_INDEX_OP(64bit, idx64, long, INDEX_OP_TYPE, ulong3);

REGISTER_SINGLE_THREADED_INDEX_OP_ALL_DTYPES(put_serial);

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

template <typename T, typename OffsetsT>
kernel void index_put_accumulate(
    constant IndexAB* indexAB [[buffer(0)]],
    constant void* indexSizes [[buffer(1)]],
    constant void* indexStrides [[buffer(2)]],
    constant OffsetsT* offsets [[buffer(3)]],
    constant void* inputData [[buffer(4)]],
    device void* outputData [[buffer(5)]],
    constant uint32_t& num_indices [[buffer(6)]],
    uint thread_index [[thread_position_in_grid]]) {
  constant int64_t* index_sizes = (constant int64_t*)indexSizes;
  constant int64_t* index_strides = (constant int64_t*)indexStrides;
  int64_t offset = offsets[thread_index].x;
  for (uint32_t i = 0; i < num_indices; i++) {
    constant int64_t* indexArray = indexAB[i].indexArray;
    int64_t index = indexArray[offsets[thread_index].z / sizeof(int64_t)];
    if (index < 0) {
      index += index_sizes[i];
    }
    offset += index * index_strides[i];
  }
  const auto in =
      *(constant T*)((constant char*)inputData + offsets[thread_index].y);
  AtomicType<T>::atomic_add(
      reinterpret_cast<device AtomicType_t<T>*>(outputData),
      offset / sizeof(T),
      in);
}

#define REGISTER_INDEX_PUT_ACCUMULATE(DTS, DTYPE, IDXS, IDX_DTYPE) \
  template [[host_name("index_put_accumulate_" #DTS "_" #DTYPE     \
                       "_" #IDXS)]] kernel void                    \
  index_put_accumulate<DTYPE, IDX_DTYPE>(                          \
      constant IndexAB * indexAB [[buffer(0)]],                    \
      constant void* indexSizes [[buffer(1)]],                     \
      constant void* indexStrides [[buffer(2)]],                   \
      constant IDX_DTYPE* offsets [[buffer(3)]],                   \
      constant void* inputData [[buffer(4)]],                      \
      device void* outputData [[buffer(5)]],                       \
      constant uint32_t& num_indices [[buffer(6)]],                \
      uint thread_index [[thread_position_in_grid]])

REGISTER_INDEX_PUT_ACCUMULATE(32bit, float, idx32, uint3);
REGISTER_INDEX_PUT_ACCUMULATE(32bit, float, idx64, ulong3);
REGISTER_INDEX_PUT_ACCUMULATE(32bit, int, idx32, uint3);
REGISTER_INDEX_PUT_ACCUMULATE(32bit, int, idx64, ulong3);
REGISTER_INDEX_PUT_ACCUMULATE(16bit, half, idx32, uint3);
REGISTER_INDEX_PUT_ACCUMULATE(16bit, half, idx64, ulong3);

#if __METAL_VERSION__ >= 310
REGISTER_INDEX_PUT_ACCUMULATE(16bit, bfloat, idx32, uint3);
REGISTER_INDEX_PUT_ACCUMULATE(16bit, bfloat, idx64, ulong3);
#endif

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
    if (indices[i] == orig_dim) {
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

#if __METAL_VERSION__ >= 310
INSTANTIATE_INDEX_COPY(bfloat, int);
INSTANTIATE_INDEX_COPY(bfloat, long);
#endif
INSTANTIATE_INDEX_COPY(float2, int);
INSTANTIATE_INDEX_COPY(float2, long);
INSTANTIATE_INDEX_COPY(half2, int);
INSTANTIATE_INDEX_COPY(half2, long);
