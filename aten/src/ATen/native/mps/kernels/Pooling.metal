#include <metal_array>

#include <metal_stdlib>
using namespace metal;

// Kernel computes one element of the output per kernel call. For an input with
// 2 leading dims and 3 pooling dims, the formula in Python syntax is:
//
//   output[N, C, d, h, w] = max(input[
//       N, C,
//       (stride[0] * d):(stride[0] * d + kernel_size[0] * dilation[0]):dilation[0],
//       (stride[1] * h):(stride[1] * h + kernel_size[1] * dilation[1]):dilation[1],
//       (stride[2] * w):(stride[2] * w + kernel_size[2] * dilation[2]):dilation[2],
//   ])
//
// We need to read each of the input elements that the above pseudocode accesses
// and keep track of the biggest one and its index.

// TODO: The above does not take padding into account, and I'll need to find out
// how to do that.

// Iterates through all the input elements that this kernel needs to
// apply max to. Specialized for 3 pooling dimensions.
// TODO: Make it work for any number of pooling dims.
template <typename T>
void max_pool_3d_input_iter(
  constant T* input,
  device T* output,
  device int64_t* indices,
  constant int64_t* input_strides,
  device int64_t* work_pooling_dim_indices,
  constant int64_t* kernel_size,
  constant int64_t* stride,
  constant int64_t* padding,
  constant int64_t* dilation
) {
  int64_t o0 = work_pooling_dim_indices[0];
  int64_t o1 = work_pooling_dim_indices[1];
  int64_t o2 = work_pooling_dim_indices[2];

  int64_t k0 = kernel_size[0];
  int64_t k1 = kernel_size[1];
  int64_t k2 = kernel_size[2];

  int64_t s0 = stride[0];
  int64_t s1 = stride[1];
  int64_t s2 = stride[2];

  int64_t d0 = dilation[0];
  int64_t d1 = dilation[1];
  int64_t d2 = dilation[2];

  // TODO: Pick the first visited element.
  T max_value = 0;
  int64_t max_index = -1;

  for (int64_t i0 = (s0 * o0); i0 < (s0 * o0 + k0 * d0); i0 += d0) {
    int64_t offset0 = input_strides[0] * i0;

    for (int64_t i1 = (s1 * o1); i1 < (s1 * o1 + k1 * d1); i1 += d1) {
      int64_t offset1 = input_strides[1] * i1;

      for (int64_t i2 = (s2 * o2); i2 < (s2 * o2 + k2 * d2); i2 += d2) {
        int64_t offset2 = input_strides[2] * i2;

        const T input_value = input[offset0 + offset1 + offset2];
        max_value = (input_value > max_value) ? input_value : max_value;
        // TODO: Also get the index
      }
    }
  }

  *output = max_value;
}

// NOTE: Probably don't need the nthreads arg
template <typename T>
kernel void max_pool(
  constant void* input_ [[buffer(0)]],
  device void* output_ [[buffer(1)]],
  device void* indices_ [[buffer(2)]],
  constant int64_t& dims [[buffer(3)]],
  constant int64_t& pooling_dims [[buffer(4)]],
  constant int64_t* input_sizes [[buffer(5)]],
  constant int64_t* input_strides [[buffer(6)]],
  constant int64_t* output_sizes [[buffer(7)]],
  constant int64_t* output_strides [[buffer(8)]],
  device int64_t* work_pooling_dim_indices_ [[buffer(9)]],
  constant int32_t& nthreads[[buffer(10)]],
  constant int64_t* kernel_size [[buffer(11)]],
  constant int64_t* stride [[buffer(12)]],
  constant int64_t* padding [[buffer(13)]],
  constant int64_t* dilation [[buffer(14)]],
  uint tid [[thread_position_in_grid]]
) {
  int64_t leading_dims = dims - pooling_dims;
  constant T* input = reinterpret_cast<constant T*>(input_);
  device T* output = reinterpret_cast<device T*>(output_);
  device int64_t* indices = reinterpret_cast<device int64_t*>(indices_);

  // This buffer keeps track of the pooling dimension indices of this thread's
  // element of the output. We need to fill it with the proper values below.
  device int64_t* work_pooling_dim_indices = work_pooling_dim_indices_ + tid * pooling_dims;
  int64_t output_idx = static_cast<int64_t>(tid);
  int64_t output_offset = 0;
  int64_t input_leading_offset = 0;

  // First, find the offset of the output element this thread will calculate,
  // `output[N, C, d, h, w]`. Also, find the offset of the input for the leading
  // dim indices, `input[N, C]` and keep track of the pooling dimension indices,
  // `[d, h , w]`.
  for (int64_t dim = dims - 1; dim >= 0; dim--) {
    int64_t dim_idx = output_idx % (output_sizes[dim]);
    output_offset += output_strides[dim] * dim_idx;

    if (dim < leading_dims) {
      input_leading_offset += input_strides[dim] * dim_idx;
    } else {
      // Keep track of pooling dimension indices of the output element, so we
      // can use them in the input iteration later on.
      work_pooling_dim_indices[dim - leading_dims] = dim_idx;
    }
    output_idx = output_idx / output_sizes[dim];
  }
  output += output_offset;
  indices += output_offset;
  input += input_leading_offset;

  max_pool_3d_input_iter<T>(
    input
    ,output
    ,indices
    ,input_strides + leading_dims
    ,work_pooling_dim_indices
    ,kernel_size
    ,stride
    ,padding
    ,dilation
  );
}

#define REGISTER_MAX_POOL_OP(DTYPE)                                     \
template [[host_name("max_pool_" #DTYPE)]] kernel void max_pool<DTYPE>( \
  constant void* input_ [[buffer(0)]],                                  \
  device void* output_ [[buffer(1)]],                                   \
  device void* indices_ [[buffer(2)]],                                  \
  constant int64_t& dims [[buffer(3)]],                                 \
  constant int64_t& pooling_dims [[buffer(4)]],                         \
  constant int64_t* input_sizes [[buffer(5)]],                          \
  constant int64_t* input_strides [[buffer(6)]],                        \
  constant int64_t* output_sizes [[buffer(7)]],                         \
  constant int64_t* output_strides [[buffer(8)]],                       \
  device int64_t* work_pooling_dim_indices_ [[buffer(9)]],              \
  constant int32_t& nthreads[[buffer(10)]],                             \
  constant int64_t* kernel_size [[buffer(11)]],                         \
  constant int64_t* stride [[buffer(12)]],                              \
  constant int64_t* padding [[buffer(13)]],                             \
  constant int64_t* dilation [[buffer(14)]],                            \
  uint tid [[thread_position_in_grid]]);

REGISTER_MAX_POOL_OP(float);
REGISTER_MAX_POOL_OP(half);
REGISTER_MAX_POOL_OP(int);
REGISTER_MAX_POOL_OP(long);
REGISTER_MAX_POOL_OP(short);
REGISTER_MAX_POOL_OP(char);
REGISTER_MAX_POOL_OP(uchar);
REGISTER_MAX_POOL_OP(bool);
#if __METAL_VERSION__ >= 310
REGISTER_MAX_POOL_OP(bfloat);
#endif
