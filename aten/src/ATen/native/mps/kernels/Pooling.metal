#include <ATen/native/mps/kernels/Pooling.h>
#include <c10/metal/atomic.h>
#include <metal_array>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

// Iterates through all the input elements that this kernel needs to
// apply max to. Specialized for 3 pooling dimensions.
// TODO: Support any number of pooling dims
template <typename T>
void max_pool_3d_input_iter(
    constant T* input,
    device T* output,
    device int64_t* indices,
    constant int64_t* input_sizes,
    constant int64_t* input_strides,
    device int64_t* work_pooling_dim_indices,
    constant int64_t* kernel_size,
    constant int64_t* stride,
    constant int64_t* padding,
    constant int64_t* dilation) {
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

  T max_value = 0;
  int64_t max_index = -1;

  int64_t size12 = input_sizes[1] * input_sizes[2];

  for (int64_t i0 = (s0 * o0) - padding[0];
       i0 < (s0 * o0 - padding[0] + k0 * d0) && i0 < input_sizes[0];
       i0 += d0) {
    if (i0 < 0) {
      continue;
    }
    int64_t offset0 = input_strides[0] * i0;

    for (int64_t i1 = (s1 * o1) - padding[1];
         i1 < (s1 * o1 - padding[1] + k1 * d1) && i1 < input_sizes[1];
         i1 += d1) {
      if (i1 < 0) {
        continue;
      }
      int64_t offset1 = input_strides[1] * i1;

      for (int64_t i2 = (s2 * o2) - padding[2];
           i2 < (s2 * o2 - padding[2] + k2 * d2) && i2 < input_sizes[2];
           i2 += d2) {
        if (i2 < 0) {
          continue;
        }
        int64_t offset2 = input_strides[2] * i2;

        const T input_value = input[offset0 + offset1 + offset2];
        int64_t input_index = i0 * size12 + i1 * input_sizes[2] + i2;

        T new_max_value = (max_index == -1 || input_value > max_value)
            ? input_value
            : max_value;
        int64_t new_max_index = (max_index == -1 || input_value > max_value)
            ? input_index
            : max_index;

        max_value = new_max_value;
        max_index = new_max_index;
      }
    }
  }

  *output = max_value;
  *indices = max_index;
}

struct PoolOffsets {
  int64_t output;
  int64_t indices;
  int64_t input_leading;

  PoolOffsets() : output(0), indices(0), input_leading(0) {}
};

// Finds the offset of the output element that a forward pass thread will
// calculate, `output[N, C, d, h, w]`. Also, find the offset of the input for
// the leading dim indices, `input[N, C]`. Optionally, keep track of the output
// pooling dimension indices, `[d, h , w]`.
PoolOffsets find_pool_offsets(
    constant int64_t* output_sizes,
    constant int64_t* output_strides,
    constant int64_t* indices_strides,
    constant int64_t* input_strides,
    device int64_t* work_pooling_dim_indices,
    int32_t dims,
    int32_t leading_dims,
    uint tid) {
  int64_t output_idx = static_cast<int64_t>(tid);
  PoolOffsets offsets;

  for (int64_t dim = dims - 1; dim >= 0; dim--) {
    int64_t dim_idx = output_idx % (output_sizes[dim]);
    offsets.output += output_strides[dim] * dim_idx;
    offsets.indices += indices_strides[dim] * dim_idx;

    if (dim < leading_dims) {
      offsets.input_leading += input_strides[dim] * dim_idx;
    } else {
      // Keep track of pooling dimension indices of the output element, so we
      // can use them in the input iteration later on.
      if (work_pooling_dim_indices != nullptr) {
        work_pooling_dim_indices[dim - leading_dims] = dim_idx;
      }
    }
    output_idx = output_idx / output_sizes[dim];
  }

  return offsets;
}

// Kernel computes one element of the output per kernel call.
template <typename T>
kernel void max_pool(
    constant void* input_ [[buffer(0)]],
    device void* output_ [[buffer(1)]],
    device void* indices_ [[buffer(2)]],
    device int64_t* work_pooling_dim_indices_ [[buffer(3)]],
    constant PoolingParams<5>& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  int32_t pooling_dims = params.pooling_dims;
  int32_t dims = params.dims;
  constant int64_t* input_sizes = params.input_sizes.data();
  constant int64_t* input_strides = params.input_strides.data();
  constant int64_t* output_sizes = params.output_sizes.data();
  constant int64_t* output_strides = params.output_strides.data();
  constant int64_t* indices_strides = params.indices_strides.data();
  constant int64_t* kernel_size = params.kernel_size.data();
  constant int64_t* stride = params.stride.data();
  constant int64_t* padding = params.padding.data();
  constant int64_t* dilation = params.dilation.data();

  int32_t leading_dims = dims - pooling_dims;
  constant T* input = reinterpret_cast<constant T*>(input_);
  device T* output = reinterpret_cast<device T*>(output_);
  device int64_t* indices = reinterpret_cast<device int64_t*>(indices_);

  // This buffer keeps track of the pooling dimension indices of this thread's
  // element of the output. We need to fill it with the proper values below.
  device int64_t* work_pooling_dim_indices =
      work_pooling_dim_indices_ + tid * pooling_dims;

  PoolOffsets offsets = find_pool_offsets(
      output_sizes,
      output_strides,
      indices_strides,
      input_strides,
      work_pooling_dim_indices,
      dims,
      leading_dims,
      tid);

  output += offsets.output;
  indices += offsets.indices;
  input += offsets.input_leading;

  max_pool_3d_input_iter<T>(
      input,
      output,
      indices,
      input_sizes + leading_dims,
      input_strides + leading_dims,
      work_pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      dilation);
}

// Finds the element in the grad input which corresponds to the index into the
// pool, and then adds the grad output element to it.
template <typename T>
void max_pool_backward_impl(
    device AtomicType_t<T>* grad_input,
    T grad_output_element,
    int32_t input_index,
    constant int64_t* grad_input_sizes,
    constant int64_t* grad_input_strides,
    int32_t grad_input_leading_offset,
    int32_t pooling_dims) {
  int32_t size_prod = 1;
  int32_t pool_offset = 0;

  for (int32_t dim = pooling_dims - 1; dim >= 0; dim--) {
    int32_t next_size_prod = grad_input_sizes[dim] * size_prod;
    pool_offset +=
        grad_input_strides[dim] * ((input_index % next_size_prod) / size_prod);
    size_prod *= grad_input_sizes[dim];
  }

  AtomicType<T>::atomic_add(
      grad_input, grad_input_leading_offset + pool_offset, grad_output_element);
}

// Kernel computes one element of the grad input per kernel call.
template <typename T>
kernel void max_pool_backward(
    device AtomicType_t<T>* grad_input [[buffer(0)]],
    constant T* grad_output [[buffer(1)]],
    constant int64_t* indices [[buffer(2)]],
    constant PoolingBackwardParams<5>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int32_t pooling_dims = params.pooling_dims;
  int32_t dims = params.dims;
  constant int64_t* grad_input_sizes = params.grad_input_sizes.data();
  constant int64_t* grad_input_strides = params.grad_input_strides.data();
  constant int64_t* grad_output_sizes = params.grad_output_sizes.data();
  constant int64_t* grad_output_strides = params.grad_output_strides.data();
  constant int64_t* indices_strides = params.indices_strides.data();

  int32_t leading_dims = dims - pooling_dims;

  PoolOffsets offsets = find_pool_offsets(
      grad_output_sizes,
      grad_output_strides,
      indices_strides,
      grad_input_strides,
      nullptr,
      dims,
      leading_dims,
      tid);

  max_pool_backward_impl<T>(
      grad_input,
      grad_output[offsets.output],
      indices[offsets.indices],
      grad_input_sizes + leading_dims,
      grad_input_strides + leading_dims,
      offsets.input_leading,
      pooling_dims);
}

#define REGISTER_MAX_POOL_OP(DTYPE)                                       \
  template [[host_name("max_pool_" #DTYPE)]] kernel void max_pool<DTYPE>( \
      constant void* input_ [[buffer(0)]],                                \
      device void* output_ [[buffer(1)]],                                 \
      device void* indices_ [[buffer(2)]],                                \
      device int64_t* work_pooling_dim_indices_ [[buffer(3)]],            \
      constant PoolingParams<5>& params [[buffer(4)]],                    \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_MAX_POOL_BACKWARD_OP(DTYPE)                   \
  template [[host_name("max_pool_backward_" #DTYPE)]]          \
  kernel void max_pool_backward<DTYPE>(                        \
      device AtomicType_t<DTYPE> * grad_input [[buffer(0)]],   \
      constant DTYPE * grad_output_ [[buffer(1)]],             \
      constant int64_t* grad_indices_ [[buffer(2)]],           \
      constant PoolingBackwardParams<5>& params [[buffer(3)]], \
      uint tid [[thread_position_in_grid]]);

REGISTER_MAX_POOL_OP(float);
REGISTER_MAX_POOL_OP(half);
REGISTER_MAX_POOL_OP(int);
REGISTER_MAX_POOL_OP(long);
REGISTER_MAX_POOL_OP(short);
REGISTER_MAX_POOL_OP(char);
REGISTER_MAX_POOL_OP(uchar);
REGISTER_MAX_POOL_OP(bool);

REGISTER_MAX_POOL_BACKWARD_OP(float);
REGISTER_MAX_POOL_BACKWARD_OP(half);

#if __METAL_VERSION__ >= 310
REGISTER_MAX_POOL_OP(bfloat);
REGISTER_MAX_POOL_BACKWARD_OP(bfloat);
#endif
