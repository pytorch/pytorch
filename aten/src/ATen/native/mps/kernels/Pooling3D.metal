#include <metal_stdlib>
using namespace metal;

// 3D Average Pooling Kernel
// This kernel computes the average pooling operation for 3D tensors
// Using Metal's built-in packed_int4 type for more efficient parameter passing

template <typename T>
kernel void avg_pool3d(
    device T* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant int& batch_size [[buffer(2)]],
    constant int& channels [[buffer(3)]],
    constant packed_int4& input_dims
    [[buffer(4)]], // [input_depth, input_height, input_width, output_depth]
    constant packed_int4& output_dims
    [[buffer(5)]], // [output_height, output_width, kernel_depth, kernel_height]
    constant packed_int4& kernel_dims
    [[buffer(6)]], // [kernel_width, stride_depth, stride_height, stride_width]
    constant packed_int4& padding_dims
    [[buffer(7)]], // [padding_depth, padding_height, padding_width,
                   // count_include_pad]
    constant int& divisor_override [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
  // Extract dimensions from packed vectors
  int input_depth = input_dims.x;
  int input_height = input_dims.y;
  int input_width = input_dims.z;
  int output_depth = input_dims.w;

  int output_height = output_dims.x;
  int output_width = output_dims.y;
  int kernel_depth = output_dims.z;
  int kernel_height = output_dims.w;

  int kernel_width = kernel_dims.x;
  int stride_depth = kernel_dims.y;
  int stride_height = kernel_dims.z;
  int stride_width = kernel_dims.w;

  int padding_depth = padding_dims.x;
  int padding_height = padding_dims.y;
  int padding_width = padding_dims.z;
  int count_include_pad = padding_dims.w;

  // Calculate output indices
  const int output_size =
      batch_size * channels * output_depth * output_height * output_width;
  if (index >= output_size)
    return;

  // Calculate position in output tensor
  int ow = index % output_width;
  int oh = (index / output_width) % output_height;
  int od = (index / (output_width * output_height)) % output_depth;
  int c = (index / (output_width * output_height * output_depth)) % channels;
  int n = index / (output_width * output_height * output_depth * channels);

  // Calculate input position (top-left corner of pooling window)
  int id_start = od * stride_depth - padding_depth;
  int ih_start = oh * stride_height - padding_height;
  int iw_start = ow * stride_width - padding_width;

  // Calculate input position (bottom-right corner of pooling window)
  int id_end = id_start + kernel_depth;
  int ih_end = ih_start + kernel_height;
  int iw_end = iw_start + kernel_width;

  // Adjust to valid input range for actual computation
  int valid_id_start = max(0, id_start);
  int valid_ih_start = max(0, ih_start);
  int valid_iw_start = max(0, iw_start);

  int valid_id_end = min(id_end, input_depth);
  int valid_ih_end = min(ih_end, input_height);
  int valid_iw_end = min(iw_end, input_width);

  // Calculate pool size
  int pool_size;
  if (count_include_pad) {
    pool_size = kernel_depth * kernel_height * kernel_width;
  } else {
    pool_size = (valid_id_end - valid_id_start) *
        (valid_ih_end - valid_ih_start) * (valid_iw_end - valid_iw_start);
  }

  // Use custom divisor if provided
  if (divisor_override > 0) {
    pool_size = divisor_override;
  }

  // Compute average
  T sum = 0;
  for (int id = valid_id_start; id < valid_id_end; id++) {
    for (int ih = valid_ih_start; ih < valid_ih_end; ih++) {
      for (int iw = valid_iw_start; iw < valid_iw_end; iw++) {
        int input_idx = ((n * channels + c) * input_depth + id) * input_height *
                input_width +
            ih * input_width + iw;
        sum += input[input_idx];
      }
    }
  }

  // Write output
  if (pool_size > 0) {
    output[index] = sum / static_cast<T>(pool_size);
  } else {
    output[index] = 0;
  }
}

// 3D Average Pooling Backward Kernel
// This kernel computes the gradient for 3D average pooling operation
template <typename T>
kernel void avg_pool3d_backward(
    device T* grad_input [[buffer(0)]],
    constant T* grad_output [[buffer(1)]],
    constant int& batch_size [[buffer(2)]],
    constant int& channels [[buffer(3)]],
    constant packed_int4& input_dims
    [[buffer(4)]], // [input_depth, input_height, input_width, output_depth]
    constant packed_int4& output_dims
    [[buffer(5)]], // [output_height, output_width, kernel_depth, kernel_height]
    constant packed_int4& kernel_dims
    [[buffer(6)]], // [kernel_width, stride_depth, stride_height, stride_width]
    constant packed_int4& padding_dims
    [[buffer(7)]], // [padding_depth, padding_height, padding_width,
                   // count_include_pad]
    constant int& divisor_override [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
  // Extract dimensions from packed vectors
  int input_depth = input_dims.x;
  int input_height = input_dims.y;
  int input_width = input_dims.z;
  int output_depth = input_dims.w;

  int output_height = output_dims.x;
  int output_width = output_dims.y;
  int kernel_depth = output_dims.z;
  int kernel_height = output_dims.w;

  int kernel_width = kernel_dims.x;
  int stride_depth = kernel_dims.y;
  int stride_height = kernel_dims.z;
  int stride_width = kernel_dims.w;

  int padding_depth = padding_dims.x;
  int padding_height = padding_dims.y;
  int padding_width = padding_dims.z;
  int count_include_pad = padding_dims.w;

  // Calculate input indices
  const int input_size =
      batch_size * channels * input_depth * input_height * input_width;
  if (index >= input_size)
    return;

  // Calculate position in input tensor
  int iw = index % input_width;
  int ih = (index / input_width) % input_height;
  int id = (index / (input_width * input_height)) % input_depth;
  int c = (index / (input_width * input_height * input_depth)) % channels;
  int n = index / (input_width * input_height * input_depth * channels);

  // Initialize gradient
  T gradient = 0;

  // Iterate over all output positions that could contribute to this input
  // position
  for (int od = 0; od < output_depth; od++) {
    // Compute the window boundaries for this output position
    int id_start = od * stride_depth - padding_depth;
    int id_end = min(id_start + kernel_depth, input_depth + padding_depth);

    // Check if this input position is within the window
    if (id < id_start || id >= id_end)
      continue;

    for (int oh = 0; oh < output_height; oh++) {
      // Compute the window boundaries for this output position
      int ih_start = oh * stride_height - padding_height;
      int ih_end = min(ih_start + kernel_height, input_height + padding_height);

      // Check if this input position is within the window
      if (ih < ih_start || ih >= ih_end)
        continue;

      for (int ow = 0; ow < output_width; ow++) {
        // Compute the window boundaries for this output position
        int iw_start = ow * stride_width - padding_width;
        int iw_end = min(iw_start + kernel_width, input_width + padding_width);

        // Check if this input position is within the window
        if (iw < iw_start || iw >= iw_end)
          continue;

        // Adjust window boundaries to valid input range
        int valid_id_start = max(0, id_start);
        int valid_ih_start = max(0, ih_start);
        int valid_iw_start = max(0, iw_start);

        int valid_id_end = min(id_end, input_depth);
        int valid_ih_end = min(ih_end, input_height);
        int valid_iw_end = min(iw_end, input_width);

        // Calculate pool size
        int pool_size;
        if (count_include_pad) {
          pool_size = kernel_depth * kernel_height * kernel_width;
        } else {
          pool_size = (valid_id_end - valid_id_start) *
              (valid_ih_end - valid_ih_start) * (valid_iw_end - valid_iw_start);
        }

        // Use custom divisor if provided
        if (divisor_override > 0) {
          pool_size = divisor_override;
        }

        // Get the gradient from output
        int output_idx = ((n * channels + c) * output_depth + od) *
                output_height * output_width +
            oh * output_width + ow;

        // Distribute the gradient
        if (pool_size > 0) {
          gradient += grad_output[output_idx] / static_cast<T>(pool_size);
        }
      }
    }
  }

  // Write gradient to output
  grad_input[index] = gradient;
}

// Register kernels for different data types
#define REGISTER_AVG_POOL3D_KERNELS(DTYPE)                                    \
  template [[host_name("avg_pool3d_" #DTYPE)]] kernel void avg_pool3d<DTYPE>( \
      device DTYPE * output [[buffer(0)]],                                    \
      constant DTYPE * input [[buffer(1)]],                                   \
      constant int& batch_size [[buffer(2)]],                                 \
      constant int& channels [[buffer(3)]],                                   \
      constant packed_int4& input_dims [[buffer(4)]],                         \
      constant packed_int4& output_dims [[buffer(5)]],                        \
      constant packed_int4& kernel_dims [[buffer(6)]],                        \
      constant packed_int4& padding_dims [[buffer(7)]],                       \
      constant int& divisor_override [[buffer(8)]],                           \
      uint index [[thread_position_in_grid]]);                                \
                                                                              \
  template [[host_name("avg_pool3d_backward_" #DTYPE)]] kernel void           \
  avg_pool3d_backward<DTYPE>(                                                 \
      device DTYPE * grad_input [[buffer(0)]],                                \
      constant DTYPE * grad_output [[buffer(1)]],                             \
      constant int& batch_size [[buffer(2)]],                                 \
      constant int& channels [[buffer(3)]],                                   \
      constant packed_int4& input_dims [[buffer(4)]],                         \
      constant packed_int4& output_dims [[buffer(5)]],                        \
      constant packed_int4& kernel_dims [[buffer(6)]],                        \
      constant packed_int4& padding_dims [[buffer(7)]],                       \
      constant int& divisor_override [[buffer(8)]],                           \
      uint index [[thread_position_in_grid]])

// Register for float
REGISTER_AVG_POOL3D_KERNELS(float);
// Register for half (float16)
REGISTER_AVG_POOL3D_KERNELS(half);
// Register for bfloat16 if Metal version supports it
#if __METAL_VERSION__ >= 310
REGISTER_AVG_POOL3D_KERNELS(bfloat);
#endif
