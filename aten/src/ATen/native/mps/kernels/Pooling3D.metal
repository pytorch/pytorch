#include <metal_stdlib>
using namespace metal;

// 3D Average Pooling Kernel
// This kernel computes the average pooling operation for 3D tensors
template <typename T>
kernel void avg_pool3d(
    device T* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant int& batch_size [[buffer(2)]],
    constant int& channels [[buffer(3)]],
    constant int& input_depth [[buffer(4)]],
    constant int& input_height [[buffer(5)]],
    constant int& input_width [[buffer(6)]],
    constant int& output_depth [[buffer(7)]],
    constant int& output_height [[buffer(8)]],
    constant int& output_width [[buffer(9)]],
    constant int& kernel_depth [[buffer(10)]],
    constant int& kernel_height [[buffer(11)]],
    constant int& kernel_width [[buffer(12)]],
    constant int& stride_depth [[buffer(13)]],
    constant int& stride_height [[buffer(14)]],
    constant int& stride_width [[buffer(15)]],
    constant int& padding_depth [[buffer(16)]],
    constant int& padding_height [[buffer(17)]],
    constant int& padding_width [[buffer(18)]],
    constant int& count_include_pad [[buffer(19)]],
    constant int& divisor_override [[buffer(20)]],
    uint index [[thread_position_in_grid]])
{
    // Calculate output indices
    const int output_size = batch_size * channels * output_depth * output_height * output_width;
    if (index >= output_size) return;

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
    int id_end = min(id_start + kernel_depth, input_depth + padding_depth);
    int ih_end = min(ih_start + kernel_height, input_height + padding_height);
    int iw_end = min(iw_start + kernel_width, input_width + padding_width);

    // Adjust to valid input range
    id_start = max(0, id_start);
    ih_start = max(0, ih_start);
    iw_start = max(0, iw_start);

    id_end = min(id_end, input_depth);
    ih_end = min(ih_end, input_height);
    iw_end = min(iw_end, input_width);

    // Calculate pool size
    int pool_size;
    if (count_include_pad) {
        pool_size = kernel_depth * kernel_height * kernel_width;
    } else {
        pool_size = (id_end - id_start) * (ih_end - ih_start) * (iw_end - iw_start);
    }

    // Use custom divisor if provided
    if (divisor_override > 0) {
        pool_size = divisor_override;
    }

    // Compute average
    T sum = 0;
    for (int id = id_start; id < id_end; id++) {
        for (int ih = ih_start; ih < ih_end; ih++) {
            for (int iw = iw_start; iw < iw_end; iw++) {
                int input_idx = ((n * channels + c) * input_depth + id) * input_height * input_width +
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
    constant int& input_depth [[buffer(4)]],
    constant int& input_height [[buffer(5)]],
    constant int& input_width [[buffer(6)]],
    constant int& output_depth [[buffer(7)]],
    constant int& output_height [[buffer(8)]],
    constant int& output_width [[buffer(9)]],
    constant int& kernel_depth [[buffer(10)]],
    constant int& kernel_height [[buffer(11)]],
    constant int& kernel_width [[buffer(12)]],
    constant int& stride_depth [[buffer(13)]],
    constant int& stride_height [[buffer(14)]],
    constant int& stride_width [[buffer(15)]],
    constant int& padding_depth [[buffer(16)]],
    constant int& padding_height [[buffer(17)]],
    constant int& padding_width [[buffer(18)]],
    constant int& count_include_pad [[buffer(19)]],
    constant int& divisor_override [[buffer(20)]],
    uint index [[thread_position_in_grid]])
{
    // Calculate input indices
    const int input_size = batch_size * channels * input_depth * input_height * input_width;
    if (index >= input_size) return;

    // Calculate position in input tensor
    int iw = index % input_width;
    int ih = (index / input_width) % input_height;
    int id = (index / (input_width * input_height)) % input_depth;
    int c = (index / (input_width * input_height * input_depth)) % channels;
    int n = index / (input_width * input_height * input_depth * channels);

    // Initialize gradient
    T gradient = 0;

    // Iterate over all output positions that could contribute to this input position
    for (int od = 0; od < output_depth; od++) {
        // Compute the window boundaries for this output position
        int id_start = od * stride_depth - padding_depth;
        int id_end = min(id_start + kernel_depth, input_depth + padding_depth);

        // Check if this input position is within the window
        if (id < id_start || id >= id_end) continue;

        for (int oh = 0; oh < output_height; oh++) {
            // Compute the window boundaries for this output position
            int ih_start = oh * stride_height - padding_height;
            int ih_end = min(ih_start + kernel_height, input_height + padding_height);

            // Check if this input position is within the window
            if (ih < ih_start || ih >= ih_end) continue;

            for (int ow = 0; ow < output_width; ow++) {
                // Compute the window boundaries for this output position
                int iw_start = ow * stride_width - padding_width;
                int iw_end = min(iw_start + kernel_width, input_width + padding_width);

                // Check if this input position is within the window
                if (iw < iw_start || iw >= iw_end) continue;

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
                    pool_size = (valid_id_end - valid_id_start) * (valid_ih_end - valid_ih_start) * (valid_iw_end - valid_iw_start);
                }

                // Use custom divisor if provided
                if (divisor_override > 0) {
                    pool_size = divisor_override;
                }

                // Get the gradient from output
                int output_idx = ((n * channels + c) * output_depth + od) * output_height * output_width +
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
#define REGISTER_AVG_POOL3D_KERNELS(DTYPE)                                  \
template                                                                    \
[[host_name("avg_pool3d_" #DTYPE)]]                                         \
kernel void avg_pool3d<DTYPE>(                                              \
    device DTYPE* output [[buffer(0)]],                                     \
    constant DTYPE* input [[buffer(1)]],                                    \
    constant int& batch_size [[buffer(2)]],                                 \
    constant int& channels [[buffer(3)]],                                   \
    constant int& input_depth [[buffer(4)]],                                \
    constant int& input_height [[buffer(5)]],                               \
    constant int& input_width [[buffer(6)]],                                \
    constant int& output_depth [[buffer(7)]],                               \
    constant int& output_height [[buffer(8)]],                              \
    constant int& output_width [[buffer(9)]],                               \
    constant int& kernel_depth [[buffer(10)]],                              \
    constant int& kernel_height [[buffer(11)]],                             \
    constant int& kernel_width [[buffer(12)]],                              \
    constant int& stride_depth [[buffer(13)]],                              \
    constant int& stride_height [[buffer(14)]],                             \
    constant int& stride_width [[buffer(15)]],                              \
    constant int& padding_depth [[buffer(16)]],                             \
    constant int& padding_height [[buffer(17)]],                            \
    constant int& padding_width [[buffer(18)]],                             \
    constant int& count_include_pad [[buffer(19)]],                         \
    constant int& divisor_override [[buffer(20)]],                          \
    uint index [[thread_position_in_grid]]);                                \
                                                                            \
template                                                                    \
[[host_name("avg_pool3d_backward_" #DTYPE)]]                                \
kernel void avg_pool3d_backward<DTYPE>(                                     \
    device DTYPE* grad_input [[buffer(0)]],                                 \
    constant DTYPE* grad_output [[buffer(1)]],                              \
    constant int& batch_size [[buffer(2)]],                                 \
    constant int& channels [[buffer(3)]],                                   \
    constant int& input_depth [[buffer(4)]],                                \
    constant int& input_height [[buffer(5)]],                               \
    constant int& input_width [[buffer(6)]],                                \
    constant int& output_depth [[buffer(7)]],                               \
    constant int& output_height [[buffer(8)]],                              \
    constant int& output_width [[buffer(9)]],                               \
    constant int& kernel_depth [[buffer(10)]],                              \
    constant int& kernel_height [[buffer(11)]],                             \
    constant int& kernel_width [[buffer(12)]],                              \
    constant int& stride_depth [[buffer(13)]],                              \
    constant int& stride_height [[buffer(14)]],                             \
    constant int& stride_width [[buffer(15)]],                              \
    constant int& padding_depth [[buffer(16)]],                             \
    constant int& padding_height [[buffer(17)]],                            \
    constant int& padding_width [[buffer(18)]],                             \
    constant int& count_include_pad [[buffer(19)]],                         \
    constant int& divisor_override [[buffer(20)]],                          \
    uint index [[thread_position_in_grid]])

// Register for float
REGISTER_AVG_POOL3D_KERNELS(float);
// Register for half (float16)
REGISTER_AVG_POOL3D_KERNELS(half);
// Register for bfloat16 if Metal version supports it
// Disabled for now due to compatibility issues
// #if __METAL_VERSION__ >= 310
// REGISTER_AVG_POOL3D_KERNELS(bfloat);
// #endif
