#pragma once

#include <assert.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <limits>
#include <stdint.h>
#include <cuda_fp16.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Warp Softmax forward
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate over all data.
// WARP_SIZE number of elements working on a single batch, has to be a power of two.
template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH, int WARP_ITERATIONS, int WARP_SIZE = 32, bool is_log_softmax=true>
__global__ void softmax_warp_forward(output_t *dst, const input_t *src, int batch_size, int stride, int element_count)
{
    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;

    src += first_batch * stride + local_idx;
    dst += first_batch * stride + local_idx;

    // load data from global memory
    acc_t elements[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0;i < WARP_BATCH;++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        for (int it = 0;it < WARP_ITERATIONS;it += 1) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < batch_element_count) {
                elements[i][it] = src[i*element_count+it*WARP_SIZE];
            } else {
                elements[i][it] = -std::numeric_limits<acc_t>::infinity();
            }
        }
    }

    constexpr uint32_t  FULL_MASK = 0xffffffff;

    // compute local max_value

    // take the max_value of the first element to avoid one max call
    acc_t max_value[WARP_BATCH];
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        max_value[i] = elements[i][0];
    }

    #pragma unroll
    for (int it = 1;it < WARP_ITERATIONS;++it) {
        for (int i = 0;i < WARP_BATCH;++i) {
            max_value[i] = (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
        }
    }

    // reduction max_value
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        acc_t val[WARP_BATCH];
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            val[i] = WARP_SHFL_XOR(max_value[i], offset, WARP_SIZE, FULL_MASK);
        }
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            max_value[i] = max_value[i] > val[i] ? max_value[i] : val[i];
        }
    }

    // compute local sum
    acc_t sum[WARP_BATCH] { 0.0f };

    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        for (int it = 0;it < WARP_ITERATIONS;++it) {
            if (is_log_softmax) {
              sum[i] += std::exp(elements[i][it] - max_value[i]);
            } else {
              elements[i][it] = std::exp(elements[i][it] - max_value[i]);
              sum[i] += elements[i][it];
            }
        }
    }

    // reduction sum
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            sum[i] += WARP_SHFL_XOR(sum[i], offset, WARP_SIZE, FULL_MASK);
        }
    }

    // store result
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        if (i >= local_batches)
            break;
        if (is_log_softmax) sum[i] = max_value[i] + std::log(sum[i]);
        #pragma unroll
        for (int it = 0;it < WARP_ITERATIONS;it += 1) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                if (is_log_softmax) {
                    dst[i*element_count+it*WARP_SIZE] = elements[i][it] - sum[i];
                } else {
                    dst[i*element_count+it*WARP_SIZE] = elements[i][it] / sum[i];
                }
            } else {
                break;
            }
        }
    }
}


// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate over all data.
// WARP_SIZE number of elements working on a single batch, has to be a power of two.
template <typename input_t, typename output_t>
using softmax_forward_func = void(*)(output_t *dst, const input_t *src, int batch_size, int stride, int element_count);

template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
bool warp_softmax_kernel(int log2_elements, int &warp_size, int &batches_per_warp, softmax_forward_func<input_t, output_t> &kernel) {
    // determine size of a warp
    const int next_power_of_two = 1 << log2_elements;
    warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;

    // determine how many batches a warp should process.
    batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

    switch (log2_elements) {
    case 0: // 1
        kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2,1,1, is_log_softmax>;
        break;
    case 1: // 2
        kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2,1,2, is_log_softmax>;
        break;
    case 2: // 4
        kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2,1,4, is_log_softmax>;
        break;
    case 3: // 8
        kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2,1,8, is_log_softmax>;
        break;
    case 4: // 16
        kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2,1,16, is_log_softmax>;
        break;
    case 5: // 32
        kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2,1,32, is_log_softmax>;
        break;
    case 6: // 64
        kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2,2,32, is_log_softmax>;
        break;
    case 7: // 128
        kernel = &softmax_warp_forward<input_t, output_t, acc_t, 2,4,32, is_log_softmax>;
        break;
    case 8: // 256
        kernel = &softmax_warp_forward<input_t, output_t, acc_t, 1,8,32, is_log_softmax>;
        break;
    case 9: // 512
        kernel = &softmax_warp_forward<input_t, output_t, acc_t, 1,16,32, is_log_softmax>;
        break;
    case 10: // 1024
        kernel = &softmax_warp_forward<input_t, output_t, acc_t, 1,32,32, is_log_softmax>;
        break;
    default:
        return false;
    }
    return true;
}

template<typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
bool dispatch_softmax(output_t *dst, const input_t *src, int softmax_elements, int softmax_elements_stride, int batch_count)
{
    if (softmax_elements == 0) {
        return true;
    } else if (softmax_elements <= 1024) {
        // compute function index. there's a function for each power of two size up to 1024.
        int log2_elements = 0;
        while ((1 << log2_elements) < softmax_elements) ++log2_elements;

        softmax_forward_func<input_t, output_t> kernel;
        int warp_size, batches_per_warp;
        if (!warp_softmax_kernel<input_t, output_t, acc_t, is_log_softmax>(log2_elements, warp_size, batches_per_warp, kernel)) {
            return false;
        }

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        // compute warps per block.
        int warps_per_block = (threads_per_block / warp_size);

        // compute launch size
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);

        // launch
        kernel<<<blocks, threads>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
        return true;
    }
    return false;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Warp softmax backward
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename input_t, typename output_t, typename acc_t, int WARP_BATCH, int WARP_ITERATIONS, int WARP_SIZE=32, bool is_log_softmax>
__global__ void softmax_warp_backward(output_t *gradInput, const input_t *grad, const input_t *output, int batch_size, int stride, int element_count)
{
    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x % WARP_SIZE;

    // the first element to process by the current thread
    int thread_offset = first_batch * stride + local_idx;
    grad += thread_offset;
    output += thread_offset;
    gradInput += thread_offset;

    // load data from global memory
    acc_t grad_reg[WARP_BATCH][WARP_ITERATIONS];
    acc_t output_reg[WARP_BATCH][WARP_ITERATIONS];
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        #pragma unroll
        for (int it = 0;it < WARP_ITERATIONS;it += 1) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < batch_element_count) {
                grad_reg[i][it] = grad[i*element_count+it*WARP_SIZE];
                output_reg[i][it] = output[i*element_count+it*WARP_SIZE];
            } else {
                grad_reg[i][it] = acc_t(0);
                output_reg[i][it] = acc_t(0);
            }
        }
    }

    // compute thread local sum
    acc_t sum[WARP_BATCH] = {0};
    #pragma unroll
    for (int it = 0;it < WARP_ITERATIONS;++it) {
        for (int i = 0;i < WARP_BATCH;++i) {
            sum[i] += grad_reg[i][it];
        }
    }

    // reduction sum
    constexpr uint32_t FULL_MASK = 0xffffffff;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;i < WARP_BATCH;++i) {
            sum[i] += WARP_SHFL_XOR(sum[i], offset, WARP_SIZE, FULL_MASK);
        }
    }

    // store result
    #pragma unroll
    for (int i = 0;i < WARP_BATCH;++i) {
        if (i >= local_batches)
            break;
        #pragma unroll
        for (int it = 0;it < WARP_ITERATIONS;it += 1) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                // compute gradients
                if (is_log_softmax) {
                      gradInput[i*element_count+it*WARP_SIZE] = (grad_reg[i][it] - std::exp(output_reg[i][it]) * sum[i]);
                } else {
                    gradInput[i*element_count+it*WARP_SIZE] = (grad_reg[i][it] - output_reg[i][it] * sum[i]);
                }
            }
        }
    }
}



// WARP_BATCH number of batches.
// WARP_ITERATOINS The number of iterations required for one warp to iterate over all data.
// WARP_SIZE number of elements working on a single batch, has to be a power of two.
template <typename input_t, typename output_t>
using softmax_backward_func = void(*)(output_t *gradInput, const input_t *grad, const input_t *output, int batch_size, int stride, int element_count);

template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
bool warp_softmax_backward_kernel(int log2_elements, int &warp_size, int &batches_per_warp, softmax_backward_func<input_t, output_t> &kernel) {
    // determine size of a warp
    const int next_power_of_two = 1 << log2_elements;
    warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;

    // determine how many batches a warp should process.
    batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

    switch (log2_elements) {
    case 0: // 1
        kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2,1,1, is_log_softmax>;
        break;
    case 1: // 2
        kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2,1,2, is_log_softmax>;
        break;
    case 2: // 4
        kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2,1,4, is_log_softmax>;
        break;
    case 3: // 8
        kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2,1,8, is_log_softmax>;
        break;
    case 4: // 16
        kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2,1,16, is_log_softmax>;
        break;
    case 5: // 32
        kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2,1,32, is_log_softmax>;
        break;
    case 6: // 64
        kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2,2,32, is_log_softmax>;
        break;
    case 7: // 128
        kernel = &softmax_warp_backward<input_t, output_t, acc_t, 2,4,32, is_log_softmax>;
        break;
    case 8: // 256
        kernel = &softmax_warp_backward<input_t, output_t, acc_t, 1,8,32, is_log_softmax>;
        break;
    case 9: // 512
        kernel = &softmax_warp_backward<input_t, output_t, acc_t, 1,16,32, is_log_softmax>;
        break;
    case 10: // 1024
        kernel = &softmax_warp_backward<input_t, output_t, acc_t, 1,32,32, is_log_softmax>;
        break;
    default:
        return false;
    }
    return true;
}

template<typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
bool dispatch_softmax_backward(output_t *grad_input, const input_t *grad, const input_t *output, int softmax_elements, int softmax_elements_stride, int batch_count)
{
    if (softmax_elements == 0) {
        return true;
    } else if (softmax_elements <= 1024) {
        // compute function index. there's a function for each power of two size up to 1024.
        int log2_elements = 0;
        while ((1 << log2_elements) < softmax_elements) ++log2_elements;

        softmax_backward_func<input_t, output_t> kernel;
        int warp_size, batches_per_warp;
        if (!warp_softmax_backward_kernel<input_t, output_t, acc_t, is_log_softmax>(log2_elements, warp_size, batches_per_warp, kernel)) {
            return false;
        }

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        // compute warps per block.
        int warps_per_block = (threads_per_block / warp_size);

        // compute launch size
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);

        // launch
        kernel<<<blocks, threads>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
        return true;
    }
    return false;
}

