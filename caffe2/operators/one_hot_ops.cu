#include <cub/block/block_reduce.cuh>
#include <cub/device/device_scan.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/one_hot_ops.h"

namespace caffe2 {

__global__ void OneHotOpKernel(
    const TIndex batch_size,
    const TIndex index_size,
    const TIndex* indices,
    float* output) {
  CUDA_1D_KERNEL_LOOP(i, batch_size) {
    output[i * index_size + indices[i]] = 1.;
  }
}

template <>
void OneHotOp<CUDAContext>::DoOneHotOp(
    TIndex batch_size,
    TIndex index_size,
    const Tensor<CUDAContext>& indices,
    Tensor<CUDAContext>* output) {
  float* output_ptr = output->mutable_data<float>();
  math::Set<float, CUDAContext>(output->size(), 0., output_ptr, &context_);
  OneHotOpKernel<<<
      CAFFE_GET_BLOCKS(batch_size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      batch_size, index_size, indices.data<TIndex>(), output_ptr);
}

template <typename T>
__device__ void BinarySearchInPrefixSum(
    const TIndex arr_size,
    const int queryIdx,
    const T* arr,
    TIndex* output_idx) {
  TIndex low = 0, high = arr_size - 1;
  TIndex mid = (high - low) / 2;
  while (low < high) {
    // queryIdx starts from zero, so comparing with arr[mid]-1
    if (queryIdx == arr[mid] - 1) {
      break;
    } else if (queryIdx < arr[mid] - 1) {
      high = mid;
    } else if (queryIdx > arr[mid] - 1) {
      low = mid + 1;
    }
    mid = low + (high - low) / 2;
  }
  output_idx[queryIdx] = mid;
}

// Find correspondense from len column to output col
template <typename T>
__global__ void Val2InputIdxMappingKernel(
    const TIndex len_size,
    const T* lens,
    TIndex* val2in_idx_mapping) {
  // value size is equal to last element of prefix sum
  const TIndex val_size = lens[len_size - 1];
  // For each index in value find corresponding length index
  CUDA_1D_KERNEL_LOOP(i, val_size) {
    BinarySearchInPrefixSum<T>(len_size, i, lens, val2in_idx_mapping);
  }
}

template <typename T>
__global__ void BatchOneHotOpKernel(
    const TIndex output_size,
    const TIndex len_size,
    const TIndex val_size,
    const TIndex* val2in_idx_mapping,
    const T* input,
    const T* vals,
    T* output) {
  CUDA_1D_KERNEL_LOOP(i, output_size) {
    TIndex val_idx = i % val_size; // deducing output column from i
    TIndex batch_idx = i / val_size; // get input row(sample# in batch) from i
    if (input[batch_idx * len_size + val2in_idx_mapping[val_idx]] ==
        vals[val_idx]) {
      output[i] = static_cast<T>(1);
    }
  }
}

template <typename T>
void inclusive_scan_wrapper(
    const T* in_data,
    const TIndex in_length,
    Tensor<CUDAContext>* prefix_sum_buffer,
    Tensor<CUDAContext>* prefix_sum_length_buffer,
    CUDAContext* context_) {
  // inclusive scan
  size_t temp_storage_bytes = 0;
  // Retrieve buffer size
  cub::DeviceScan::InclusiveSum(
      NULL,
      temp_storage_bytes,
      in_data,
      prefix_sum_length_buffer->mutable_data<T>(),
      in_length,
      context_->cuda_stream());
  // Allocate temporary storage
  auto buffer_size = temp_storage_bytes / sizeof(T);
  buffer_size += temp_storage_bytes % sizeof(T) != 0 ? 1 : 0;
  prefix_sum_buffer->Resize(buffer_size);
  void* d_temp_storage =
      static_cast<void*>(prefix_sum_buffer->mutable_data<T>());
  // Run inclusive prefix sum
  cub::DeviceScan::InclusiveSum(
      d_temp_storage,
      temp_storage_bytes,
      in_data,
      prefix_sum_length_buffer->mutable_data<T>(),
      in_length,
      context_->cuda_stream());
}

// The batch operation is divided in 3 steps
// 1. Computing the cumulative/Prefix sum for lengths
//    so that it could be used to compute mapping in step2 efficiently
// 2. Compute value to input index mapping
//    using binary search on prefix sum from step1
// 3. Compute output using mapping from step2
template <>
template <typename T>
bool BatchOneHotOp<CUDAContext>::DoBatchOneHotOp(
    const TIndex batch_size,
    const TIndex in_dim,
    const Tensor<CUDAContext>& input,
    const Tensor<CUDAContext>& lens,
    const Tensor<CUDAContext>& vals,
    Tensor<CUDAContext>* output) {
  const auto* lens_data = lens.template data<int32_t>();
  const TIndex out_dim = vals.size();
  int32_t h_lens_data[in_dim];
  cudaMemcpy(
      h_lens_data,
      lens_data,
      (in_dim) * sizeof(int32_t),
      cudaMemcpyDeviceToHost);
  TIndex len_sum = 0;
  for (TIndex i = 0; i < in_dim; i++) {
    CAFFE_ENFORCE_GE(h_lens_data[i], 0);
    len_sum += h_lens_data[i];
  }
  // sum of lens should be equal to out_dim(vals.size())
  CAFFE_ENFORCE_EQ(out_dim, len_sum);

  // Compute cumulative/prefix/inclusive sum
  Tensor<CUDAContext> inclusive_scan_buffer;
  Tensor<CUDAContext> inclusive_scan_length_buffer;
  inclusive_scan_length_buffer.ResizeLike(lens);
  inclusive_scan_wrapper<int32_t>(
      lens_data,
      in_dim,
      &inclusive_scan_buffer,
      &inclusive_scan_length_buffer,
      &context_);
  const auto* prefix_sum_len_data =
      inclusive_scan_length_buffer.template data<int32_t>();

  // Compute value index to Input index mapping
  Tensor<CUDAContext> val2in_idx_mapping;
  val2in_idx_mapping.ResizeLike(vals);
  TIndex* val2in_idx_mapping_data =
      val2in_idx_mapping.template mutable_data<TIndex>();

  Val2InputIdxMappingKernel<int32_t>
      <<<CAFFE_GET_BLOCKS(out_dim),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          in_dim, prefix_sum_len_data, val2in_idx_mapping_data);
  // Update output values using mapping
  T* output_data = output->template mutable_data<T>();
  cudaMemset(output_data, 0, batch_size * out_dim * sizeof(T));
  BatchOneHotOpKernel<<<
      CAFFE_GET_BLOCKS(batch_size * out_dim),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      (batch_size * out_dim),
      in_dim,
      out_dim,
      val2in_idx_mapping_data,
      input.template data<T>(),
      vals.template data<T>(),
      output_data);
  return true;
}

REGISTER_CUDA_OPERATOR(OneHot, OneHotOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(BatchOneHot, BatchOneHotOp<CUDAContext>);
} // namespace
