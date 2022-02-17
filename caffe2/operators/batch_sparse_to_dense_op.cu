#include "caffe2/operators/batch_sparse_to_dense_op.h"

#include "caffe2/utils/cub_namespace.cuh"
#include <cub/device/device_scan.cuh>

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

template <typename TLen>
void array_prefix_sum_inclusive(
    const TLen* dev_array,
    const int num_items,
    Tensor& prefix_buffer,
    Tensor& prefix_sum,
    CUDAContext& context) {
  // Retrieve buffer size
  size_t temp_storage_bytes = 0;
  prefix_sum.Resize(num_items);
  cub::DeviceScan::InclusiveSum(
      nullptr,
      temp_storage_bytes,
      dev_array,
      prefix_sum.mutable_data<TLen>(),
      num_items,
      context.cuda_stream());

  // Allocate temporary storage
  auto buffer_size = (temp_storage_bytes + sizeof(TLen)) / sizeof(TLen);
  prefix_buffer.Resize(buffer_size);
  void* dev_temp_storage =
      static_cast<void*>(prefix_buffer.mutable_data<TLen>());

  // Inclusive sum
  cub::DeviceScan::InclusiveSum(
      dev_temp_storage,
      temp_storage_bytes,
      dev_array,
      prefix_sum.mutable_data<TLen>(),
      num_items,
      context.cuda_stream());
}

template <typename TLen, typename TInd>
__global__ void FillInDenseValuesKernel(
    const int64_t batch_size,
    const int64_t dense_last_dim,
    const TInd* indices_data,
    const float* values_data,
    const TLen* L_cum_sum_data,
    float* output_data) {
  CUDA_1D_KERNEL_LOOP(idx, batch_size) {
    int offset_start = idx == 0 ? 0 : L_cum_sum_data[idx - 1];
    int offset_end = L_cum_sum_data[idx];

    for (int q = offset_start; q < offset_end; q++) {
      int indice = indices_data[q];
      float val = values_data[q];
      output_data[idx * dense_last_dim + indice] = val;
    }
  }
}

template <typename TLen, typename TInd>
__global__ void FillInSparseValuesKernel(
    const int64_t batch_size,
    const int64_t dense_last_dim,
    const TInd* indices_data,
    const float* dense_data,
    const TLen* L_cum_sum_data,
    float* output_data) {
  CUDA_1D_KERNEL_LOOP(idx, batch_size) {
    int offset_start = idx == 0 ? 0 : L_cum_sum_data[idx - 1];
    int offset_end = L_cum_sum_data[idx];

    for (int q = offset_start; q < offset_end; q++) {
      int indice = indices_data[q];
      output_data[q] = dense_data[idx * dense_last_dim + indice];
    }
  }
}

} // namespace

template <>
template <typename TLen, typename TInd>
void BatchSparseToDenseOp<float, CUDAContext>::FillInDenseValues(
    const int64_t batch_size,
    const int64_t indice_lengths,
    const TLen* lengths_data,
    const TInd* indices_data,
    const float* values_data,
    float* output_data,
    CUDAContext* context) {
  // calculate the prefix sum of the length array
  array_prefix_sum_inclusive<TLen>(
      lengths_data, batch_size, len_prefix_tmp_, len_prefix_sum_, context_);

  // launch the gpu kernel for to fill in dense values
  const int64_t min_size = 1;
  FillInDenseValuesKernel<TLen, TInd><<<
      CAFFE_GET_BLOCKS(std::max(batch_size, min_size)),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(
      batch_size,
      dense_last_dim_,
      indices_data,
      values_data,
      len_prefix_sum_.data<TLen>(),
      output_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <>
template <typename TLen, typename TInd>
void BatchDenseToSparseOp<float, CUDAContext>::FillInSparseValues(
    const int64_t batch_size,
    const int64_t indice_lengths,
    const TLen* lengths_data,
    const TInd* indices_data,
    const float* dense_data,
    float* output_data,
    CUDAContext* context) {
  // calculate the prefix sum of the length array
  array_prefix_sum_inclusive<TLen>(
      lengths_data, batch_size, len_prefix_tmp_, len_prefix_sum_, context_);

  // launch the gpu kernel for to fill in sparse values
  const int64_t min_size = 1;
  FillInSparseValuesKernel<TLen, TInd><<<
      CAFFE_GET_BLOCKS(std::max(batch_size, min_size)),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(
      batch_size,
      dense_last_dim_,
      indices_data,
      dense_data,
      len_prefix_sum_.data<TLen>(),
      output_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

REGISTER_CUDA_OPERATOR(
    BatchSparseToDense,
    BatchSparseToDenseOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(
    BatchDenseToSparse,
    BatchDenseToSparseOp<float, CUDAContext>);
} // namespace caffe2
