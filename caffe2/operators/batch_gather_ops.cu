#include <fstream>
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/batch_gather_ops.h"

namespace caffe2 {

template <typename T_INDEX, typename TData>
__global__ void BatchGatherKernel(
    const TData* src_base,
    TData* out,
    const T_INDEX* indices,
    const int M,
    const int N,
    const int data_batch_size,
    const int gathered_batch_size,
    const int block_size) {
  const int begin_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_items = M * N * block_size;
  for (int s = begin_idx; s < num_items; s += blockDim.x * gridDim.x) {
    const int k = s % block_size;
    const int j = s / block_size % N;
    const int i = s / block_size / N;
    const T_INDEX idx = indices[j];
    const float* src_offset = src_base + i * data_batch_size + idx * block_size;
    float* dst_offset = out + i * gathered_batch_size + j * block_size;
    dst_offset[k] = src_offset[k];
  }
}

template <>
bool BatchGatherOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
      this, OperatorBase::Input<TensorCUDA>(INDICES));
}

template <>
template <typename TInd>
bool BatchGatherOp<CUDAContext>::DoRunWithType() {
  auto& data = Input(DATA);
  auto& indices = Input(INDICES);
  auto* output = Output(0);

  vector<TIndex> shape;
  shape.push_back(data.dim(0));
  shape.insert(shape.end(), indices.dims().begin(), indices.dims().end());
  shape.insert(shape.end(), data.dims().begin() + 2, data.dims().end());
  output->Resize(shape);

  const int block_size = data.size_from_dim(2);
  const int N = indices.size();
  const auto data_batch_size = data.size_from_dim(1);
  const auto gathered_batch_size = N * data.size_from_dim(2);
  const TInd* idxs = indices.template data<TInd>();
  auto src_base = static_cast<const float*>(data.raw_data());
  auto out = static_cast<float*>(output->raw_mutable_data(data.meta()));
  const int M = data.dim32(0);

  BatchGatherKernel<<<
      std::min(M, CAFFE_MAXIMUM_NUM_BLOCKS),
      std::min(N * block_size, CAFFE_CUDA_NUM_THREADS),
      0,
      context_.cuda_stream()>>>(
      src_base,
      out,
      idxs,
      M,
      N,
      data_batch_size,
      gathered_batch_size,
      block_size);
  return true;
}

template <typename T_INDEX, typename TData>
__global__ void BatchGatherGradientKernel(
    const TData* grad_data,
    TData* out,
    const T_INDEX* indices,
    const int M,
    const int N,
    const int data_batch_size,
    const int gathered_batch_size,
    const int block_size) {
  int begin_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_items = M * N * block_size;
  for (int s = begin_idx; s < num_items; s += blockDim.x * gridDim.x) {
    const int k = s % block_size;
    const int j = s / block_size % N;
    const int i = s / block_size / N;
    const T_INDEX idx = indices[j];
    const float* src_offset =
        grad_data + i * gathered_batch_size + j * block_size;
    float* dst_offset = out + i * data_batch_size + idx * block_size;
    atomicAdd(dst_offset + k, src_offset[k]);
  }
}

template <>
bool BatchGatherGradientOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
      this, OperatorBase::Input<TensorCUDA>(INDICES));
}

template <>
template <typename TInd>
bool BatchGatherGradientOp<CUDAContext>::DoRunWithType() {
  return DispatchHelper<
      TensorTypes2<float, GenericTensorImplementation>,
      TInd>::call(this, OperatorBase::Input<TensorCUDA>(DATA));
}

template <>
template <typename TInd, typename TData>
bool BatchGatherGradientOp<CUDAContext>::DoRunWithType2() {
  auto& data = Input(DATA);
  auto& indices = Input(INDICES);
  auto& grad = Input(GRAD);
  auto* output = Output(0);

  CAFFE_ENFORCE_EQ(data.dim(0), grad.dim(0), "batch sizes should be the same");

  output->ResizeLike(data);
  auto* out_data = output->template mutable_data<float>();
  math::Set<float, CUDAContext>(output->size(), 0, out_data, &context_);

  const auto* grad_data = grad.template data<float>();

  const int M = grad.dim32(0);
  const int block_size = data.size_from_dim(2);
  const int N = indices.size();
  const auto data_batch_size = data.size_from_dim(1);
  const auto gathered_batch_size = N * data.size_from_dim(2);
  const TInd* idxs = indices.template data<TInd>();

  BatchGatherGradientKernel<<<
      std::min(M, CAFFE_MAXIMUM_NUM_BLOCKS),
      std::min(N * block_size, CAFFE_CUDA_NUM_THREADS),
      0,
      context_.cuda_stream()>>>(
      grad_data,
      out_data,
      idxs,
      M,
      N,
      data_batch_size,
      gathered_batch_size,
      block_size);

  return true;
}

template <>
template <typename TInd>
bool BatchGatherGradientOp<CUDAContext>::DoRunWithOtherType2() {
  CAFFE_THROW(
      "BatchGatherGradient is not implemented on tensor of type ",
      Input(DATA).meta().name(),
      "Consider adding it a type in the list DispatchHelper or implementing "
      "a generic version (which won't work for duplicated indices though)");
}

REGISTER_CUDA_OPERATOR(BatchGather, BatchGatherOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(BatchGatherGradient, BatchGatherGradientOp<CUDAContext>);

} // namespace caffe2
