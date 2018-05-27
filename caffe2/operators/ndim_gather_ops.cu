#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/ndim_gather_ops.h"

namespace caffe2 {

template <typename T_INDEX, typename TData>
__global__ void NdimGatherKernel(
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
bool NdimGatherOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
      this, OperatorBase::Input<TensorCUDA>(INDICES));
}

template <>
template <typename TInd>
bool NdimGatherOp<CUDAContext>::DoRunWithType() {
  auto& data = Input(DATA);
  auto& indices = Input(INDICES);
  auto* output = Output(0);

  CAFFE_ENFORCE_GE(data.ndim(), 1, "DATA should be at least 1-D");
  CAFFE_ENFORCE_GE(axis_, 0, "Axis should be non-negative");
  CAFFE_ENFORCE_LT(axis_, data.ndim(), "Axis out of range");

  vector<TIndex> shape;
  if (axis_ > 0) {
    shape.insert(
        shape.end(), data.dims().begin(), data.dims().begin() + axis_);
  }
  shape.insert(shape.end(), indices.dims().begin(), indices.dims().end());
  if (axis_ < data.ndim() - 1) {
    shape.insert(
        shape.end(), data.dims().begin() + axis_ + 1, data.dims().end());
  }
  output->Resize(shape);

  const int M = data.size_to_dim(axis_);
  const int block_size = data.size_from_dim(axis_ + 1);
  const int N = indices.size();
  const auto data_batch_size = data.size_from_dim(axis_);
  const auto gathered_batch_size = N * block_size;
  const TInd* idxs = indices.template data<TInd>();
  auto src_base = static_cast<const float*>(data.raw_data());
  auto out = static_cast<float*>(output->raw_mutable_data(data.meta()));

  // return early when the input is empty, since CUDA kernel will fail for
  // empty input.
  if (N <= 0) {
   return true;
  }

  NdimGatherKernel<<<
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
__global__ void NdimGatherGradientKernel(
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
bool NdimGatherGradientOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
      this, OperatorBase::Input<TensorCUDA>(INDICES));
}

template <>
template <typename TInd>
bool NdimGatherGradientOp<CUDAContext>::DoRunWithType() {
  return DispatchHelper<
      TensorTypes2<float, GenericTensorImplementation>,
      TInd>::call(this, OperatorBase::Input<TensorCUDA>(DATA));
}

template <>
template <typename TInd, typename TData>
bool NdimGatherGradientOp<CUDAContext>::DoRunWithType2() {
  auto& data = Input(DATA);
  auto& indices = Input(INDICES);
  auto& grad = Input(GRAD);
  auto* output = Output(0);

  CAFFE_ENFORCE_GE(data.ndim(), 1, "DATA should be at least 1-D");
  CAFFE_ENFORCE_GE(axis_, 0, "Axis should be non-negative");
  CAFFE_ENFORCE_LT(axis_, data.ndim(), "Axis out of range");
  for (int i = 0; i < axis_; i++) {
    CAFFE_ENFORCE_EQ(
        data.dim(i),
        grad.dim(i),
        "The ",
        i,
        "-th dimension should be the same");
  }

  output->ResizeLike(data);
  auto* out_data = output->template mutable_data<float>();
  math::Set<float, CUDAContext>(output->size(), 0, out_data, &context_);

  const auto* grad_data = grad.template data<float>();

  const int M = data.size_to_dim(axis_);
  const int block_size = data.size_from_dim(axis_ + 1);
  const int N = indices.size();
  const auto data_batch_size = data.size_from_dim(axis_);
  const auto gathered_batch_size = N * block_size;
  const TInd* idxs = indices.template data<TInd>();

  // return early when the input is empty, since CUDA kernel will fail for
  // empty input.
  if (N <= 0) {
   return true;
  }

  NdimGatherGradientKernel<<<
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
bool NdimGatherGradientOp<CUDAContext>::DoRunWithOtherType2() {
  CAFFE_THROW(
      "NdimGatherGradient is not implemented on tensor of type ",
      Input(DATA).meta().name(),
      "Consider adding it a type in the list DispatchHelper or implementing "
      "a generic version (which won't work for duplicated indices though)");
}

REGISTER_CUDA_OPERATOR(NdimGather, NdimGatherOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(NdimGatherGradient, NdimGatherGradientOp<CUDAContext>);

} // namespace caffe2
