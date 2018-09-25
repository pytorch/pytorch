#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/gather_op.h"

namespace caffe2 {

template <typename T_INDEX>
__global__ void GatherKernel(
    const float* src_base,
    float* out,
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
bool GatherOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
      this, OperatorBase::Input<Tensor>(INDICES, CUDA));
}

template <>
template <typename TInd>
bool GatherOp<CUDAContext>::DoRunWithType() {
  auto& data = Input(DATA);
  auto& indices = Input(INDICES);
  auto axis = axis_;
  auto* output = Output(0);

  CAFFE_ENFORCE_GE(data.ndim(), 1, "DATA should be at least 1-D");
  CAFFE_ENFORCE_GE(axis_, 0, "Axis should be non-negative");
  CAFFE_ENFORCE_LT(axis_, data.ndim(), "Axis out of range");

  vector<TIndex> shape;
  shape.insert(shape.end(), data.dims().begin(), data.dims().begin() + axis);
  shape.insert(shape.end(), indices.dims().begin(), indices.dims().end());
  shape.insert(shape.end(), data.dims().begin() + axis + 1, data.dims().end());
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

  GatherKernel<<<
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

REGISTER_CUDA_OPERATOR(Gather, GatherOp<CUDAContext>);
} // namespace caffe2
