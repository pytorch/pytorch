#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/gather_op.h"

namespace caffe2 {

template <typename TData, typename TIndex>
__global__ void GatherKernel(
    const TData* X,
    TData* Y,
    const TIndex * indices,
    const int N,
    const int block_size) {
  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    TIndex idx = indices[i];
    const TData* src_offset = X + idx * block_size;
    TData* dst_offset = Y + i * block_size;
    for (int j = threadIdx.x; j < block_size; j += blockDim.x) {
      dst_offset[j] = src_offset[j];
    }
  }
}

template <>
bool GatherOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
      this, OperatorBase::Input<Tensor>(INDICES, CUDA));
}

template <>
template <typename TIndex>
bool GatherOp<CUDAContext>::DoRunWithType() {
  return DispatchHelper<
      TensorTypes2<uint8_t, int, long, float, double, int32_t, int64_t, GenericTensorImplementation>,
      TIndex>::call(this, OperatorBase::Input<Tensor>(DATA, CUDA));
}

template <>
template <typename TIndex, typename TData>
bool GatherOp<CUDAContext>::DoRunWithType2() {
  auto& data = Input(DATA);
  auto& indices = Input(INDICES);
  auto* output = Output(0);

  CAFFE_ENFORCE_GE(data.ndim(), 1, "DATA should be at least 1-D");
  auto shape = indices.dims().vec();
  shape.insert(shape.end(), data.dims().begin() + 1, data.dims().end());
  output->Resize(shape);

  int block_size = data.size() / data.dim(0);
  auto block_bytesize = data.size_from_dim(1) * data.meta().itemsize();
  CAFFE_ENFORCE(
      block_bytesize == data.nbytes() / data.dim(0),
      "block_bytesize should be consistent with data dim");
  int N = indices.size();

  auto src_base = static_cast<const TData*>(data.raw_data());
  const TIndex* idxs = indices.template data<TIndex>();
  auto out = static_cast<TData*>(output->raw_mutable_data(data.meta()));

  // return early when the input is empty, since CUDA kernel will fail for
  // empty input.
  if (N <= 0) {
    return true;
  }

  GatherKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(src_base, out, idxs, N, block_size);
  return true;
}

REGISTER_CUDA_OPERATOR(Gather, GatherOp<CUDAContext>);
} // namespace caffe2
