#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/cast_op.h"

namespace caffe2 {

namespace {
template <typename DstType, typename SrcType>
__global__ void CastKernel(const int N, const SrcType* X, DstType* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = static_cast<DstType>(X[i]);
  }
}
}  // namespace

template <>
template <typename DstType, typename SrcType>
bool CastOp<CUDAContext>::DoRunWithType() {
  auto& input = Input(0);
  auto* output = Output(0);
  output->ResizeLike(input);
  const auto* data = input.template data<SrcType>();
  auto* out = output->template mutable_data<DstType>();
  DCHECK(input.size() < INT_MAX);
  int N = input.size();
  CastKernel<DstType, SrcType><<<
      CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS,
      0, context_.cuda_stream()>>>(N, data, out);
  return true;
}

REGISTER_CUDA_OPERATOR(Cast, CastOp<CUDAContext>);

}  // namespace caffe2
