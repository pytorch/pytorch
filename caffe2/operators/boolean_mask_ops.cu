#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/boolean_mask_ops.h"

#include <cub/cub.cuh>

namespace caffe2 {

namespace {
template <typename T>
__global__ void BooleanMaskCopyKernel(
    const TIndex numOfOutput,
    const TIndex numBytes,
    const TIndex* indices,
    const T* src,
    T* dest) {
  for (TIndex i = blockIdx.x; i < numOfOutput; i += gridDim.x) {
    const auto srcBase = indices[i] * numBytes;
    const auto destBase = i * numBytes;
    for (TIndex j = threadIdx.x; j < numBytes; j += blockDim.x) {
      dest[destBase + j] = src[srcBase + j];
    }
  }
}
}

template <>
class BooleanMaskOp<CUDAContext> final : public Operator<CUDAContext> {
 public:
  BooleanMaskOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    const auto& src = Input(0);
    const auto& mask = Input(1);
    auto* dest = Output(0);

    CAFFE_ENFORCE(src.ndim() >= 1);
    CAFFE_ENFORCE_EQ(mask.ndim(), 1);
    CAFFE_ENFORCE(src.dims()[0] == mask.dims()[0]);

    const auto* maskData = mask.data<bool>();
    const auto outerSize = mask.dims()[0];
    indices_.Resize(outerSize);
    auto* indicesData = indices_.mutable_data<TIndex>();

    size_t numBytes = 0;
    cub::CountingInputIterator<int> itr(0);
    cub::DeviceSelect::Flagged(
        nullptr,
        numBytes,
        itr,
        maskData,
        indicesData,
        static_cast<TIndex*>(nullptr),
        outerSize,
        context_.cuda_stream());

    auto numTIndex =
        static_cast<TIndex>((numBytes + sizeof(TIndex) - 1) / sizeof(TIndex));
    // allocate one more TIndex at the end of scratch for storing numOfOutput
    scratch_.Resize(numTIndex + 1);
    auto* scratchData = scratch_.mutable_data<TIndex>();
    auto* numOfOutputData = scratchData + numTIndex;

    cub::DeviceSelect::Flagged(
        static_cast<void*>(scratchData),
        numBytes,
        itr,
        maskData,
        indicesData,
        numOfOutputData,
        outerSize,
        context_.cuda_stream());

    // Copy numOfOutput from gpu to cpu
    TIndex numOfOutput;
    context_.Copy<TIndex, CUDAContext, CPUContext>(
        1, numOfOutputData, &numOfOutput);

    indices_.Resize(numOfOutput);
    std::vector<TIndex> dims = src.dims();
    dims[0] = numOfOutput;
    dest->Resize(dims);
    auto* destData = (char*)dest->raw_mutable_data(src.meta());
    const auto* srcData = (char*)src.raw_data();
    if (OutputSize() == 2) {
      auto* indicesOut = Output(1);
      indicesOut->Resize(numOfOutput);
      indicesOut->mutable_data<TIndex>();
    }

    if (numOfOutput > 0) {
      BooleanMaskCopyKernel<<<
          min(numOfOutput, static_cast<TIndex>(CAFFE_MAXIMUM_NUM_BLOCKS)),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          numOfOutput,
          src.size_from_dim(1) * src.meta().itemsize(),
          indicesData,
          srcData,
          destData);

      if (OutputSize() == 2) {
        Output(1)->CopyFrom(indices_, &context_);
      }
    }

    return true;
  }

 private:
  Tensor<CUDAContext> indices_;
  Tensor<CUDAContext> scratch_;
};

REGISTER_CUDA_OPERATOR(BooleanMask, BooleanMaskOp<CUDAContext>);

} // caffe2
