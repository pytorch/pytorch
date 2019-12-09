#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/boolean_unmask_ops.h"

namespace caffe2 {

namespace {

__global__ void ComputeIndicesKernel(
    const int numMasks,
    const int maskSize,
    int* indices,
    bool* const masks[]) {
  CUDA_1D_KERNEL_LOOP(i, maskSize) {
    for (int j = 0; j < numMasks; ++j) {
      if (masks[j][i]) {
        indices[i] = j;
        return;
      }
    }
    CUDA_KERNEL_ASSERT(false);
  }
}

__global__ void FillValuesKernel(
    const int numMasks,
    const int maskSize,
    const size_t itemSize,
    const int* indices,
    char* const values[],
    int* valueSizes,
    char* dest) {
  CUDA_1D_KERNEL_LOOP(j, numMasks) {
    int k = 0;
    for (int i = 0; i < maskSize; ++i) {
      if (indices[i] == j) {
        for (int h = 0; h < itemSize; ++h) {
          dest[i * itemSize + h] = values[j][k * itemSize + h];
        }
        ++k;
      }
    }
    CUDA_KERNEL_ASSERT(valueSizes[j] == k);
  }
}

} // namespace

template <>
class BooleanUnmaskOp<CUDAContext> final : public Operator<CUDAContext> {
 public:
  BooleanUnmaskOp(const OperatorDef& def, Workspace* ws)
      : Operator<CUDAContext>(def, ws) {}

  bool RunOnDevice() override {
    int maskSize = Input(0).numel();
    int numMasks = InputSize() / 2;
    const auto& meta = Input(1).meta();

    auto* out = Output(0);
    out->Resize(maskSize);
    auto* dest = (char*)out->raw_mutable_data(meta);

    ReinitializeTensor(&hostMasks_, {numMasks}, at::dtype<bool*>().device(CPU));
    auto* hostMasksData = hostMasks_.mutable_data<bool*>();
    ReinitializeTensor(
        &hostValues_, {numMasks}, at::dtype<char*>().device(CPU));
    auto* hostValuesData = hostValues_.mutable_data<char*>();
    ReinitializeTensor(
        &hostValueSizes_, {numMasks}, at::dtype<int>().device(CPU));
    auto* hostValueSizesData = hostValueSizes_.mutable_data<int>();
    for (int i = 0; i < numMasks; ++i) {
      auto& mask = Input(i * 2);
      CAFFE_ENFORCE_EQ(mask.dim(), 1);
      CAFFE_ENFORCE_EQ(mask.numel(), maskSize);
      hostMasksData[i] = const_cast<bool*>(mask.data<bool>());

      const auto& value = Input(i * 2 + 1);
      CAFFE_ENFORCE_EQ(value.dim(), 1);
      hostValuesData[i] = (char*)value.raw_data();
      hostValueSizesData[i] = value.numel();
    }
    masks_.CopyFrom(hostMasks_);
    values_.CopyFrom(hostValues_);
    valueSizes_.CopyFrom(hostValueSizes_);

    ReinitializeTensor(&indices_, {maskSize}, at::dtype<int>().device(CUDA));
    auto* indicesData = indices_.mutable_data<int>();

    ComputeIndicesKernel<<<
        min(maskSize, CAFFE_MAXIMUM_NUM_BLOCKS),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        numMasks, maskSize, indicesData, masks_.data<bool*>());

    auto* valueSizesData = valueSizes_.mutable_data<int>();
    FillValuesKernel<<<
        min(numMasks, CAFFE_MAXIMUM_NUM_BLOCKS),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        numMasks,
        maskSize,
        meta.itemsize(),
        indicesData,
        values_.data<char*>(),
        valueSizesData,
        dest);

    return true;
  }

 private:
  Tensor indices_;
  Tensor masks_{CUDA};
  Tensor values_{CUDA};
  Tensor valueSizes_{CUDA};

  Tensor hostMasks_;
  Tensor hostValues_;
  Tensor hostValueSizes_;
};

REGISTER_CUDA_OPERATOR(BooleanUnmask, BooleanUnmaskOp<CUDAContext>);

} // caffe2
