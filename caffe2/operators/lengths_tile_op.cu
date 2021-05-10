#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/lengths_tile_op.h"

namespace caffe2 {

template <typename T>
__global__ void lengthsTileKernel(
    int numElements,
    int rowSize,
    const T* input,
    T* output,
    const int32_t* inputRowOffsets) {
  CUDA_1D_KERNEL_LOOP(i, numElements) {
    auto outputRowIndex = i / rowSize;
    auto inputBlockOffset = inputRowOffsets[outputRowIndex];
    auto indexInRow = i - outputRowIndex * rowSize;
    output[i] = input[inputBlockOffset + indexInRow];
  }
}

template <>
bool LengthsTileOp<CUDAContext>::RunOnDevice() {
  auto& data = Input(DATA);
  auto& lengths = Input(LENGTHS);


  CAFFE_ENFORCE_EQ(lengths.dim(), 1, "LENGTHS must be 1-D");
  CAFFE_ENFORCE_GE(data.dim(), 1, "DATA should be at least 1-D");
  CAFFE_ENFORCE_EQ(lengths.numel(), data.dim(0));

  lengths_host_.CopyFrom(lengths); // sync copy
  auto lengths_size = lengths_host_.numel();
  auto* lengths_data = lengths_host_.data<int32_t>();

  int32_t total_length = 0;
  CPUContext cpuContext;
  math::Sum<int32_t, CPUContext>(
      lengths_size, lengths_data, &total_length, &cpuContext);

  auto shape = data.sizes().vec();
  shape[0] = total_length;
  auto* output = Output(0, shape, at::dtype<float>());

  auto numElementsPerRow = data.size_from_dim(1);
  auto numElements = total_length * numElementsPerRow;
  auto numBlocks = CAFFE_GET_BLOCKS(numElements);

  ReinitializeTensor(&rowMappingHost_, {total_length}, at::dtype<int32_t>().device(CPU));
  ReinitializeTensor(&rowMappingDevice_, {total_length}, at::dtype<int32_t>().device(CPU));
  auto* rowOffsets = rowMappingHost_.mutable_data<int32_t>();
  int32_t outputRow = 0;
  for (int64_t i = 0; i < lengths_size; i++) {
    auto length = lengths_data[i];
    for (int32_t j = 0; j < length; j++) {
      rowOffsets[outputRow++] = i * numElementsPerRow;
    }
  }

  context_.CopyFromCPU<int32_t>(
      total_length,
      rowMappingHost_.data<int32_t>(),
      rowMappingDevice_.mutable_data<int32_t>());
  context_.FinishDeviceComputation();

  if (data.template IsType<float>()) {
    lengthsTileKernel<<<
        numBlocks,
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        numElements,
        numElementsPerRow,
        data.data<float>(),
        output->mutable_data<float>(),
        rowMappingDevice_.data<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if (data.template IsType<int>()) {
    lengthsTileKernel<<<
        numBlocks,
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        numElements,
        numElementsPerRow,
        data.data<int>(),
        output->mutable_data<int>(),
        rowMappingDevice_.data<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if (data.template IsType<int64_t>()) {
    lengthsTileKernel<<<
        numBlocks,
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        numElements,
        numElementsPerRow,
        data.data<int64_t>(),
        output->mutable_data<int64_t>(),
        rowMappingDevice_.data<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    CAFFE_THROW(
        "LengthsTile operator only supports 32-bit float, int and int64_t"
        " types but input was of type ",
        data.meta().name());
  }
  return true;
}

REGISTER_CUDA_OPERATOR(LengthsTile, LengthsTileOp<CUDAContext>);

} // namespace caffe2
