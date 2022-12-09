#include <algorithm>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/scale_blobs_op.h"

namespace caffe2 {

template <typename T>
__global__ void ScaleBlobsCUDAKernel(
    const float scale,
    const int numBlobs,
    const int* sizeArr,
    T** X,
    T** Y) {
  for (size_t i = 0; i < numBlobs; ++i) {
    CUDA_1D_KERNEL_LOOP(j, sizeArr[i]) {
      Y[i][j] = X[i][j] * scale;
    }
  }
}

template <typename T>
__global__ void ScaleBlobsCUDAKernelManyTensors(
    const float scale,
    const int* sizeArr,
    T** X,
    T** Y) {
  for (size_t i = threadIdx.x; i < sizeArr[blockIdx.x]; i += blockDim.x) {
    Y[blockIdx.x][i] = X[blockIdx.x][i] * scale;
  }
}

template <>
template <typename T>
bool ScaleBlobsOp<CUDAContext>::DoRunWithType() {
  const int numBlobs = InputSize();

  ReinitializeTensor(&hostBlobSizes_, {numBlobs}, at::dtype<int>().device(CPU));
  int* hostBlobSizesData = hostBlobSizes_.mutable_data<int>();

  ReinitializeTensor(&hostInputs_, {numBlobs}, at::dtype<T*>().device(CPU));
  T** hostInputsData = hostInputs_.mutable_data<T*>();

  ReinitializeTensor(&hostOutputs_, {numBlobs}, at::dtype<T*>().device(CPU));
  T** hostOutputsData = hostOutputs_.mutable_data<T*>();

  int totalSize = 0;
  int maxSize = 0;
  for (int i = 0; i < numBlobs; ++i) {
    hostBlobSizesData[i] = Input(i).numel();
    totalSize += hostBlobSizesData[i];
    maxSize = std::max(maxSize, hostBlobSizesData[i]);
    hostInputsData[i] = Input(i).template data<T>();
    hostOutputsData[i] = Output(i)->template mutable_data<T>();
  }

  ReinitializeTensor(&inputs_, {numBlobs}, at::dtype<T*>().device(CUDA));
  ReinitializeTensor(&outputs_, {numBlobs}, at::dtype<T*>().device(CUDA));
  ReinitializeTensor(&blobSizes_, {numBlobs}, at::dtype<T*>().device(CUDA));

  blobSizes_.CopyFrom(hostBlobSizes_);
  inputs_.CopyFrom(hostInputs_);
  outputs_.CopyFrom(hostOutputs_);

  // Select which kernel to launch based on the length of the tensors
  // The first one performs better when there are many tensors of short length
  // The second one is better when there are small number of long tensors
  if (numBlobs > CAFFE_GET_BLOCKS(maxSize)) {
    // Note: number of blocks has to be equal to the numBlobs
    ScaleBlobsCUDAKernelManyTensors<T>
        <<<numBlobs, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
            scale_,
            blobSizes_.data<int>(),
            inputs_.mutable_data<T*>(),
            outputs_.mutable_data<T*>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    ScaleBlobsCUDAKernel<T>
        <<<CAFFE_GET_BLOCKS(maxSize),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            scale_,
            numBlobs,
            blobSizes_.data<int>(),
            inputs_.mutable_data<T*>(),
            outputs_.mutable_data<T*>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return true;
}

template <>
bool ScaleBlobsOp<CUDAContext>::RunOnDevice() {
  for (int i = 0; i < InputSize(); ++i) {
    auto& input = this->template Input<Tensor>(i, CUDA);
    auto* output = this->template Output<Tensor>(i, CUDA);
    output->ResizeLike(input);
  }
  return DispatchHelper<TensorTypes<at::Half, float>>::call(this, Input(0));
}

REGISTER_CUDA_OPERATOR(ScaleBlobs, ScaleBlobsOp<CUDAContext>);

/*
 * Implementation of a different version of the kernel
 * This balances the work per thread and could be useful
 * when there is a high imbalance between tensors
 * However the memory requirement is very high so it does
 * not perform well for common scenarios
 *
 *
 * Additional storage for the start pointers is required
 * for ScaleBlobsCUDAKernelBalanced setup
 *
    int threadsPerBlock = CAFFE_CUDA_NUM_THREADS;
    int coorArrSize = 2 * ((totalSize - 1) / threadsPerBlock + 1);
    int startCoorArr[coorArrSize];
    int* dStartCoorArr;

    int j = 0, cur = 0, elemsLeftInRow = 0;
    for (int i = 0; i < numBlobs; ++i) {
      if (i == 0) {
        startCoorArr[cur++] = i;
        startCoorArr[cur++] = j;
        elemsLeftInRow = 0;
      }
      while (j < sizeArr[i]) {
        j += threadsPerBlock - elemsLeftInRow;
        if (j < sizeArr[i]) {
          startCoorArr[cur++] = i;
          startCoorArr[cur++] = j;
          elemsLeftInRow = 0;
        } else {
          elemsLeftInRow = sizeArr[i] - j + threadsPerBlock;
          j = 0;
          break;
        }
      }
    }
    C10_CUDA_CHECK(cudaMalloc(&dStartCoorArr, sizeof(int) * coorArrSize));
    C10_CUDA_CHECK(cudaMemcpy(dStartCoorArr, startCoorArr, sizeof(int) * coorArrSize,
      cudaMemcpyHostToDevice));

  // ScaleBlobsCUDAKernelBalanced kernel launch
  ScaleBlobsCUDAKernelBalanced<T>
   <<<(totalSize-1)/CAFFE_CUDA_NUM_THREADS+1, CAFFE_CUDA_NUM_THREADS, 0,
   context_.cuda_stream()>>>(
     scale_, numBlobs, coorArrSize, dStartCoorArr, dSizeArr, dInputArr,
     dOutputArr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  C10_CUDA_CHECK(cudaFree(dStartCoorArr));
*/

template <typename T>
__global__ void ScaleBlobsCUDAKernelBalanced(
    const float scale,
    const int numBlobs,
    const int coorArrSize,
    const int* coorArr,
    const int* sizeArr,
    T** X,
    T** Y) {
  int i = coorArr[2 * blockIdx.x + 1] + threadIdx.x;
  int curTen = coorArr[2 * blockIdx.x];
  while (curTen < numBlobs && i >= sizeArr[curTen]) {
    i -= sizeArr[curTen++];
  }
  if (curTen < numBlobs) {
    Y[curTen][i] = X[curTen][i] * scale;
  }
}

} // namespace caffe2
