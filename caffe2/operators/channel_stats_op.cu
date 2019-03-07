#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/channel_stats_op.h"

namespace caffe2 {

namespace {

// based on "Optimizing Parallel Reduction in CUDA" by Mark Harris

// note - volatile keyword is needed to allow doing a warp reduction without
// synchronization on recent architectures
template <unsigned int blockSize>
__device__ void warpReduce(volatile float* sdata, unsigned int tid) {
  // note - the if statements are "free" as they are resolved at compile time
  if (blockSize >= 64)
    sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32)
    sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16)
    sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8)
    sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4)
    sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2)
    sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void ChannelStatsBlockKernel(
    int N,
    int C,
    int valsPerChannel,
    const float* inputData,
    float* sums,
    float* sumsq) {
  __shared__ float sumData[blockSize];
  __shared__ float sumSqData[blockSize];

  auto tid = threadIdx.x;
  auto numBlocksPerChannel = (valsPerChannel + blockSize - 1) / blockSize;
  auto localBlockIndex = blockIdx.x % numBlocksPerChannel;
  auto inputIndex = (blockIdx.x / numBlocksPerChannel) * valsPerChannel +
      localBlockIndex * blockSize + tid;

  sumData[tid] = 0;
  sumSqData[tid] = 0;

  if (localBlockIndex * blockSize + tid < valsPerChannel) {
    sumData[tid] += inputData[inputIndex];
    sumSqData[tid] += inputData[inputIndex] * inputData[inputIndex];
  }

  __syncthreads();
  if (blockSize >= 512) {
    if (tid < 256) {
      sumData[tid] += sumData[tid + 256];
      sumSqData[tid] += sumSqData[tid + 256];
    }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) {
      sumData[tid] += sumData[tid + 128];
      sumSqData[tid] += sumSqData[tid + 128];
    }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) {
      sumData[tid] += sumData[tid + 64];
      sumSqData[tid] += sumSqData[tid + 64];
    }
    __syncthreads();
  }

  if (tid < 32) {
    warpReduce<blockSize>(sumData, tid);
    warpReduce<blockSize>(sumSqData, tid);
  }

  // output block data sorted by C to simplify second reduction
  if (tid == 0) {
    auto n = blockIdx.x / numBlocksPerChannel / C;
    auto c = (blockIdx.x / numBlocksPerChannel) % C;
    auto outputIndex = (c * N + n) * numBlocksPerChannel + localBlockIndex;
    sums[outputIndex] = sumData[0];
    sumsq[outputIndex] = sumSqData[0];
  }
}

template <unsigned int blockSize>
__global__ void ChannelStatsFinalSumsKernel(
    int N,
    int C,
    int numSumsPerChannel,
    const float* sumsScratch,
    const float* sumsqScratch,
    float* channelSums,
    float* channelSumsq) {
  __shared__ float sumData[blockSize];
  __shared__ float sumSqData[blockSize];

  auto tid = threadIdx.x;
  auto inputIndex = blockIdx.x * N * numSumsPerChannel + tid;
  sumData[tid] = 0;
  sumSqData[tid] = 0;
  for (auto i = inputIndex; i < (blockIdx.x + 1) * N * numSumsPerChannel;
       i += blockSize) {
    sumData[tid] += sumsScratch[i];
    sumSqData[tid] += sumsqScratch[i];
  }
  __syncthreads();
  if (blockSize >= 512) {
    if (tid < 256) {
      sumData[tid] += sumData[tid + 256];
      sumSqData[tid] += sumSqData[tid + 256];
    }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) {
      sumData[tid] += sumData[tid + 128];
      sumSqData[tid] += sumSqData[tid + 128];
    }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) {
      sumData[tid] += sumData[tid + 64];
      sumSqData[tid] += sumSqData[tid + 64];
    }
    __syncthreads();
  }
  if (tid < 32) {
    warpReduce<blockSize>(sumData, tid);
    warpReduce<blockSize>(sumSqData, tid);
  }

  if (tid == 0) {
    channelSums[blockIdx.x] = sumData[0];
    channelSumsq[blockIdx.x] = sumSqData[0];
  }
}
} // namespace

template <>
bool ChannelStatsOp<CUDAContext>::RunOnDevice() {
  const auto& X = Input(INPUT);
  CAFFE_ENFORCE(X.dim() >= 3 && X.dim() <= 5);
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim() > 3 ? X.dim32(3) : 1;
  const int D = X.dim() > 4 ? X.dim32(4) : 1;

  const auto X_arr = X.data<float>();
  const auto valsPerChannel = H * W * D;

  const auto numBlocksPerChannel = CAFFE_GET_BLOCKS(valsPerChannel);
  const auto numBlocksTotal = numBlocksPerChannel * N * C;

  ReinitializeTensor(
      &sumScratch_, {numBlocksTotal}, at::dtype<float>().device(CUDA));
  ReinitializeTensor(
      &sumsqScratch_, {numBlocksTotal}, at::dtype<float>().device(CUDA));

  auto sum = Output(SUM, {C}, at::dtype<float>());
  auto sumsq = Output(SUMSQ, {C}, at::dtype<float>());

  ChannelStatsBlockKernel<CAFFE_CUDA_NUM_THREADS>
      <<<numBlocksTotal, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N,
          C,
          valsPerChannel,
          X_arr,
          sumScratch_.mutable_data<float>(),
          sumsqScratch_.mutable_data<float>());

  ChannelStatsFinalSumsKernel<CAFFE_CUDA_NUM_THREADS>
      <<<C, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N,
          C,
          numBlocksPerChannel,
          sumScratch_.data<float>(),
          sumsqScratch_.data<float>(),
          sum->template mutable_data<float>(),
          sumsq->template mutable_data<float>());

  return true;
}

REGISTER_CUDA_OPERATOR(ChannelStats, ChannelStatsOp<CUDAContext>);

} // namespace caffe2
