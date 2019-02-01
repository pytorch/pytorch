#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/channel_backprop_stats_op.h"

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
__global__ void ChannelBackpropStatsBlockKernel(
    int N,
    int C,
    int valsPerChannel,
    const float* X,
    const float* dY,
    const float* mean,
    const float* invStddev,
    float* dBiasBlocks,
    float* dScaleBlocks) {
  __shared__ float dBiasData[blockSize];
  __shared__ float dScaleData[blockSize];

  auto tid = threadIdx.x;
  auto numBlocksPerChannel = (valsPerChannel + blockSize - 1) / blockSize;
  auto localBlockIndex = blockIdx.x % numBlocksPerChannel;
  auto inputIndex = (blockIdx.x / numBlocksPerChannel) * valsPerChannel +
      localBlockIndex * blockSize + tid;
  auto n = blockIdx.x / numBlocksPerChannel / C;
  auto c = (blockIdx.x / numBlocksPerChannel) % C;

  dBiasData[tid] = 0;
  dScaleData[tid] = 0;

  if (localBlockIndex * blockSize + tid < valsPerChannel) {
    dBiasData[tid] += dY[inputIndex];
    dScaleData[tid] +=
        (X[inputIndex] - mean[c]) * invStddev[c] * dY[inputIndex];
  }

  __syncthreads();
  if (blockSize >= 512) {
    if (tid < 256) {
      dBiasData[tid] += dBiasData[tid + 256];
      dScaleData[tid] += dScaleData[tid + 256];
    }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) {
      dBiasData[tid] += dBiasData[tid + 128];
      dScaleData[tid] += dScaleData[tid + 128];
    }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) {
      dBiasData[tid] += dBiasData[tid + 64];
      dScaleData[tid] += dScaleData[tid + 64];
    }
    __syncthreads();
  }

  if (tid < 32) {
    warpReduce<blockSize>(dBiasData, tid);
    warpReduce<blockSize>(dScaleData, tid);
  }

  // output block data sorted by C to simplify second reduction
  if (tid == 0) {
    auto outputIndex = (c * N + n) * numBlocksPerChannel + localBlockIndex;
    dBiasBlocks[outputIndex] = dBiasData[0];
    dScaleBlocks[outputIndex] = dScaleData[0];
  }
}

template <unsigned int blockSize>
__global__ void ChannelBackpropStatsFinalSumsKernel(
    int N,
    int C,
    int numSumsPerChannel,
    const float* dBiasScratch,
    const float* dScaleScratch,
    float* dBias,
    float* dScale) {
  __shared__ float dBiasData[blockSize];
  __shared__ float dScaleData[blockSize];

  auto tid = threadIdx.x;
  auto inputIndex = blockIdx.x * N * numSumsPerChannel + tid;
  dBiasData[tid] = 0;
  dScaleData[tid] = 0;
  for (auto i = inputIndex; i < (blockIdx.x + 1) * N * numSumsPerChannel;
       i += blockSize) {
    dBiasData[tid] += dBiasScratch[i];
    dScaleData[tid] += dScaleScratch[i];
  }
  __syncthreads();
  if (blockSize >= 512) {
    if (tid < 256) {
      dBiasData[tid] += dBiasData[tid + 256];
      dScaleData[tid] += dScaleData[tid + 256];
    }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) {
      dBiasData[tid] += dBiasData[tid + 128];
      dScaleData[tid] += dScaleData[tid + 128];
    }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) {
      dBiasData[tid] += dBiasData[tid + 64];
      dScaleData[tid] += dScaleData[tid + 64];
    }
    __syncthreads();
  }
  if (tid < 32) {
    warpReduce<blockSize>(dBiasData, tid);
    warpReduce<blockSize>(dScaleData, tid);
  }

  if (tid == 0) {
    dBias[blockIdx.x] = dBiasData[0];
    dScale[blockIdx.x] = dScaleData[0];
  }
}
} // namespace

template <>
bool ChannelBackpropStatsOp<CUDAContext>::RunOnDevice() {
  const auto& X = Input(INPUT);
  const auto& dY = Input(OUTPUT_GRAD);
  const auto& mean = Input(SAVED_MEAN);
  const auto& invStddev = Input(SAVED_INV_STDDEV);
  CAFFE_ENFORCE(X.dim() >= 3 && X.dim() <= 5);
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim() > 3 ? X.dim32(3) : 1;
  const int D = X.dim() > 4 ? X.dim32(4) : 1;

  const auto Xarr = X.data<float>();
  const auto dYarr = dY.data<float>();
  const auto meanArr = mean.data<float>();
  const auto invStddevArr = invStddev.data<float>();

  auto dBias = Output(BIAS_GRAD, {C}, at::dtype<float>());
  auto dScale = Output(SCALE_GRAD, {C}, at::dtype<float>());

  const auto valsPerChannel = H * W * D;

  const auto numBlocksPerChannel = CAFFE_GET_BLOCKS(valsPerChannel);
  const auto numBlocksTotal = numBlocksPerChannel * N * C;

  ReinitializeTensor(
      &dBiasScratch_, {numBlocksTotal}, at::dtype<float>().device(CUDA));
  ReinitializeTensor(
      &dScaleScratch_, {numBlocksTotal}, at::dtype<float>().device(CUDA));

  ChannelBackpropStatsBlockKernel<CAFFE_CUDA_NUM_THREADS>
      <<<numBlocksTotal, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N,
          C,
          valsPerChannel,
          Xarr,
          dYarr,
          meanArr,
          invStddevArr,
          dBiasScratch_.mutable_data<float>(),
          dScaleScratch_.mutable_data<float>());

  ChannelBackpropStatsFinalSumsKernel<CAFFE_CUDA_NUM_THREADS>
      <<<C, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N,
          C,
          numBlocksPerChannel,
          dBiasScratch_.data<float>(),
          dScaleScratch_.data<float>(),
          dBias->template mutable_data<float>(),
          dScale->template mutable_data<float>());

  return true;
}

REGISTER_CUDA_OPERATOR(
    ChannelBackpropStats,
    ChannelBackpropStatsOp<CUDAContext>);

} // namespace caffe2
