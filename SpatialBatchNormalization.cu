#include "THCUNN.h"

#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"

typedef THCDeviceTensor<float, 4> DeviceTensor4;
typedef THCDeviceTensor<float, 1> DeviceTensor1;

// Returns the index of the most significant 1 bit in `val`.
__device__ __forceinline__ int getMSB(int val) {
  return 31 - __clz(val);
}

struct Float2 {
  float v1, v2;
  __device__ Float2() {}
  __device__ Float2(float v1, float v2) : v1(v1), v2(v2) {}
  __device__ Float2(float v) : v1(v), v2(v) {}
  __device__ Float2& operator+=(const Float2& a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

struct SumOp {
  __device__ SumOp(const DeviceTensor4 t) : tensor(t) {}
  __device__ __forceinline__ float operator()(int batch, int plane, int y, int x) {
    return tensor[batch][plane][y][x];
  }
  const DeviceTensor4 tensor;
};

struct VarOp {
  __device__ VarOp(float m, const DeviceTensor4 t) : mean(m), tensor(t) {}
  __device__ __forceinline__ float operator()(int batch, int plane, int y, int x) {
    float val = tensor[batch][plane][y][x];
    return (val - mean) * (val - mean);
  }
  const float mean;
  const DeviceTensor4 tensor;
};

struct GradOp {
  __device__ GradOp(float m, const DeviceTensor4 i, const DeviceTensor4 g)
    : mean(m), input(i), gradOutput(g) {}
  __device__ __forceinline__ Float2 operator()(int batch, int plane, int y, int x) {
    float g = gradOutput[batch][plane][y][x];
    float c = input[batch][plane][y][x] - mean;
    return Float2(g, g * c);
  }
  const float mean;
  const DeviceTensor4 input;
  const DeviceTensor4 gradOutput;
};

// Sum across NumThreads threads within a warp
template<int NumThreads>
static __device__ __forceinline__ float warpSum(float val) {
#if __CUDA_ARCH__ >= 300
  for (int i = 0; i < getMSB(NumThreads); ++i) {
    val += __shfl_xor(val, 1 << i, NumThreads);
  }
#else
  __shared__ float values[NumThreads][NumThreads];
  __syncthreads();
  values[threadIdx.y][threadIdx.x] = val;
  __syncthreads();
  for (int i = 1; i < NumThreads; i++) {
    val += values[threadIdx.y][(i + threadIdx.x) % NumThreads];
  }
  __syncthreads();
#endif
  return val;
}

template<int NumThreads>
static __device__ __forceinline__ Float2 warpSum(Float2 value) {
  value.v1 = warpSum<NumThreads>(value.v1);
  value.v2 = warpSum<NumThreads>(value.v2);
  return value;
}

// Sum across (batch, y, x) applying Op() pointwise
template<typename T, int NumThreads, typename Op>
__device__ T reduce(Op op, DeviceTensor4 tensor, int plane) {
  T sum = (T)0;
  for (int y = threadIdx.y; y < tensor.getSize(2); y += NumThreads) {
    for (int batch = 0; batch < tensor.getSize(0); ++batch) {
      for (int x = threadIdx.x; x < tensor.getSize(3); x += NumThreads) {
        sum += op(batch, plane, y, x);
      }
    }
  }

  // sum over NumThreads within a warp
  sum = warpSum<NumThreads>(sum);

  // 'transpose', and reduce within warp again
  __shared__ T shared[NumThreads];
  if (threadIdx.x == 0) {
    shared[threadIdx.y] = sum;
  }
  __syncthreads();
  sum = warpSum<NumThreads>(shared[threadIdx.x]);
  if (threadIdx.y == 0) {
    shared[threadIdx.x] = sum;
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole gradInput
  return shared[0];
}

template <int Dim>
static THCDeviceTensor<float, Dim> checktensor(THCState *state, THCudaTensor *t) {
  if (!t) {
    return THCDeviceTensor<float, Dim>();
  }
  return toDeviceTensor<float, Dim>(state, t);
}

template<int NumThreads>
__global__ void SpatialBatchNormalizationUpdateOutputInference_kernel(
    const DeviceTensor4 input,
    DeviceTensor4 output,
    DeviceTensor1 runningMean,
    DeviceTensor1 runningVar,
    const DeviceTensor1 weight,
    const DeviceTensor1 bias,
    float epsilon) {

  int plane = blockIdx.x;
  int batch = blockIdx.y;

  float invstd = 1.0f / sqrt(runningVar[plane].ldg() + epsilon);
  float mean = runningMean[plane].ldg();
  float gamma = weight.numElements() > 0 ? weight[plane].ldg() : 1.0f;
  float beta = bias.numElements() > 0 ? bias[plane].ldg() : 0.0f;

  // Write normalized and update the output
  for (int y = threadIdx.y; y < input.getSize(2); y += NumThreads) {
    for (int x = threadIdx.x; x < input.getSize(3); x += NumThreads) {
      float inp = input[batch][plane][y][x].ldg();
      output[batch][plane][y][x] = gamma * (inp - mean) * invstd + beta;
    }
  }
}

template<int NumThreads>
__global__ void SpatialBatchNormalizationUpdateOutput_kernel(
    const DeviceTensor4 input,
    DeviceTensor4 output,
    const DeviceTensor1 weight,
    const DeviceTensor1 bias,
    const float epsilon,
    const float momentum,
    DeviceTensor1 runningMean,
    DeviceTensor1 runningVar,
    DeviceTensor1 saveMean,
    DeviceTensor1 saveStd) {

  assert(blockDim.x == NumThreads);
  assert(blockDim.y == NumThreads);

  int plane = blockIdx.x;
  int N = input.getSize(0) * input.getSize(2) * input.getSize(3);

  float norm = 1.0f / N;

  // Compute the mean and variance across (batch, y, x)
  float mean = reduce<float, NumThreads>(SumOp(input), input, plane) * norm;
  __syncthreads();
  float varN = reduce<float, NumThreads>(VarOp(mean, input), input, plane);
  float invStd = 0.0f;
  if (varN != 0.0f || epsilon != 0.0f) {
    invStd = 1 / sqrt(varN * norm + epsilon);
  }

  // Save the mean, variance, and moving averages
  if (threadIdx.y == 0 && threadIdx.x == 0) {
    // Momentum based writeback
    float unbiasedVar = varN / (N - 1);
    saveMean[plane] = mean;
    saveStd[plane] = invStd;
    runningMean[plane] = (1 - momentum) * runningMean[plane] + momentum * mean;
    runningVar[plane] = (1 - momentum) * runningVar[plane] + momentum * unbiasedVar;
  }

  // Write normalized and update the output
  float gamma = weight.numElements() > 0 ? weight[plane] : 1.0f;
  float beta = bias.numElements() > 0 ? bias[plane] : 0.0f;
  for (int y = threadIdx.y; y < input.getSize(2); y += NumThreads) {
    for (int batch = 0; batch < input.getSize(0); ++batch) {
      for (int x = threadIdx.x; x < input.getSize(3); x += NumThreads) {
        float inp = input[batch][plane][y][x].ldg();
        output[batch][plane][y][x] = gamma * (inp - mean) * invStd + beta;
      }
    }
  }
}

void THNN_CudaSpatialBatchNormalization_updateOutput(THCState *state, THCudaTensor *input_, THCudaTensor *output_, THCudaTensor *weight_, THCudaTensor *bias_, THCudaTensor *runningMean_, THCudaTensor *runningVar_, THCudaTensor *saveMean_, THCudaTensor *saveStd_, bool train, double momentum, double eps) {

  DeviceTensor4 input = checktensor<4>(state, input_);
  DeviceTensor4 output = checktensor<4>(state, output_);
  DeviceTensor1 weight = checktensor<1>(state, weight_);
  DeviceTensor1 bias = checktensor<1>(state, bias_);
  DeviceTensor1 runningMean = checktensor<1>(state, runningMean_);
  DeviceTensor1 runningVar = checktensor<1>(state, runningVar_);
  DeviceTensor1 saveMean = checktensor<1>(state, saveMean_);
  DeviceTensor1 saveStd = checktensor<1>(state, saveStd_);

  cudaStream_t s = THCState_getCurrentStream(state);
  cudaDeviceProp *prop = THCState_getCurrentDeviceProperties(state);

  if (!train) {
    dim3 blocks(input.getSize(1), input.getSize(0));
    if (input.getSize(3) >= 12 && input.getSize(2) >= 12) {
      dim3 threads(16, 16);
      SpatialBatchNormalizationUpdateOutputInference_kernel<16>
        <<<blocks, threads, 0, s>>>
        (input, output, runningMean, runningVar, weight, bias, eps);
    } else {
      dim3 threads(8, 8);
      SpatialBatchNormalizationUpdateOutputInference_kernel<8>
        <<<blocks, threads, 0, s>>>
        (input, output, runningMean, runningVar, weight, bias, eps);
    }
  } else {
    dim3 blocks(input.getSize(1));
    if (input.getSize(3) >= 12 && input.getSize(2) >= 12) {
      dim3 threads(16, 16);
      SpatialBatchNormalizationUpdateOutput_kernel<16>
        <<<blocks, threads, 0, s>>>
        (input, output, weight, bias, eps, momentum, runningMean, runningVar,
         saveMean, saveStd);
    } else {
      dim3 threads(8, 8);
      SpatialBatchNormalizationUpdateOutput_kernel<8>
        <<<blocks, threads, 0, s>>>
        (input, output, weight, bias, eps, momentum, runningMean, runningVar,
         saveMean, saveStd);
    }
  }
}

template<int NumThreads>
__global__ void SpatialBatchNormalizationBackward_kernel(
    const DeviceTensor4 input,
    const DeviceTensor4 gradOutput,
    DeviceTensor4 gradInput,
    DeviceTensor1 gradWeight,
    DeviceTensor1 gradBias,
    const DeviceTensor1 weight,
    const DeviceTensor1 saveMean,
    const DeviceTensor1 saveStd,
    float scale) {

  assert(blockDim.x == NumThreads);
  assert(blockDim.y == NumThreads);

  int plane = blockIdx.x;
  int N = gradOutput.getSize(0) * gradOutput.getSize(2) * gradOutput.getSize(3);

  float mean = saveMean[plane];
  float stdVal = saveStd[plane];
  float weightVal = weight.numElements() > 0 ? weight[plane] : 1.0f;
  float norm = 1.0f / N;

  // Compute two values across (batch, y, x) in one pass:
  // 1. Sum(gradOutput)
  // 2. DotProduct(gradOutput - mean, input)
  Float2 res = reduce<Float2, NumThreads>(GradOp(mean, input, gradOutput), gradOutput, plane);
  float gradOutputSum = res.v1;
  float dotP = res.v2;

  float gradMean = gradOutputSum * norm;
  float projScale = dotP * norm * stdVal * stdVal;
  float gradScale = stdVal * weightVal;

  if (gradInput.numElements() > 0) {
    for (int y = threadIdx.y; y < gradOutput.getSize(2); y += NumThreads) {
      for (int batch = 0; batch < gradOutput.getSize(0); ++batch) {
        for (int x = threadIdx.x; x < gradOutput.getSize(3); x += NumThreads) {
          float gradOut = gradOutput[batch][plane][y][x];
          float inp = input[batch][plane][y][x];
          float proj = (inp - mean) * projScale;

          gradInput[batch][plane][y][x] = (gradOut - proj - gradMean) * gradScale;
        }
      }
    }
  }

  if (gradWeight.numElements() > 0) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      gradWeight[plane] += scale * dotP * stdVal;
    }
  }

  if (gradBias.numElements() > 0) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      gradBias[plane] += scale * gradOutputSum;
    }
  }
}

void THNN_CudaSpatialBatchNormalization_backward(THCState *state, THCudaTensor *input_, THCudaTensor *gradOutput_, THCudaTensor *gradInput_, THCudaTensor *gradWeight_, THCudaTensor *gradBias_, THCudaTensor *weight_, THCudaTensor *saveMean_, THCudaTensor *saveStd_, float scale) {
  DeviceTensor4 input = checktensor<4>(state, input_);
  DeviceTensor4 gradOutput = checktensor<4>(state, gradOutput_);
  DeviceTensor4 gradInput = checktensor<4>(state, gradInput_);
  DeviceTensor1 gradWeight = checktensor<1>(state, gradWeight_);
  DeviceTensor1 gradBias = checktensor<1>(state, gradBias_);
  DeviceTensor1 weight = checktensor<1>(state, weight_);
  DeviceTensor1 saveMean = checktensor<1>(state, saveMean_);
  DeviceTensor1 saveStd = checktensor<1>(state, saveStd_);

  cudaStream_t s = THCState_getCurrentStream(state);

  dim3 blocks(gradOutput.getSize(1));
  if (gradOutput.getSize(3) >= 12 && gradOutput.getSize(2) >= 12) {
    dim3 threads(16, 16);
    SpatialBatchNormalizationBackward_kernel<16>
      <<<blocks, threads, 0, s>>>
      (input, gradOutput, gradInput, gradWeight, gradBias, weight,
       saveMean, saveStd, scale);
  } else {
    dim3 threads(8, 8);
    SpatialBatchNormalizationBackward_kernel<8>
      <<<blocks, threads, 0, s>>>
      (input, gradOutput, gradInput, gradWeight, gradBias, weight,
       saveMean, saveStd, scale);
  }
}
