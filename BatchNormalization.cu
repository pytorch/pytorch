#include "THCUNN.h"

#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"

const int WARP_SIZE = 32;
typedef THCDeviceTensor<float, 3> DeviceTensor3;
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
  __device__ SumOp(const DeviceTensor3 t) : tensor(t) {}
  __device__ __forceinline__ float operator()(int batch, int plane, int n) {
    return tensor[batch][plane][n];
  }
  const DeviceTensor3 tensor;
};

struct VarOp {
  __device__ VarOp(float m, const DeviceTensor3 t) : mean(m), tensor(t) {}
  __device__ __forceinline__ float operator()(int batch, int plane, int n) {
    float val = tensor[batch][plane][n];
    return (val - mean) * (val - mean);
  }
  const float mean;
  const DeviceTensor3 tensor;
};

struct GradOp {
  __device__ GradOp(float m, const DeviceTensor3 i, const DeviceTensor3 g)
    : mean(m), input(i), gradOutput(g) {}
  __device__ __forceinline__ Float2 operator()(int batch, int plane, int n) {
    float g = gradOutput[batch][plane][n];
    float c = input[batch][plane][n] - mean;
    return Float2(g, g * c);
  }
  const float mean;
  const DeviceTensor3 input;
  const DeviceTensor3 gradOutput;
};

// Sum across NumThreads threads within a warp
static __device__ __forceinline__ float warpSum(float val) {
#if __CUDA_ARCH__ >= 300
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    val += __shfl_xor(val, 1 << i, WARP_SIZE);
  }
#else
  const int MAX_BLOCK_SIZE = 256; // maximum block size this module uses
  __shared__ float values[MAX_BLOCK_SIZE];
  __syncthreads();
  values[threadIdx.x] = val;
  __syncthreads();
  const int base = (threadIdx.x / WARP_SIZE) * WARP_SIZE;
  for (int i = 1; i < WARP_SIZE; i++) {
    val += values[base + ((i + threadIdx.x) % WARP_SIZE)];
  }
  __syncthreads();
#endif
  return val;
}

static __device__ __forceinline__ Float2 warpSum(Float2 value) {
  value.v1 = warpSum(value.v1);
  value.v2 = warpSum(value.v2);
  return value;
}

// Sum across (batch, x/y/z) applying Op() pointwise
template<typename T, typename Op>
__device__ T reduce(Op op, DeviceTensor3 tensor, int plane) {
  T sum = (T)0;
  for (int batch = 0; batch < tensor.getSize(0); ++batch) {
    for (int x = threadIdx.x; x < tensor.getSize(2); x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // sum over NumThreads within a warp
  sum = warpSum(sum);

  // 'transpose', and reduce within warp again
  __shared__ T shared[32];
  if (threadIdx.x % WARP_SIZE == 0) {
    shared[threadIdx.x / WARP_SIZE] = sum;
  }
  if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
    // zero out the other entries in shared
    shared[threadIdx.x] = (T)0;
  }
  __syncthreads();
  if (threadIdx.x / WARP_SIZE == 0) {
    sum = warpSum(shared[threadIdx.x]);
    if (threadIdx.x == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole gradInput
  return shared[0];
}

template <int Dim>
static THCDeviceTensor<float, Dim> devicetensor(THCState *state, THCudaTensor *t) {
  if (!t) {
    return THCDeviceTensor<float, Dim>();
  }

  int inDim = THCudaTensor_nDimension(state, t);
  if (inDim == Dim) {
    return toDeviceTensor<float, Dim>(state, t);
  }

  // View in which the last dimensions are collapsed or expanded as needed
  THAssert(THCudaTensor_isContiguous(state, t));
  int size[Dim];
  for (int i = 0; i < Dim || i < inDim; ++i) {
    if (i < Dim && i < inDim) {
      size[i] = t->size[i];
    } else if (i < Dim) {
      size[i] = 1;
    } else {
      size[Dim - 1] *= t->size[i];
    }
  }
  return THCDeviceTensor<float, Dim>(THCudaTensor_data(state, t), size);
}

__global__ void BatchNormalizationUpdateOutputInference_kernel(
    const DeviceTensor3 input,
    DeviceTensor3 output,
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
  for (int x = threadIdx.x; x < input.getSize(2); x += blockDim.x) {
    float inp = input[batch][plane][x].ldg();
    output[batch][plane][x] = gamma * (inp - mean) * invstd + beta;
  }
}

__global__ void BatchNormalizationUpdateOutput_kernel(
    const DeviceTensor3 input,
    DeviceTensor3 output,
    const DeviceTensor1 weight,
    const DeviceTensor1 bias,
    const float epsilon,
    const float momentum,
    DeviceTensor1 runningMean,
    DeviceTensor1 runningVar,
    DeviceTensor1 saveMean,
    DeviceTensor1 saveStd) {

  int plane = blockIdx.x;
  int N = input.getSize(0) * input.getSize(2);

  float norm = 1.0f / N;

  // Compute the mean and variance across (batch, x/y/z)
  float mean = reduce<float>(SumOp(input), input, plane) * norm;
  __syncthreads();
  float varN = reduce<float>(VarOp(mean, input), input, plane);
  float invStd = 0.0f;
  if (varN != 0.0f || epsilon != 0.0f) {
    invStd = 1 / sqrt(varN * norm + epsilon);
  }

  // Save the mean, variance, and moving averages
  if (threadIdx.x == 0) {
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
  for (int batch = 0; batch < input.getSize(0); ++batch) {
    for (int x = threadIdx.x; x < input.getSize(2); x += blockDim.x) {
      float inp = input[batch][plane][x].ldg();
      output[batch][plane][x] = gamma * (inp - mean) * invStd + beta;
    }
  }
}

static int getNumThreads(int nElem) {
  int threadSizes[5] = { 32, 64, 128, 256, 512 };
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return 512;
}

void THNN_CudaBatchNormalization_updateOutput(
  THCState *state, THCudaTensor *input_, THCudaTensor *output_,
  THCudaTensor *weight_, THCudaTensor *bias_, THCudaTensor *runningMean_,
  THCudaTensor *runningVar_, THCudaTensor *saveMean_, THCudaTensor *saveStd_,
  bool train, double momentum, double eps) {

  DeviceTensor3 input = devicetensor<3>(state, input_);
  DeviceTensor3 output = devicetensor<3>(state, output_);
  DeviceTensor1 weight = devicetensor<1>(state, weight_);
  DeviceTensor1 bias = devicetensor<1>(state, bias_);
  DeviceTensor1 runningMean = devicetensor<1>(state, runningMean_);
  DeviceTensor1 runningVar = devicetensor<1>(state, runningVar_);
  DeviceTensor1 saveMean = devicetensor<1>(state, saveMean_);
  DeviceTensor1 saveStd = devicetensor<1>(state, saveStd_);

  cudaStream_t s = THCState_getCurrentStream(state);
  cudaDeviceProp *prop = THCState_getCurrentDeviceProperties(state);

  if (!train) {
    dim3 blocks(input.getSize(1), input.getSize(0));
    dim3 threads(getNumThreads(input.getSize(2)));
    BatchNormalizationUpdateOutputInference_kernel<<<blocks, threads, 0, s>>>(
      input, output, runningMean, runningVar, weight, bias, eps);
  } else {
    dim3 blocks(input.getSize(1));
    dim3 threads(getNumThreads(input.getSize(2)));
    BatchNormalizationUpdateOutput_kernel<<<blocks, threads, 0, s>>>(
      input, output, weight, bias, eps, momentum, runningMean, runningVar,
      saveMean, saveStd);
  }
  THCudaCheck(cudaGetLastError());
}

__global__ void BatchNormalizationBackward_kernel(
    const DeviceTensor3 input,
    const DeviceTensor3 gradOutput,
    DeviceTensor3 gradInput,
    DeviceTensor1 gradWeight,
    DeviceTensor1 gradBias,
    const DeviceTensor1 weight,
    const DeviceTensor1 runningMean,
    const DeviceTensor1 runningVar,
    const DeviceTensor1 saveMean,
    const DeviceTensor1 saveStd,
    bool train,
    float scale,
    double eps) {

  int plane = blockIdx.x;
  int N = gradOutput.getSize(0) * gradOutput.getSize(2);

  float mean, stdVal;
  if (train) {
    mean = saveMean[plane];
    stdVal = saveStd[plane];
  } else {
    mean = runningMean[plane];
    stdVal = 1 / sqrt(runningVar[plane] + eps);
  }

  float weightVal = weight.numElements() > 0 ? weight[plane] : 1.0f;
  float norm = 1.0f / N;

  // Compute two values across (batch, x/y/z) in one pass:
  // 1. Sum(gradOutput)
  // 2. DotProduct(input - mean, gradOutput)
  Float2 res = reduce<Float2>(GradOp(mean, input, gradOutput), gradOutput, plane);
  float gradOutputSum = res.v1;
  float dotP = res.v2;

  float gradMean = gradOutputSum * norm;
  float projScale = dotP * norm * stdVal * stdVal;
  float gradScale = stdVal * weightVal;

  if (gradInput.numElements() > 0) {
    for (int batch = 0; batch < gradOutput.getSize(0); ++batch) {
      for (int x = threadIdx.x; x < gradOutput.getSize(2); x += blockDim.x) {
        float gradOut = gradOutput[batch][plane][x];
        if (train) {
          float inp = input[batch][plane][x];
          float proj = (inp - mean) * projScale;
          gradInput[batch][plane][x] = (gradOut - proj - gradMean) * gradScale;
        } else {
          gradInput[batch][plane][x] = gradOut * gradScale;
        }
      }
    }
  }

  if (gradWeight.numElements() > 0) {
    if (threadIdx.x == 0) {
      gradWeight[plane] += scale * dotP * stdVal;
    }
  }

  if (gradBias.numElements() > 0) {
    if (threadIdx.x == 0) {
      gradBias[plane] += scale * gradOutputSum;
    }
  }
}

void THNN_CudaBatchNormalization_backward(
  THCState *state, THCudaTensor *input_, THCudaTensor *gradOutput_,
  THCudaTensor *gradInput_, THCudaTensor *gradWeight_, THCudaTensor *gradBias_,
  THCudaTensor *weight_, THCudaTensor *runningMean_, THCudaTensor *runningVar_,
  THCudaTensor *saveMean_, THCudaTensor *saveStd_, bool train, float scale, double eps) {

  DeviceTensor3 input = devicetensor<3>(state, input_);
  DeviceTensor3 gradOutput = devicetensor<3>(state, gradOutput_);
  DeviceTensor3 gradInput = devicetensor<3>(state, gradInput_);
  DeviceTensor1 gradWeight = devicetensor<1>(state, gradWeight_);
  DeviceTensor1 gradBias = devicetensor<1>(state, gradBias_);
  DeviceTensor1 weight = devicetensor<1>(state, weight_);
  DeviceTensor1 runningMean = devicetensor<1>(state, runningMean_);
  DeviceTensor1 runningVar = devicetensor<1>(state, runningVar_);
  DeviceTensor1 saveMean = devicetensor<1>(state, saveMean_);
  DeviceTensor1 saveStd = devicetensor<1>(state, saveStd_);

  cudaStream_t s = THCState_getCurrentStream(state);

  dim3 blocks(gradOutput.getSize(1));
  dim3 threads(getNumThreads(gradOutput.getSize(2)));
  BatchNormalizationBackward_kernel<<<blocks, threads, 0, s>>>(
    input, gradOutput, gradInput, gradWeight, gradBias, weight, runningMean, runningVar,
    saveMean, saveStd, train, scale, eps);
  THCudaCheck(cudaGetLastError());
}
