#include "THCUNN.h"
#include "common.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
const int WARP_SIZE = 32;

// The maximum number of threads in a block
const int MAX_BLOCK_SIZE = 512;

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static int getNumThreads(int nElem) {
  int threadSizes[5] = { 32, 64, 128, 256, MAX_BLOCK_SIZE };
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}

// Returns the index of the most significant 1 bit in `val`.
__device__ __forceinline__ int getMSB(int val) {
  return 31 - __clz(val);
}

template <typename Dtype, typename Acctype>
struct Float2 {
  Acctype v1, v2;
  __device__ Float2() {}
  __device__ Float2(Dtype v1, Dtype v2) : v1(ScalarConvert<Dtype, Acctype>::to(v1)), v2(ScalarConvert<Dtype, Acctype>::to(v2)) {}
  __device__ Float2(Dtype v) : v1(ScalarConvert<Dtype, Acctype>::to(v)), v2(ScalarConvert<Dtype, Acctype>::to(v)) {}
  __device__ Float2(int v) : v1(ScalarConvert<int, Acctype>::to(v)), v2(ScalarConvert<int, Acctype>::to(v)) {}
  __device__ Float2& operator+=(const Float2& a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

template <typename Dtype, typename Acctype, typename DeviceTensor3>
struct SumOp {
  __device__ SumOp(const DeviceTensor3 t) : tensor(t) {}
  __device__ __forceinline__ Acctype operator()(int batch, int plane, int n) {
    return ScalarConvert<Dtype, Acctype>::to(tensor[batch][plane][n]);
  }
  const DeviceTensor3 tensor;
};

template <typename Dtype, typename Acctype, typename DeviceTensor3>
struct VarOp {
  __device__ VarOp(Acctype m, const DeviceTensor3 t) : mean(m), tensor(t) {}
  __device__ __forceinline__ Acctype operator()(int batch, int plane, int n) {
    Dtype val = tensor[batch][plane][n];
    return (val - mean) * (val - mean);
  }
  const Acctype mean;
  const DeviceTensor3 tensor;
};

template <typename Dtype, typename Acctype, typename DeviceTensor3>
struct GradOp {
  __device__ GradOp(Acctype m, const DeviceTensor3 i, const DeviceTensor3 g)
    : mean(m), input(i), gradOutput(g) {}
  __device__ __forceinline__ Float2<Dtype, Acctype> operator()(int batch, int plane, int n) {
    Dtype g = gradOutput[batch][plane][n];
    Dtype c = ScalarConvert<Acctype, Dtype>::to(input[batch][plane][n] - mean);
    return Float2<Dtype, Acctype>(g, g * c);
  }
  const Acctype mean;
  const DeviceTensor3 input;
  const DeviceTensor3 gradOutput;
};

// Sum across all threads within a warp
template <typename T>
static __device__ __forceinline__ T warpSum(T val) {
#if __CUDA_ARCH__ >= 300
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    val += WARP_SHFL_XOR(val, 1 << i, WARP_SIZE);
  }
#else
  __shared__ T values[MAX_BLOCK_SIZE];
  values[threadIdx.x] = val;
  __threadfence_block();
  const int base = (threadIdx.x / WARP_SIZE) * WARP_SIZE;
  for (int i = 1; i < WARP_SIZE; i++) {
    val += values[base + ((i + threadIdx.x) % WARP_SIZE)];
  }
#endif
  return val;
}

template <typename Dtype, typename Acctype>
static __device__ __forceinline__ Float2<Dtype, Acctype> warpSum(Float2<Dtype, Acctype> value) {
  value.v1 = warpSum(value.v1);
  value.v2 = warpSum(value.v2);
  return value;
}

// Sum across (batch, x/y/z) applying Op() pointwise
template<typename T, typename Op, typename DeviceTensor3>
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
  __syncthreads();
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

template <typename Dtype, typename Acctype, typename DeviceTensor1, typename DeviceTensor3>
__global__ void BatchNormalizationUpdateOutputInference_kernel(
    const DeviceTensor3 input,
    DeviceTensor3 output,
    const DeviceTensor1 runningMean,
    const DeviceTensor1 runningVar,
    const DeviceTensor1 weight,
    const DeviceTensor1 bias,
    Acctype epsilon) {

  int plane = blockIdx.x;

  Acctype invstd = Acctype(1) / sqrt(runningVar[plane].ldg() + epsilon);
  Acctype mean = ScalarConvert<Dtype, Acctype>::to(runningMean[plane].ldg());
  Acctype gamma = weight.numElements() > 0 ? ScalarConvert<Dtype, Acctype>::to(weight[plane].ldg()) : Acctype(1);
  Acctype beta = bias.numElements() > 0 ? ScalarConvert<Dtype, Acctype>::to(bias[plane].ldg()) : Acctype(0);

  // Write normalized and update the output
  for (int batch = 0; batch < input.getSize(0); batch++) {
    for (int x = threadIdx.x; x < input.getSize(2); x += blockDim.x) {
      Dtype inp = input[batch][plane][x].ldg();
      output[batch][plane][x] = ScalarConvert<Acctype, Dtype>::to(gamma * (inp - mean) * invstd + beta);
    }
  }
}

template <typename Dtype, typename Acctype, typename DeviceTensor1, typename DeviceTensor3>
__global__ void BatchNormalizationUpdateOutput_kernel(
    const DeviceTensor3 input,
    DeviceTensor3 output,
    const DeviceTensor1 weight,
    const DeviceTensor1 bias,
    const Acctype epsilon,
    const Acctype momentum,
    DeviceTensor1 runningMean,
    DeviceTensor1 runningVar,
    DeviceTensor1 saveMean,
    DeviceTensor1 saveStd) {

  int plane = blockIdx.x;
  int N = input.getSize(0) * input.getSize(2);

  Acctype norm = Acctype(1) / N;

  // Compute the mean and variance across (batch, x/y/z)
  Acctype mean = reduce<Acctype>(SumOp<Dtype, Acctype, DeviceTensor3>(input), input, plane) * norm;
  __syncthreads();
  Acctype varN = reduce<Acctype>(VarOp<Dtype, Acctype, DeviceTensor3>(mean, input), input, plane);
  Acctype invStd = 0;
  if (varN != Acctype(0) || epsilon != Acctype(0)) {
    invStd = 1 / sqrt(varN * norm + epsilon);
  }

  // Save the mean, variance, and moving averages
  if (threadIdx.x == 0) {
    // Momentum based writeback
    Acctype unbiasedVar = varN / (N - 1);
    saveMean[plane] = ScalarConvert<Acctype, Dtype>::to(mean);
    saveStd[plane] = ScalarConvert<Acctype, Dtype>::to(invStd);
    if (runningMean.data() != NULL) {
      runningMean[plane] = ScalarConvert<Acctype, Dtype>::to((1 - momentum) * runningMean[plane] + momentum * mean);
    }
    if (runningVar.data() != NULL) {
      runningVar[plane] = ScalarConvert<Acctype, Dtype>::to((1 - momentum) * runningVar[plane] + momentum * unbiasedVar);
    }
  }

  // Write normalized and update the output
  Acctype gamma = weight.numElements() > 0 ? ScalarConvert<Dtype, Acctype>::to(weight[plane]) : ScalarConvert<int, Acctype>::to(1);
  Acctype beta = bias.numElements() > 0 ? ScalarConvert<Dtype, Acctype>::to(bias[plane]) : ScalarConvert<int, Acctype>::to(0);
  for (int batch = 0; batch < input.getSize(0); ++batch) {
    for (int x = threadIdx.x; x < input.getSize(2); x += blockDim.x) {
      Dtype inp = input[batch][plane][x].ldg();
      output[batch][plane][x] = ScalarConvert<Acctype, Dtype>::to(gamma * (inp - mean) * invStd + beta);
    }
  }
}

template <typename Dtype, typename Acctype, typename DeviceTensor1, typename DeviceTensor3>
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
    Acctype scale,
    double eps) {

  int plane = blockIdx.x;
  int N = gradOutput.getSize(0) * gradOutput.getSize(2);

  Acctype mean, stdVal;
  if (train) {
    mean = ScalarConvert<Dtype, Acctype>::to(saveMean[plane]);
    stdVal = ScalarConvert<Dtype, Acctype>::to(saveStd[plane]);
  } else {
    mean = ScalarConvert<Dtype, Acctype>::to(runningMean[plane]);
    stdVal = 1 / sqrt(runningVar[plane] + eps);
  }

  Acctype weightVal = weight.numElements() > 0 ? ScalarConvert<Dtype, Acctype>::to(weight[plane]) : Acctype(1);
  Acctype norm = Acctype(1) / N;

  // Compute two values across (batch, x/y/z) in one pass:
  // 1. Sum(gradOutput)
  // 2. DotProduct(input - mean, gradOutput)
  GradOp<Dtype, Acctype, DeviceTensor3> g(mean, input, gradOutput);
  Float2<Dtype, Acctype> res = reduce<Float2<Dtype, Acctype>, GradOp<Dtype, Acctype, DeviceTensor3>, DeviceTensor3>(g, gradOutput, plane);
  Acctype gradOutputSum = res.v1;
  Acctype dotP = res.v2;

  Acctype gradMean = gradOutputSum * norm;
  Acctype projScale = dotP * norm * stdVal * stdVal;
  Acctype gradScale = stdVal * weightVal;

  if (gradInput.numElements() > 0) {
    for (int batch = 0; batch < gradOutput.getSize(0); ++batch) {
      for (int x = threadIdx.x; x < gradOutput.getSize(2); x += blockDim.x) {
        Dtype gradOut = gradOutput[batch][plane][x];
        if (train) {
          Dtype inp = input[batch][plane][x];
          Acctype proj = (inp - mean) * projScale;
          gradInput[batch][plane][x] = ScalarConvert<Acctype, Dtype>::to((gradOut - proj - gradMean) * gradScale);
        } else {
          gradInput[batch][plane][x] = ScalarConvert<Acctype, Dtype>::to(gradOut * gradScale);
        }
      }
    }
  }

  if (gradWeight.numElements() > 0) {
    if (threadIdx.x == 0) {
      gradWeight[plane] += ScalarConvert<Acctype, Dtype>::to(scale * dotP * stdVal);
    }
  }

  if (gradBias.numElements() > 0) {
    if (threadIdx.x == 0) {
      gradBias[plane] += ScalarConvert<Acctype, Dtype>::to(scale * gradOutputSum);
    }
  }
}

#include "generic/BatchNormalization.cu"
#include "THCGenerateFloatTypes.h"
