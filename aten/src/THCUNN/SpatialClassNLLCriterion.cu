#include <limits>

#include <THCUNN/THCUNN.h>
#include <TH/THHalf.h>
#include <THC/THCNumerics.cuh>
#include <THC/THCAtomics.cuh>
#include <THCUNN/common.h>
#include <THC/THCDeviceTensor.cuh>
#include <THC/THCDeviceTensorUtils.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <THC/THCApply.cuh>
#include <c10/macros/Macros.h>
#include <ATen/cuda/detail/KernelUtils.h>

#include <thrust/functional.h>

template <typename Dtype>
__global__ void SpatialClassNLLCriterion_updateOutput_no_reduce_kernel(
    int64_t nthreads,
    THCDeviceTensor<Dtype, 4> input,
    THCDeviceTensor<THCIndex_t, 3> target,
    THCDeviceTensor<Dtype, 3> output,
    Dtype *weights,
    int64_t ignore_index) {
  int64_t batch_size = input.getSize(0);
  int64_t H = input.getSize(2);
  int64_t W = input.getSize(3);

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int64_t b = index % batch_size;
    const int64_t h = (index / batch_size) % H;
    const int64_t w = (index / (batch_size * H)) % W;

    int64_t cur_target = target[b][h][w];
    if (cur_target == ignore_index) {
      output[b][h][w] = ScalarConvert<int, Dtype>::to(0);
      continue;
    }
    Dtype value = input[b][cur_target][h][w];
    Dtype weight =
        weights ? weights[cur_target] : ScalarConvert<int, Dtype>::to(1);
    output[b][h][w] = -value * weight;
  }
}

template <typename Dtype>
__global__ void SpatialClassNLLCriterion_updateGradInput_no_reduce_kernel(
    int64_t nthreads,
    THCDeviceTensor<THCIndex_t, 3> target,
    THCDeviceTensor<Dtype, 3> gradOutput,
    THCDeviceTensor<Dtype, 4> gradInput,
    Dtype *weights,
    int64_t ignore_index) {
  int64_t batch_size = target.getSize(0);
  int64_t H = target.getSize(1);
  int64_t W = target.getSize(2);

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int64_t b = index % batch_size;
    const int64_t h = (index / batch_size) % H;
    const int64_t w = (index / (batch_size * H)) % W;

    int64_t cur_target = target[b][h][w];
    if (cur_target == ignore_index) {
      continue;
    }
    Dtype value =
        -(weights ? weights[cur_target] : ScalarConvert<int, Dtype>::to(1));
    gradInput[b][cur_target][h][w] = value * gradOutput[b][h][w];
  }
}

template <typename T, typename AccumT>
#if defined(__HIP_PLATFORM_HCC__)
C10_LAUNCH_BOUNDS_1(1024)
#endif
__global__ void cunn_SpatialClassNLLCriterion_updateOutput_kernel(
          T *output,
          T *total_weight,
          T *input,
          THCIndex_t *target,
          T *weights,
          int size_average,
          int batch_size,
          int n_classes,
          int map_nelem,
          int blocks_per_sample,
          int64_t ignore_index)
{
  __shared__ AccumT partial_sums[CUDA_NUM_THREADS];

  int i, t;
  T cur_weight;
  AccumT input_sum = 0;
  AccumT acc_weight = 0;

  int sample = blockIdx.x / blocks_per_sample;
  int toffset = sample * map_nelem;
  int ioffset = sample * map_nelem * n_classes;
  int step = blockDim.x * blocks_per_sample;
  for (i = (blockIdx.x % blocks_per_sample) * blockDim.x + threadIdx.x;
       i < map_nelem;
       i += step) {
    t = target[toffset + i];
    if (t != ignore_index) {
      CUDA_KERNEL_ASSERT(t >= 0 && t < n_classes);
      cur_weight = weights ? weights[t] : ScalarConvert<int, T>::to(1);
      input_sum -= input[ioffset + i + map_nelem * t] * cur_weight;
      acc_weight += cur_weight;
    }
  }

  input_sum = reduceBlock(partial_sums, blockDim.x, input_sum, thrust::plus<AccumT>(), AccumT(0));
  __syncthreads();
  acc_weight = reduceBlock(partial_sums, blockDim.x, acc_weight, thrust::plus<AccumT>(), AccumT(0));

  if (threadIdx.x == 0) {
    gpuAtomicAdd(total_weight, ScalarConvert<AccumT, T>::to(acc_weight));
    gpuAtomicAdd(output, ScalarConvert<AccumT, T>::to(input_sum));
  }
}

template<typename T>
__global__ void cunn_SpatialClassNLLCriterion_sizeAverage_kernel(
          T *output,
          T *total_weight,
          int nElement)
{
  if (nElement == 0) {
    // Mean reduction on empty tensors produces NaN
    *output = std::numeric_limits<double>::quiet_NaN();
  }
  if (*total_weight != 0) {
    *output = THCNumerics<T>::div(*output, *total_weight);
  }
}

template<typename T>
__global__ void cunn_SpatialClassNLLCriterion_updateGradInput_kernel(
          T *gradInput,
          T *gradOutput,
          THCIndex_t *target,
          T *weights,
          T *total_weight,
          int size_average,
          int batch_size,
          int n_classes,
          int map_nelem,
          int blocks_per_sample,
          int64_t ignore_index)
{
  if (*total_weight <= 0)
    return;

  int i, t;
  T norm = size_average ? (ScalarConvert<int, T>::to(1) / *total_weight) : ScalarConvert<int, T>::to(1);

  int sample = blockIdx.x / blocks_per_sample;
  int step = blockDim.x * blocks_per_sample;
  int toffset = sample * map_nelem;
  int ioffset = sample * map_nelem * n_classes;
  for (i = (blockIdx.x % blocks_per_sample) * blockDim.x + threadIdx.x;
       i < map_nelem;
       i += step) {
    t = (int)target[toffset + i];
    if (t != ignore_index) {
      CUDA_KERNEL_ASSERT(t >= 0 && t < n_classes);
      gradInput[ioffset + i + map_nelem * t] = -(weights ? weights[t] : ScalarConvert<int, T>::to(1)) * norm * gradOutput[0];
    }
  }
}

#include <THCUNN/generic/SpatialClassNLLCriterion.cu>
#include <THC/THCGenerateFloatTypes.h>

#include <THCUNN/generic/SpatialClassNLLCriterion.cu>
#include <THC/THCGenerateBFloat16Type.h>
