#include <limits>

#include <ATen/cuda/detail/KernelUtils.h>
#include <TH/THHalf.h>
#include <THCUNN/THCUNN.h>
#include <THCUNN/common.h>
#include <c10/macros/Macros.h>
#include <THC/THCDeviceTensor.cuh>
#include <THC/THCDeviceTensorUtils.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <THC/THCNumerics.cuh>

#include <stdio.h>

static const int NTHREADS = 32;

template <typename Dtype>
__global__ void ClassNLLCriterion_updateGradInput_no_reduce_kernel(
    int batch_size,
    THCDeviceTensor<THCIndex_t, 1> target,
    THCDeviceTensor<Dtype, 1> gradOutput,
    THCDeviceTensor<Dtype, 2> gradInput,
    Dtype* weights,
    int n_classes,
    int ignore_index) {
  CUDA_KERNEL_LOOP(index, batch_size) {
    int cur_target = target[index];
    if (cur_target == ignore_index) {
      continue;
    }
    CUDA_KERNEL_ASSERT(cur_target >= 0 && cur_target < n_classes);
    Dtype weight =
        weights ? weights[cur_target] : ScalarConvert<int, Dtype>::to(1);
    gradInput[index][cur_target] = -weight * gradOutput[index];
  }
}

template <typename Dtype>
__global__ void cunn_ClassNLLCriterion_updateGradInput_kernel1(
    Dtype* gradInput,
    Dtype* gradOutput,
    Dtype* weights,
    THCIndex_t* target,
    Dtype* total_weight,
    int size_average,
    int n_classes,
    int64_t ignore_index) {
  if (*total_weight <= 0) {
    return;
  }
  Dtype norm = size_average ? (ScalarConvert<int, Dtype>::to(1) / *total_weight)
                            : ScalarConvert<int, Dtype>::to(1);
  int t = (int)*target;
  if (t != (int)ignore_index) {
    CUDA_KERNEL_ASSERT(t >= 0 && t < n_classes);
    gradInput[t] = -(weights ? weights[t] : ScalarConvert<int, Dtype>::to(1)) *
        norm * gradOutput[0];
  }
}

template <typename Dtype>
__global__ void cunn_ClassNLLCriterion_updateGradInput_kernel(
    Dtype* gradInput,
    Dtype* gradOutput,
    THCIndex_t* target,
    Dtype* weights,
    Dtype* total_weight,
    int size_average,
    int nframe,
    int ndim,
    int n_classes,
    int64_t ignore_index) {
  if (*total_weight <= 0) {
    return;
  }
  int i, t;
  Dtype norm = size_average ? (ScalarConvert<int, Dtype>::to(1) / *total_weight)
                            : ScalarConvert<int, Dtype>::to(1);

  for (i = threadIdx.x; i < nframe; i += NTHREADS) {
    t = (int)target[i];
    if (t != (int)ignore_index) {
      CUDA_KERNEL_ASSERT(t >= 0 && t < n_classes);
      gradInput[i * ndim + t] =
          -(weights ? weights[t] : ScalarConvert<int, Dtype>::to(1)) * norm *
          gradOutput[0];
    }
  }
}

#include <THC/THCGenerateFloatTypes.h>
#include <THCUNN/generic/ClassNLLCriterion.cu>

#include <THC/THCGenerateBFloat16Type.h>
#include <THCUNN/generic/ClassNLLCriterion.cu>
