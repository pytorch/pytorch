#include <limits>

#include <THCUNN/THCUNN.h>
#include <THCUNN/common.h>
#include <TH/THHalf.h>
#include <THC/THCNumerics.cuh>
#include <THC/THCDeviceTensor.cuh>
#include <THC/THCDeviceTensorUtils.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/macros/Macros.h>

#include <stdio.h>

static const int NTHREADS = 32;

template <typename Dtype>
__global__ void cunn_ClassNLLCriterion_updateOutput_kernel1(Dtype *output,
                                                           Dtype *total_weight,
                                                           Dtype *input,
                                                           THCIndex_t  *target,
                                                           Dtype *weights,
                                                           int size_average,
                                                           int n_classes,
                                                           int64_t ignore_index) {
  CUDA_KERNEL_ASSERT(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);

  // TODO: T4951791 Reuse code between updateOutput_kernel1 and
  // updateOutput_kernel.

  int t = (int) *target;
  if (t != (int) ignore_index) {
    CUDA_KERNEL_ASSERT(t >= 0 && t < n_classes);
    Dtype cur_weight = weights ? weights[t] : ScalarConvert<int, Dtype>::to(1);
    *output = -cur_weight * input[t];
    *total_weight = cur_weight;
    if (size_average && *total_weight > 0) {
      *output /= *total_weight;
    }
  }
}

template <typename Dtype>
__global__ void ClassNLLCriterion_updateOutput_no_reduce_kernel(
    int batch_size,
    THCDeviceTensor<Dtype, 2> input,
    THCDeviceTensor<THCIndex_t, 1> target,
    THCDeviceTensor<Dtype, 1> output,
    Dtype *weights,
    int n_classes,
    int ignore_index) {

  CUDA_KERNEL_LOOP(index, batch_size) {
    int cur_target = target[index];
    if (cur_target == ignore_index) {
      output[index] = ScalarConvert<int, Dtype>::to(0);
      continue;
    }
    CUDA_KERNEL_ASSERT(cur_target  >= 0 && cur_target  < n_classes);
    Dtype weight =
       weights ? weights[cur_target] : ScalarConvert<int, Dtype>::to(1);
    output[index] = -weight * input[index][cur_target];
  }
}

template <typename Dtype, typename Acctype>
__global__ void cunn_ClassNLLCriterion_updateOutput_kernel(Dtype *output,
                                                           Dtype *total_weight,
                                                           Dtype *input,
                                                           THCIndex_t *target,
                                                           Dtype *weights,
                                                           int size_average,
                                                           int nframe,
                                                           int ndim,
                                                           int n_classes,
                                                           int64_t ignore_index) {
  __shared__ Acctype shInputs[NTHREADS], acc_weight[NTHREADS];
  int i, t;
  Dtype cur_weight;

  shInputs[threadIdx.x] = ScalarConvert<int, Acctype>::to(0);
  acc_weight[threadIdx.x] = ScalarConvert<int, Acctype>::to(0);
  for (i = threadIdx.x; i < nframe; i += NTHREADS) {
      t = target[i];
      if (t != (int) ignore_index) {
        CUDA_KERNEL_ASSERT(t >= 0 && t < n_classes);
        cur_weight = weights ? weights[t] : ScalarConvert<int, Dtype>::to(1);
        shInputs[threadIdx.x] -= input[i * ndim + t] * cur_weight;
        acc_weight[threadIdx.x] += cur_weight;
      }
  }
  __syncthreads();

  // TODO: T4951791 Reuse code between updateOutput_kernel1 and
  // updateOutput_kernel

  if (threadIdx.x == 0) {
    *output = *total_weight = ScalarConvert<int, Dtype>::to(0);
    Acctype outputAcc = 0;
    Acctype total_weightAcc = 0;
    for (i = 0; i < NTHREADS; ++i){
      // FIXME should we do somethigng here
      outputAcc += shInputs[i];
      total_weightAcc += acc_weight[i];
    }
    *total_weight = ScalarConvert<Acctype, Dtype>::to(total_weightAcc);
    *output = ScalarConvert<Acctype, Dtype>::to(outputAcc);
    if (size_average) {
      if (nframe == 0) {
        // Mean reduction on empty tensors produces NaN
        *output = std::numeric_limits<double>::quiet_NaN();
      }
      if (*total_weight != 0) {
        *output = ScalarConvert<Acctype, Dtype>::to(outputAcc / total_weightAcc);
      }
    }

  }
}

#include <THCUNN/generic/ClassNLLCriterion.cu>
#include <THC/THCGenerateFloatTypes.h>

#include <THCUNN/generic/ClassNLLCriterion.cu>
#include <THC/THCGenerateBFloat16Type.h>
