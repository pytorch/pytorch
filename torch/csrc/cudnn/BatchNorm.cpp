#include "BatchNorm.h"

#include "Descriptors.h"
#include "Types.h"


namespace torch { namespace cudnn {

namespace {

void setInputDescriptor(TensorDescriptor& desc, cudnnDataType_t dataType, THVoidTensor* tensor)
{
  CHECK_ARG(tensor->nDimension >= 2 && tensor->nDimension <= 5);
  int inputSize[5] = {0};
  int inputStride[5] = {0};
  int nDimension = (tensor->nDimension <= 4) ? 4 : tensor->nDimension;
  for (int i = 0; i < tensor->nDimension; ++i) {
    inputSize[i] = (int) tensor->size[i];
    inputStride[i] = (int) tensor->stride[i];
  }
  for (int i = tensor->nDimension; i < nDimension; ++i) {
    inputSize[i] = 1;
    inputStride[i] = 1;
  }
  desc.set(dataType, nDimension, inputSize, inputStride);
}

void setScaleDescriptor(TensorDescriptor& desc, cudnnDataType_t dataType, THVoidTensor* tensor, int nDim)
{
  CHECK_ARG(tensor->nDimension == 1);
  CHECK_ARG(tensor->stride[0] == 1);  // scale must be contiguous
  int size = (int) tensor->size[0];
  int stride = (int) tensor->stride[0];
  int inputSize[5] = { 1, size, 1, 1, 1 };
  int inputStride[5] = { size * stride, stride, 1, 1, 1 };
  desc.set(dataType, (nDim <= 4) ? 4 : 5, inputSize, inputStride);
}

void* tensorPointer(cudnnDataType_t dataType, THVoidTensor* tensor)
{
  int elementSize = dataSize(dataType);
  char* ptr = (char*) tensor->storage->data;
  ptr += elementSize * tensor->storageOffset;
  return ptr;
}

cudnnDataType_t scaleDataType(cudnnDataType_t dataType)
{
  // half inputs still use float data type for scale descriptor
  if (dataType == CUDNN_DATA_HALF) {
    return CUDNN_DATA_FLOAT;
  }
  return dataType;
}

}  // namespace

void cudnn_batch_norm_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* input, THVoidTensor* output, THVoidTensor* weight,
    THVoidTensor* bias, THVoidTensor* running_mean, THVoidTensor* running_var,
    THVoidTensor* save_mean, THVoidTensor* save_var, bool training,
    double exponential_average_factor, double epsilon)
{
  assertSameGPU(dataType, input, output, weight, bias, running_mean, running_var,
      save_mean, save_var);
  cudnnBatchNormMode_t mode;
  if (input->nDimension == 2) {
    mode = CUDNN_BATCHNORM_PER_ACTIVATION;
  } else {
    mode = CUDNN_BATCHNORM_SPATIAL;
  }

  TensorDescriptor idesc;  // input descriptor
  TensorDescriptor odesc;  // output descriptor
  TensorDescriptor wdesc;  // descriptor for weight, bias, running_mean, etc.
  setInputDescriptor(idesc, dataType, input);
  setInputDescriptor(odesc, dataType, output);
  setScaleDescriptor(wdesc, scaleDataType(dataType), running_mean, input->nDimension);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  if (training) {
    THVoidTensor_assertContiguous(bias);
    THVoidTensor_assertContiguous(running_mean);
    THVoidTensor_assertContiguous(running_var);
    THVoidTensor_assertContiguous(save_mean);
    THVoidTensor_assertContiguous(save_var);
    CHECK(cudnnBatchNormalizationForwardTraining(
      handle, mode, &one, &zero,
      idesc.desc, tensorPointer(dataType, input),
      odesc.desc, tensorPointer(dataType, output),
      wdesc.desc, tensorPointer(dataType, weight),
      tensorPointer(dataType, bias),
      exponential_average_factor,
      tensorPointer(dataType, running_mean),
      tensorPointer(dataType, running_var),
      epsilon,
      tensorPointer(dataType, save_mean),
      tensorPointer(dataType, save_var)));
  } else {
    THVoidTensor_assertContiguous(bias);
    THVoidTensor_assertContiguous(running_mean);
    THVoidTensor_assertContiguous(running_var);
    CHECK(cudnnBatchNormalizationForwardInference(
      handle, mode, &one, &zero,
      idesc.desc, tensorPointer(dataType, input),
      odesc.desc, tensorPointer(dataType, output),
      wdesc.desc, tensorPointer(dataType, weight),
      tensorPointer(dataType, bias),
      tensorPointer(dataType, running_mean),
      tensorPointer(dataType, running_var),
      epsilon));
  }
}

void cudnn_batch_norm_backward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* input, THVoidTensor* grad_output, THVoidTensor* grad_input,
    THVoidTensor* grad_weight, THVoidTensor* grad_bias, THVoidTensor* weight,
    THVoidTensor* running_mean, THVoidTensor* running_var,
    THVoidTensor* save_mean, THVoidTensor* save_var, bool training,
    double epsilon)
{
  assertSameGPU(dataType, input, grad_output, grad_input, grad_weight, grad_bias, weight,
      running_mean, running_var, save_mean, save_var);
  cudnnBatchNormMode_t mode;
  if (input->nDimension == 2) {
    mode = CUDNN_BATCHNORM_PER_ACTIVATION;
  } else {
    mode = CUDNN_BATCHNORM_SPATIAL;
  }

  TensorDescriptor idesc;  // input descriptor
  TensorDescriptor odesc;  // output descriptor
  TensorDescriptor gdesc;  // grad_input descriptor
  TensorDescriptor wdesc;  // descriptor for weight, bias, running_mean, etc.
  setInputDescriptor(idesc, dataType, input);
  setInputDescriptor(odesc, dataType, grad_output);
  setInputDescriptor(gdesc, dataType, grad_input);
  setScaleDescriptor(wdesc, scaleDataType(dataType), weight, input->nDimension);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  THVoidTensor_assertContiguous(grad_weight);
  THVoidTensor_assertContiguous(grad_bias);
  THVoidTensor_assertContiguous(save_mean);
  THVoidTensor_assertContiguous(save_var);
  CHECK(cudnnBatchNormalizationBackward(
    handle, mode, &one, &zero, &one, &one,
    idesc.desc, tensorPointer(dataType, input),
    odesc.desc, tensorPointer(dataType, grad_output),
    gdesc.desc, tensorPointer(dataType, grad_input),
    wdesc.desc, tensorPointer(dataType, weight),
    tensorPointer(dataType, grad_weight),
    tensorPointer(dataType, grad_bias),
    epsilon,
    tensorPointer(dataType, save_mean),
    tensorPointer(dataType, save_var)));
}

}}  // namespace torch::cudnn
