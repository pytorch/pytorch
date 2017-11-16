#include "BatchNorm.h"

#include "Descriptors.h"
#include "Types.h"


namespace torch { namespace cudnn {

namespace {

void setInputDescriptor(TensorDescriptor& desc, cudnnDataType_t dataType, const at::Tensor& tensor)
{
  CHECK_ARG(tensor.dim() >= 2 && tensor.dim() <= 5);
  int inputSize[5] = {0};
  int inputStride[5] = {0};
  int nDimension = (tensor.dim() <= 4) ? 4 : tensor.dim();
  for (int i = 0; i < tensor.dim(); ++i) {
    inputSize[i] = (int) tensor.size(i);
    inputStride[i] = (int) tensor.stride(i);
  }
  for (int i = tensor.dim(); i < nDimension; ++i) {
    inputSize[i] = 1;
    inputStride[i] = 1;
  }
  desc.set(dataType, nDimension, inputSize, inputStride);
}

void setScaleDescriptor(TensorDescriptor& desc, cudnnDataType_t dataType, const at::Tensor& tensor, int nDim)
{
  CHECK_ARG(tensor.dim() == 1);
  CHECK_ARG(tensor.stride(0) == 1);  // scale must be contiguous
  int size = (int) tensor.size(0);
  int stride = (int) tensor.stride(0);
  int inputSize[5] = { 1, size, 1, 1, 1 };
  int inputStride[5] = { size * stride, stride, 1, 1, 1 };
  desc.set(dataType, (nDim <= 4) ? 4 : 5, inputSize, inputStride);
}

void* tensorPointer(cudnnDataType_t dataType, const at::Tensor& tensor)
{
  return tensor.data_ptr();
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
    const at::Tensor& input, const at::Tensor& output, const at::Tensor& weight,
    const at::Tensor& bias, const at::Tensor& running_mean, const at::Tensor& running_var,
    const at::Tensor& save_mean, const at::Tensor& save_var, bool training,
    double exponential_average_factor, double epsilon)
{
  CHECK(cudnnSetStream(handle, THCState_getCurrentStream(state)));
  assertSameGPU(input, output, weight, bias, running_mean, running_var,
      save_mean, save_var);
  cudnnBatchNormMode_t mode;
  if (input.dim() == 2) {
    mode = CUDNN_BATCHNORM_PER_ACTIVATION;
  } else {
    mode = CUDNN_BATCHNORM_SPATIAL;
#if CUDNN_VERSION >= 7003
    if(training)
      mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#endif
  }

  TensorDescriptor idesc;  // input descriptor
  TensorDescriptor odesc;  // output descriptor
  TensorDescriptor wdesc;  // descriptor for weight, bias, running_mean, etc.
  setInputDescriptor(idesc, dataType, input);
  setInputDescriptor(odesc, dataType, output);
  setScaleDescriptor(wdesc, scaleDataType(dataType), running_mean, input.dim());

  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  if (training) {
    cudnn_assertContiguous(input);
    cudnn_assertContiguous(bias);
    cudnn_assertContiguous(running_mean);
    cudnn_assertContiguous(running_var);
    cudnn_assertContiguous(save_mean);
    cudnn_assertContiguous(save_var);
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
    cudnn_assertContiguous(input);
    cudnn_assertContiguous(bias);
    cudnn_assertContiguous(running_mean);
    cudnn_assertContiguous(running_var);
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
    const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& grad_input,
    const at::Tensor& grad_weight, const at::Tensor& grad_bias, const at::Tensor& weight,
    const at::Tensor& running_mean, const at::Tensor& running_var,
    const at::Tensor& save_mean, const at::Tensor& save_var, bool training,
    double epsilon)
{
  CHECK(cudnnSetStream(handle, THCState_getCurrentStream(state)));
  assertSameGPU(input, grad_output, grad_input, grad_weight, grad_bias, weight,
      running_mean, running_var, save_mean, save_var);
  cudnnBatchNormMode_t mode;
  if (input.dim() == 2) {
    mode = CUDNN_BATCHNORM_PER_ACTIVATION;
  } else {
    mode = CUDNN_BATCHNORM_SPATIAL;
#if CUDNN_VERSION >= 7003
    if(training)
      mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#endif

  }

  cudnn_assertContiguous(input);
  cudnn_assertContiguous(grad_output);
  cudnn_assertContiguous(grad_weight);
  cudnn_assertContiguous(grad_bias);
  cudnn_assertContiguous(save_mean);
  cudnn_assertContiguous(save_var);

  TensorDescriptor idesc;  // input descriptor
  TensorDescriptor odesc;  // output descriptor
  TensorDescriptor gdesc;  // grad_input descriptor
  TensorDescriptor wdesc;  // descriptor for weight, bias, running_mean, etc.
  setInputDescriptor(idesc, dataType, input);
  setInputDescriptor(odesc, dataType, grad_output);
  setInputDescriptor(gdesc, dataType, grad_input);
  setScaleDescriptor(wdesc, scaleDataType(dataType), weight, input.dim());

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  CHECK(cudnnBatchNormalizationBackward(
    handle, mode, &one, &zero, &one, &zero,
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
