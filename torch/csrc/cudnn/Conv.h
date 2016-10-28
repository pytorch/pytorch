#ifndef THP_CUDNN_CONV_INC
#define THP_CUDNN_CONV_INC

#include <cudnn.h>
#include "THC/THC.h"

#include "../Types.h"
#include "Descriptors.h"

namespace torch { namespace cudnn {

struct ConvolutionParams
{
  cudnnDataType_t dataType;
  int input_size[4];
  int input_stride[4];
  int weight_size[4];
  int pad[2];
  int stride[2];
  int groups;
};

struct Convolution
{
  ConvolutionParams params;
  TensorDescriptor idesc;
  TensorDescriptor odesc;
  TensorDescriptor odesc_bias;
  TensorDescriptor bdesc;
  FilterDescriptor wdesc;
  ConvolutionDescriptor cdesc;

  Convolution(
      cudnnDataType_t dataType, THVoidTensor* input, THVoidTensor* weight,
      THVoidTensor* bias, THVoidTensor* output, int pad[2], int stride[2], int groups);
};

Convolution* cudnn_convolution_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* input, THVoidTensor* weight, THVoidTensor* bias, THVoidTensor* output,
    int padH, int padW, int dH, int dW, int groups, bool benchmark);

void cudnn_convolution_backward_data(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* gradOutput, THVoidTensor* gradInput, THVoidTensor* weight,
    Convolution* info, bool benchmark);

void cudnn_convolution_backward_filter(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* gradOutput, THVoidTensor* input, THVoidTensor* gradWeight,
    Convolution* info, bool benchmark);

void cudnn_convolution_backward_bias(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* gradOutput, THVoidTensor* gradBias, Convolution* info);

}}  // namespace torch::cudnn

#endif
