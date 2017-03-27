#ifndef THP_CUDNN_CONV_INC
#define THP_CUDNN_CONV_INC

#include <vector>
#include <cudnn.h>
#include "THC/THC.h"

#include "../Types.h"
#include "Descriptors.h"

namespace torch { namespace cudnn {

struct ConvolutionParams
{
  cudnnDataType_t dataType;
  int input_size[5];
  int input_stride[5];
  int weight_size[5];
  int pad[3];
  int stride[3];
  int dilation[3];
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
  int groups;
  bool transposed;

  // WARNING: if transposed == true, then idesc and odesc are swapped!
  // WARNING2: WARNING does not apply to odesc_bias :)
  // This allows for reusing the function code (with a small exception in
  // backward_filter)

  Convolution(
      cudnnDataType_t dataType, THVoidTensor* input, THVoidTensor* weight,
      THVoidTensor* bias, THVoidTensor* output, std::vector<int> pad,
      std::vector<int> stride, std::vector<int> dilation, int groups, bool transposed);
};

void cudnn_convolution_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* input, THVoidTensor* weight, THVoidTensor* output,
    Convolution* info, bool benchmark);

void cudnn_convolution_add_bias(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* bias, THVoidTensor* output,
    Convolution* info);

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

// Helpers that allow to queue initialization, conv kernel and bias addition
// without reacquiring GIL in between.
Convolution* cudnn_convolution_full_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* input, THVoidTensor* weight, THVoidTensor* bias, THVoidTensor* output,
    std::vector<int> pad, std::vector<int> stride, std::vector<int> dilation, int groups, bool benchmark);

Convolution* cudnn_convolution_transpose_full_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* input, THVoidTensor* weight, THVoidTensor* bias, THVoidTensor* output,
    std::vector<int> pad, std::vector<int> stride, std::vector<int> dilation, int groups, bool benchmark);

}}  // namespace torch::cudnn

#endif
