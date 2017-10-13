#ifndef THP_CUDNN_BATCH_NORM_INC
#define THP_CUDNN_BATCH_NORM_INC

#include "../Types.h"
#include "cudnn-wrapper.h"
#include "THC/THC.h"


namespace torch { namespace cudnn {

void cudnn_batch_norm_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* input, THVoidTensor* output, THVoidTensor* weight,
    THVoidTensor* bias, THVoidTensor* running_mean, THVoidTensor* running_var,
    THVoidTensor* save_mean, THVoidTensor* save_var, bool training,
    double exponential_average_factor, double epsilon);

void cudnn_batch_norm_backward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* input, THVoidTensor* grad_output, THVoidTensor* grad_input,
    THVoidTensor* grad_weight, THVoidTensor* grad_bias, THVoidTensor* weight,
    THVoidTensor* running_mean, THVoidTensor* running_var,
    THVoidTensor* save_mean, THVoidTensor* save_var, bool training,
    double epsilon);

}}

#endif
