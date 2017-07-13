#ifndef THP_CUDNN_GRID_SAMPLER_INC
#define THP_CUDNN_GRID_SAMPLER_INC

#include "../Types.h"
#include "THC/THC.h"

#include <cudnn.h>

namespace torch { namespace cudnn {

void cudnn_grid_sampler_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* input, THVoidTensor* grid, THVoidTensor* output);

void cudnn_grid_sampler_backward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* input, THVoidTensor* grad_input,
    THVoidTensor* grid, THVoidTensor* grad_grid,
    THVoidTensor* grad_output);

}}

#endif
