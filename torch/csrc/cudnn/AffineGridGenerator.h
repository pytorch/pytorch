#ifndef THP_CUDNN_AFFINE_GRID_GENERATOR_INC
#define THP_CUDNN_AFFINE_GRID_GENERATOR_INC

#include "../Types.h"
#include "THC/THC.h"

#include <cudnn.h>

namespace torch { namespace cudnn {

void cudnn_affine_grid_generator_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* theta, THVoidTensor* grid,
    int N, int C, int H, int W);

void cudnn_affine_grid_generator_backward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* grad_theta, THVoidTensor* grad_grid,
    int N, int C, int H, int W);

}}

#endif
