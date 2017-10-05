#include "AffineGridGenerator.h"

#include "Descriptors.h"
#include "Types.h"
#include "Handles.h"


namespace torch { namespace cudnn {

namespace {

void setSamplerDescriptor(SpatialTransformerDescriptor& desc,
			  cudnnDataType_t dataType,
			  int N, int C, int H, int W)
{
  int inputSize[4] = {N, C, H, W};
  desc.set(dataType, 4, inputSize);
}

void* tensorPointer(cudnnDataType_t dataType, THVoidTensor* tensor)
{
  int elementSize = dataSize(dataType);
  char* ptr = (char*) tensor->storage->data;
  ptr += elementSize * tensor->storageOffset;
  return ptr;
}

void checkIOSize(THVoidTensor *theta, THVoidTensor *grid,
		   int N, int H, int W)
{
  THVoidTensor_assertContiguous(theta);
  THVoidTensor_assertContiguous(grid);

  CHECK_ARG(grid->nDimension == 4);
  CHECK_ARG(grid->size[0] == N);
  CHECK_ARG(grid->size[1] == H);
  CHECK_ARG(grid->size[2] == W);
  CHECK_ARG(grid->size[3] == 2);

  CHECK_ARG(theta->nDimension == 3);
  CHECK_ARG(theta->size[0] == N);
  CHECK_ARG(theta->size[1] == 2);
  CHECK_ARG(theta->size[2] == 3);
}

}  // namespace

void cudnn_affine_grid_generator_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* theta, THVoidTensor* grid,
    int N, int C, int H, int W)
{
  useCurrentStream(handle, state);
  assertSameGPU(dataType, theta, grid);
  checkIOSize(theta, grid, N, H, W);  
  SpatialTransformerDescriptor desc;
  setSamplerDescriptor(desc, dataType, N, C, H, W);
  CHECK(cudnnSpatialTfGridGeneratorForward(handle, desc.desc,
					   tensorPointer(dataType, theta),
					   tensorPointer(dataType, grid)));
}

void cudnn_affine_grid_generator_backward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* grad_theta, THVoidTensor* grad_grid,
    int N, int C, int H, int W)
{
  useCurrentStream(handle, state);
  assertSameGPU(dataType, grad_theta, grad_grid);
  checkIOSize(grad_theta, grad_grid, N, H, W);
  SpatialTransformerDescriptor desc;
  setSamplerDescriptor(desc, dataType, N, C, H, W);
  CHECK(cudnnSpatialTfGridGeneratorBackward(handle, desc.desc,
					    tensorPointer(dataType, grad_grid),
					    tensorPointer(dataType, grad_theta)));
}

}}  // namespace torch::cudnn
