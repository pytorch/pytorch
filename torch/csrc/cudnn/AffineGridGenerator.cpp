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

void* tensorPointer(cudnnDataType_t dataType, const at::Tensor& tensor)
{
  return tensor.data_ptr();
}

void checkIOSize(const at::Tensor& theta, const at::Tensor& grid,
                 int N, int H, int W)
{
  cudnn_assertContiguous(theta);
  cudnn_assertContiguous(grid);

  CHECK_ARG(grid.dim() == 4);
  CHECK_ARG(grid.size(0) == N);
  CHECK_ARG(grid.size(1) == H);
  CHECK_ARG(grid.size(2) == W);
  CHECK_ARG(grid.size(3) == 2);

  CHECK_ARG(theta.dim() == 3);
  CHECK_ARG(theta.size(0) == N);
  CHECK_ARG(theta.size(1) == 2);
  CHECK_ARG(theta.size(2) == 3);
}

}  // namespace

void cudnn_affine_grid_generator_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    const at::Tensor& theta, const at::Tensor& grid,
    int N, int C, int H, int W)
{
  CHECK(cudnnSetStream(handle, THCState_getCurrentStream(state)));
  assertSameGPU(theta, grid);
  checkIOSize(theta, grid, N, H, W);
  SpatialTransformerDescriptor desc;
  setSamplerDescriptor(desc, dataType, N, C, H, W);
  CHECK(cudnnSpatialTfGridGeneratorForward(handle, desc.desc,
                                           tensorPointer(dataType, theta),
                                           tensorPointer(dataType, grid)));
}

void cudnn_affine_grid_generator_backward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    const at::Tensor& grad_theta, const at::Tensor& grad_grid,
    int N, int C, int H, int W)
{
  CHECK(cudnnSetStream(handle, THCState_getCurrentStream(state)));
  assertSameGPU(grad_theta, grad_grid);
  checkIOSize(grad_theta, grad_grid, N, H, W);
  SpatialTransformerDescriptor desc;
  setSamplerDescriptor(desc, dataType, N, C, H, W);
  CHECK(cudnnSpatialTfGridGeneratorBackward(handle, desc.desc,
                                            tensorPointer(dataType, grad_grid),
                                            tensorPointer(dataType, grad_theta)));
}

}}  // namespace torch::cudnn
