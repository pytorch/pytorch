#include "GridSampler.h"

#include "Descriptors.h"
#include "Types.h"


namespace torch { namespace cudnn {

namespace {

void setInputDescriptor(TensorDescriptor& desc, cudnnDataType_t dataType, const at::Tensor& tensor)
{
  CHECK_ARG(tensor.dim() == 4);
  int inputSize[4] = {0};
  int inputStride[4] = {0};
  for (int i = 0; i < tensor.dim(); ++i) {
    inputSize[i] = (int) tensor.size(i);
    inputStride[i] = (int) tensor.stride(i);
  }
  desc.set(dataType, 4, inputSize, inputStride);
}

void setSamplerDescriptor(SpatialTransformerDescriptor& desc, cudnnDataType_t dataType, const at::Tensor& tensor)
{
  CHECK_ARG(tensor.dim() == 4);
  int inputSize[4] = {0};
  for (int i = 0; i < tensor.dim(); ++i) {
    inputSize[i] = (int) tensor.size(i);
  }
  desc.set(dataType, 4, inputSize);
}

void* tensorPointer(cudnnDataType_t dataType, const at::Tensor& tensor)
{
  return tensor.data_ptr();
}

void checkGridSize(const at::Tensor& grid, const at::Tensor& input)
{
  // assert size of grid is n*h*w*2
  // FYI: grid is between [-1, 1], where -1 left most pixel,
  // 1 represents right most pixel (and hence 0 is the center pixel)
  // if grid has values >1 or <-1, those values are ignored
  cudnn_assertContiguous(grid);
  CHECK_ARG(grid.dim() == 4);
  CHECK_ARG(grid.size(0) == input.size(0));
  CHECK_ARG(grid.size(3) == 2);
}

void checkIOSize(const at::Tensor& input, const at::Tensor& output, const at::Tensor& grid)
{
  // assert input = 4 dim, and input, output are same size
  CHECK_ARG(input.dim() == 4);
  CHECK_ARG(input.dim() == output.dim());
  CHECK_ARG(input.size(0) == output.size(0));
  CHECK_ARG(input.size(1) == output.size(1));
  CHECK_ARG(grid.size(1) == output.size(2));
  CHECK_ARG(grid.size(2) == output.size(3));
}

}  // namespace

void cudnn_grid_sampler_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    const at::Tensor& input, const at::Tensor& grid, const at::Tensor& output)
{
  CHECK(cudnnSetStream(handle, THCState_getCurrentStream(state)));
  assertSameGPU(input, output, grid);
  checkGridSize(grid, input);
  checkIOSize(input, output, grid);

  TensorDescriptor idesc;  // input descriptor
  TensorDescriptor odesc;  // output descriptor
  SpatialTransformerDescriptor desc; // sampler descriptor
  setInputDescriptor(idesc, dataType, input);
  setInputDescriptor(odesc, dataType, output);
  setSamplerDescriptor(desc, dataType, output);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  CHECK(cudnnSpatialTfSamplerForward(
      handle, desc.desc, &one,
      idesc.desc, tensorPointer(dataType, input),
      tensorPointer(dataType, grid), &zero,
      odesc.desc, tensorPointer(dataType, output)
  ));
}

void cudnn_grid_sampler_backward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    const at::Tensor& input, const at::Tensor& grad_input,
    const at::Tensor& grid, const at::Tensor& grad_grid,
    const at::Tensor& grad_output)
{
  CHECK(cudnnSetStream(handle, THCState_getCurrentStream(state)));
  assertSameGPU(input, grad_output, grad_input, grid, grad_grid);
  checkGridSize(grid, input);
  checkGridSize(grad_grid, input);
  checkGridSize(grid, grad_input);
  checkIOSize(input, grad_output, grid);

  TensorDescriptor idesc;  // input descriptor
  TensorDescriptor odesc;  // grad_output descriptor
  TensorDescriptor gdesc;  // grad_input descriptor
  SpatialTransformerDescriptor desc; // sampler descriptor
  setInputDescriptor(idesc, dataType, input);
  setInputDescriptor(odesc, dataType, grad_output);
  setInputDescriptor(gdesc, dataType, grad_input);
  setSamplerDescriptor(desc, dataType, grad_output);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  CHECK(cudnnSpatialTfSamplerBackward(
    handle, desc.desc, &one,
    idesc.desc, tensorPointer(dataType, input),
    &zero, gdesc.desc, tensorPointer(dataType, grad_input), &one,
    odesc.desc, tensorPointer(dataType, grad_output),
    tensorPointer(dataType, grid),
    &zero, tensorPointer(dataType, grad_grid)
  ));
}

}}  // namespace torch::cudnn
