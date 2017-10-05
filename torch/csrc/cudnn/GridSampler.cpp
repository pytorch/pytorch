#include "GridSampler.h"

#include "Descriptors.h"
#include "Types.h"
#include "Handles.h"


namespace torch { namespace cudnn {

namespace {

void setInputDescriptor(TensorDescriptor& desc, cudnnDataType_t dataType, THVoidTensor* tensor)
{
  CHECK_ARG(tensor->nDimension == 4);
  int inputSize[4] = {0};
  int inputStride[4] = {0};
  for (int i = 0; i < tensor->nDimension; ++i) {
    inputSize[i] = (int) tensor->size[i];
    inputStride[i] = (int) tensor->stride[i];
  }
  desc.set(dataType, 4, inputSize, inputStride);
}

void setSamplerDescriptor(SpatialTransformerDescriptor& desc, cudnnDataType_t dataType, THVoidTensor* tensor)
{
  CHECK_ARG(tensor->nDimension == 4);
  int inputSize[4] = {0};
  for (int i = 0; i < tensor->nDimension; ++i) {
    inputSize[i] = (int) tensor->size[i];
  }
  desc.set(dataType, 4, inputSize);
}

void* tensorPointer(cudnnDataType_t dataType, THVoidTensor* tensor)
{
  int elementSize = dataSize(dataType);
  char* ptr = (char*) tensor->storage->data;
  ptr += elementSize * tensor->storageOffset;
  return ptr;
}

void checkGridSize(THVoidTensor *grid, THVoidTensor *input)
{
  // assert size of grid is n*h*w*2
  // FYI: grid is between [-1, 1], where -1 left most pixel,
  // 1 represents right most pixel (and hence 0 is the center pixel)
  // if grid has values >1 or <-1, those values are ignored
  THVoidTensor_assertContiguous(grid);
  CHECK_ARG(grid->nDimension == 4);
  CHECK_ARG(grid->size[0] == input->size[0]);
  CHECK_ARG(grid->size[3] == 2);  
}

void checkIOSize(THVoidTensor *input, THVoidTensor *output, THVoidTensor *grid)
{
  // assert input = 4 dim, and input, output are same size
  CHECK_ARG(input->nDimension == 4);
  CHECK_ARG(input->nDimension == output->nDimension);
  CHECK_ARG(input->size[0] == output->size[0]);
  CHECK_ARG(input->size[1] == output->size[1]);
  CHECK_ARG(grid->size[1] == output->size[2]);
  CHECK_ARG(grid->size[2] == output->size[3]);
}

}  // namespace

void cudnn_grid_sampler_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* input, THVoidTensor* grid, THVoidTensor* output)
{
  useCurrentStream(handle, state);
  assertSameGPU(dataType, input, output, grid);
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
    THVoidTensor* input, THVoidTensor* grad_input,
    THVoidTensor* grid, THVoidTensor* grad_grid,
    THVoidTensor* grad_output)
{
  useCurrentStream(handle, state);
  assertSameGPU(dataType, input, grad_output, grad_input, grid, grad_grid);
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
