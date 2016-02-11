#include "utils.h"
#include "common.h"

template <typename Dtype>
__global__ void MaxUnpoolForward(const int nthreads, const Dtype* bottom_data, const Dtype* bottom_mask, 
    const int num, const int channels, const int iheight, const int iwidth, const int oheight, const int owidth, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) { //index here indices the input pixels
    int c = (index / iwidth / iheight) % channels;
    int n = index / iwidth / iheight / channels;
    top_data += (n*channels + c)*oheight*owidth;
    int maxind = bottom_mask[index]-1;
    
    top_data[maxind] = bottom_data[index];
  }
}

template <typename Dtype>
__global__ void MaxUnpoolBackward(const int nthreads, const Dtype* top_diff, const Dtype* bottom_mask, 
    const int num, const int channels, const int iheight, const int iwidth, const int oheight, const int owidth, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = (index / iwidth / iheight) % channels;
    int n = index / iwidth / iheight / channels;
    top_diff += (n*channels + c)*oheight*owidth; 
    int maxind = bottom_mask[index]-1;

    bottom_diff[index] = top_diff[maxind];
  }
}

static int cunn_SpatialMaxUnpooling_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  int owidth = luaT_getfieldcheckint(L, 1, "owidth");
  int oheight = luaT_getfieldcheckint(L, 1, "oheight");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *indices = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 3, input, output, indices));
  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

  long nInputCols, nInputRows, nInputPlane, batchSize;

  if (input->nDimension == 3) {
    nInputCols = input->size[2];
    nInputRows = input->size[1];
    nInputPlane = input->size[0];
    batchSize = 1;
  }
  else
  {
    nInputCols = input->size[3];
    nInputRows = input->size[2];
    nInputPlane = input->size[1];
    batchSize = input->size[0];
  }

  input = THCudaTensor_newContiguous(state, input);
  indices = THCudaTensor_newContiguous(state, indices);
  THCudaTensor_resize4d(state, output, batchSize, nInputPlane, oheight, owidth);
  THCudaTensor_zero(state, output);

  int count = THCudaTensor_nElement(state, input);

  MaxUnpoolForward <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
      (count, THCudaTensor_data(state, input), THCudaTensor_data(state, indices),
      batchSize, nInputPlane, nInputRows, nInputCols, oheight, owidth, THCudaTensor_data(state, output));

  if(input->nDimension == 3)
    THCudaTensor_resize3d(state, output, nInputPlane, oheight, owidth);

  THCudaTensor_free(state, input);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SpatialMaxUnpooling.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}

static int cunn_SpatialMaxUnpooling_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  int owidth = luaT_getfieldcheckint(L, 1, "owidth");
  int oheight = luaT_getfieldcheckint(L, 1, "oheight");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THCudaTensor *indices = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 4, input, gradOutput, indices, gradInput));

  long nInputCols, nInputRows, nInputPlane, batchSize;

  if (input->nDimension == 3) {
    nInputCols = input->size[2];
    nInputRows = input->size[1];
    nInputPlane = input->size[0];
    batchSize = 1;
  }
  else
  {
    nInputCols = input->size[3];
    nInputRows = input->size[2];
    nInputPlane = input->size[1];
    batchSize = input->size[0];
  }

  input = THCudaTensor_newContiguous(state, input);
  indices = THCudaTensor_newContiguous(state, indices);
  gradOutput = THCudaTensor_newContiguous(state, gradOutput);
  THCudaTensor_resizeAs(state, gradInput, input);
  
  int count = THCudaTensor_nElement(state, input);

  MaxUnpoolBackward <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>> 
      (count, THCudaTensor_data(state, gradOutput), THCudaTensor_data(state, indices),
      batchSize, nInputPlane, nInputRows, nInputCols, oheight, owidth, THCudaTensor_data(state, gradInput));

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SpatialMaxUnpooling.updateGradInput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  // clean
  THCudaTensor_free(state, input);
  THCudaTensor_free(state, gradOutput);

  return 1;
}

static const struct luaL_Reg cunn_SpatialMaxUnpooling__ [] = {
  {"SpatialMaxUnpooling_updateOutput", cunn_SpatialMaxUnpooling_updateOutput},
  {"SpatialMaxUnpooling_updateGradInput", cunn_SpatialMaxUnpooling_updateGradInput},
  {NULL, NULL}
};

void cunn_SpatialMaxUnpooling_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SpatialMaxUnpooling__, "nn");
  lua_pop(L,1);
}
