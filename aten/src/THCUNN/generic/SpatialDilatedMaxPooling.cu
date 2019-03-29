#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/SpatialDilatedMaxPooling.cu"
#else

#include <THCUNN/common.h>
#include <THCUNN/generic/pooling_shape.h>
#include <ATen/cuda/CUDAContext.h>

static inline void THNN_(SpatialDilatedMaxPooling_shapeCheck)(
                         THCState *state,
                         THCTensor *input, THCTensor *gradOutput, THCIndexTensor *indices,
                         int kH, int kW, int dH, int dW, int padH, int padW,
                         int dilationH, int dilationW, bool ceil_mode) {

  THArgCheck(kW > 0 && kH > 0, 5,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 8,
             "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);
  THArgCheck(dilationH > 0 && dilationW > 0, 12,
             "dilation should be greater than zero, but got dilationH: %d dilationW: %d",
             dilationH, dilationW);

  int ndim = input->dim();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;
  int batchSize = 1;

  if (ndim == 4) {
    batchSize = input->size(0);
    dimf++;
    dimh++;
    dimw++;
  }

  THCUNN_argCheck(state, !input->is_empty() && (ndim == 3 || ndim == 4), 2, input,
                  "non-empty 3D or 4D input tensor expected but got: %s");
  THArgCheck(kW/2 >= padW && kH/2 >= padH, 2,
             "pad should be smaller than half of kernel size, but got "
             "padW = %d, padH = %d, kW = %d, kH = %d",
             padW, padH, kW, kH);

  int64_t nInputPlane = input->size(dimh-1);
  int64_t nInputRows = input->size(dimh);
  int64_t nInputCols = input->size(dimw);
  int64_t nOutputPlane = nInputPlane;

  int64_t nOutputRows = pooling_output_shape<int64_t>(nInputRows, kH, padH, dH, dilationH, ceil_mode);
  int64_t nOutputCols = pooling_output_shape<int64_t>(nInputCols, kW, padW, dW, dilationW, ceil_mode);

  if (nOutputCols < 1 || nOutputRows < 1)
    THError("Given input size: (%dx%dx%d). "
            "Calculated output size: (%dx%dx%d). Output size is too small",
            nInputPlane,nInputRows,nInputCols,nInputPlane,nOutputRows,nOutputCols);

  if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, ndim, dimf, nOutputPlane);
    THCUNN_check_dim_size(state, gradOutput, ndim, dimh, nOutputRows);
    THCUNN_check_dim_size(state, gradOutput, ndim, dimw, nOutputCols);
  }
  if (indices != NULL) {
    THCUNN_check_dim_size_indices(state, indices, 4, 0, batchSize);
    THCUNN_check_dim_size_indices(state, indices, 4, 1, nOutputPlane);
    THCUNN_check_dim_size_indices(state, indices, 4, 2, nOutputRows);
    THCUNN_check_dim_size_indices(state, indices, 4, 3, nOutputCols);
  }
}

void THNN_(SpatialDilatedMaxPooling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCIndexTensor *indices,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH,
           int dilationW, int dilationH,
           bool ceil_mode)
{

  THCUNN_assertSameGPU(state, 3, input, output, indices);
  THNN_(SpatialDilatedMaxPooling_shapeCheck)
       (state, input, NULL, NULL, kH, kW, dH, dW,
        padH, padW, dilationH, dilationW, ceil_mode);

  int64_t nInputCols, nInputRows, nInputPlane, batchSize;
  int64_t nOutputCols, nOutputRows;

  if (input->dim() == 3) {
    nInputCols = input->size(2);
    nInputRows = input->size(1);
    nInputPlane = input->size(0);
    batchSize = 1;
  }
  else
  {
    nInputCols = input->size(3);
    nInputRows = input->size(2);
    nInputPlane = input->size(1);
    batchSize = input->size(0);
  }

  nOutputCols = pooling_output_shape<int64_t>(nInputCols, kW, padW, dW, dilationW, ceil_mode);
  nOutputRows = pooling_output_shape<int64_t>(nInputRows, kH, padH, dH, dilationH, ceil_mode);

  input = THCTensor_(newContiguous)(state, input);
  scalar_t* input_data = THCTensor_(data)(state, input);

  THCTensor_(resize4d)(state, output, batchSize, nInputPlane, nOutputRows, nOutputCols);
  THCUNN_resizeAs_indices(state, indices, output);

  THCIndex_t* indices_data = THCIndexTensor_(data)(state, indices);
  scalar_t* output_data = THCTensor_(data)(state, output);

  int count = THCTensor_(nElement)(state, output);

  MaxPoolForward<scalar_t, accreal> <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
      (count, input_data,
      batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
      kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
  THCudaCheck(cudaGetLastError());

  if(input->dim() == 3)
    THCTensor_(resize3d)(state, output, nInputPlane, nOutputRows, nOutputCols);

  THCTensor_(free)(state, input);
}

void THNN_(SpatialDilatedMaxPooling_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCIndexTensor *indices,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH,
           int dilationW, int dilationH,
           bool ceil_mode)
{
  THCUNN_assertSameGPU(state, 4, input, gradOutput, indices, gradInput);
  THNN_(SpatialDilatedMaxPooling_shapeCheck)
       (state, input, gradOutput, indices, kH, kW, dH, dW,
       padH, padW, dilationH, dilationW, ceil_mode);

  input = THCTensor_(newContiguous)(state, input);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  int64_t nInputCols, nInputRows, nInputPlane, batchSize;
  int64_t nOutputCols, nOutputRows;

  if (THTensor_nDimensionLegacyAll(input) == 3) {
    nInputCols = input->size(2);
    nInputRows = input->size(1);
    nInputPlane = input->size(0);
    batchSize = 1;
  }
  else
  {
    nInputCols = input->size(3);
    nInputRows = input->size(2);
    nInputPlane = input->size(1);
    batchSize = input->size(0);
  }

  nOutputCols = pooling_output_shape<int64_t>(nInputCols, kW, padW, dW, dilationW, ceil_mode);
  nOutputRows = pooling_output_shape<int64_t>(nInputRows, kH, padH, dH, dilationH, ceil_mode);

  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  THCTensor_(resizeAs)(state, gradInput, input);

  int count = THCTensor_(nElement)(state, input);
  dim3 grid;
  int imgcount = nInputCols * nInputRows;
  const int blocks = (imgcount + BACKWARD_THREADS - 1) / BACKWARD_THREADS;
  grid.x = blocks;
  grid.y = batchSize;
  grid.z = nInputPlane;
  uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
  uint64_t maxGridZ = at::cuda::getCurrentDeviceProperties()->maxGridSize[2];
  if (maxGridY < grid.y) grid.y = maxGridY;
  if (maxGridZ < grid.z) grid.z = maxGridZ;
  MaxPoolBackward<scalar_t, accreal> <<< grid, BACKWARD_THREADS, 0, THCState_getCurrentStream(state) >>>
      (count,
      THCTensor_(data)(state, gradOutput),
      THCIndexTensor_(data)(state, indices),
      batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
      kH, kW, dH, dW, padH, padW, dilationH, dilationW,
      THCTensor_(data)(state, gradInput));
  THCudaCheck(cudaGetLastError());

  THCTensor_(free)(state, gradOutput);

  // clean
  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
}

#endif
