#include "THCUNN.h"
#include "common.h"
#include "im2col.h"

void THNN_CudaSpatialConvolutionLocal_updateOutput(
    THCState *state,
    THCudaTensor *input,
    THCudaTensor *output,
    THCudaTensor *weight,
    THCudaTensor *bias,
    THCudaTensor *finput,
    THCudaTensor *fgradInput,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    long inputWidth, long inputHeight,
    long outputWidth, long outputHeight)
{
  THCUNN_assertSameGPU(state, 5, input, output, weight,
                                 bias, finput);

  long nInputPlane = THCudaTensor_size(state,weight,2)/(kW*kH);
  long nOutputPlane = THCudaTensor_size(state,weight,1);

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(state, input, 1, nInputPlane, inputHeight, inputWidth);
  }

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCudaTensor_resize4d(state, output, batchSize, nOutputPlane, outputHeight, outputWidth);

  // Augment the input
  THCudaTensor_resize3d(state, finput, batchSize, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Helpers
  THCudaTensor *input_n = THCudaTensor_new(state);
  THCudaTensor *finput_n = THCudaTensor_new(state);
  THCudaTensor *output_n = THCudaTensor_new(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    THCudaTensor *finput3d, *output3d;
    THCudaTensor *wslice = THCudaTensor_new(state);
    THCudaTensor *islice = THCudaTensor_new(state);
    THCudaTensor *oslice = THCudaTensor_new(state);

    // Matrix mulitply per output:
    THCudaTensor_select(state, input_n, input, 0, elt);
    THCudaTensor_select(state, finput_n, finput, 0, elt);
    THCudaTensor_select(state, output_n, output, 0, elt);

    // Extract columns:
    im2col(
      THCState_getCurrentStream(state),
      THCudaTensor_data(state, input_n),
      nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
      THCudaTensor_data(state, finput_n)
    );

    output3d = THCudaTensor_newWithStorage3d(state, output_n->storage, output_n->storageOffset,
                                             outputHeight*outputWidth, 1,
                                             nOutputPlane, outputHeight*outputWidth,
                                             1, nOutputPlane*outputHeight*outputWidth);

    finput3d = THCudaTensor_newWithStorage3d(state, finput_n->storage, finput_n->storageOffset,
                                             outputHeight*outputWidth, 1,
                                             kW*kH*nInputPlane, outputHeight*outputWidth,
                                             1, kW*kH*nInputPlane*outputHeight*outputWidth);

    THCudaTensor_copy(state, output_n, bias);

    for (int i = 0; i < outputHeight; i++) {
      for(int j = 0; j < outputWidth; j++) {
        int sliceidx = i * outputWidth + j;
        THCudaTensor_select(state, wslice, weight, 0, sliceidx);
        THCudaTensor_select(state, islice, finput3d, 0, sliceidx);
        THCudaTensor_select(state, oslice, output3d, 0, sliceidx);
        THCudaTensor_addmm(state, oslice, 1.0, oslice, 1.0, wslice, islice);
      }
    }


    // weight:    oH*oW x nOutputPlane x nInputPlane*kH*kW
    // finput3d:  oH*oW x nInputPlane*kH*kW x 1
    // THCudaTensor_baddbmm(state, output3d, 1.0, output3d, 1.0, weight, finput3d);
    // output3d:  oH*oW x nOutputPlane x 1

    THCudaTensor_free(state, output3d);
    THCudaTensor_free(state, finput3d);
    THCudaTensor_free(state, wslice);
    THCudaTensor_free(state, islice);
    THCudaTensor_free(state, oslice);
  }

  // Free
  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, finput_n);
  THCudaTensor_free(state, output_n);

  // Resize output
  if (batch == 0) {
    THCudaTensor_resize3d(state, output, nOutputPlane, outputHeight, outputWidth);
    THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
  }
}

void THNN_CudaSpatialConvolutionLocal_updateGradInput(
    THCState *state,
    THCudaTensor *input,
    THCudaTensor *gradOutput,
    THCudaTensor *gradInput,
    THCudaTensor *weight,
    THCudaTensor *finput,
    THCudaTensor *fgradInput,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    long inputWidth, long inputHeight,
    long outputWidth, long outputHeight)
{
  THCUNN_assertSameGPU(state, 5, input, gradOutput, weight,
                                 fgradInput, gradInput);

  long nInputPlane = THCudaTensor_size(state,weight,2)/(kW*kH);
  long nOutputPlane = THCudaTensor_size(state,weight,1);

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(state, input, 1, nInputPlane, inputHeight, inputWidth);
    THCudaTensor_resize4d(state, gradOutput, 1, nOutputPlane, outputHeight, outputWidth);
  }

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCudaTensor_resize4d(state, gradInput, batchSize, nInputPlane, inputHeight, inputWidth);

  // Resize temporary columns
  THCudaTensor_resize3d(state, fgradInput, batchSize, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Helpers
  THCudaTensor *gradInput_n = THCudaTensor_new(state);
  THCudaTensor *fgradInput_n = THCudaTensor_new(state);
  THCudaTensor *gradOutput_n = THCudaTensor_new(state);

  THCudaTensor_transpose(state, weight, weight, 1, 2);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    THCudaTensor *gradOutput3d, *fgradInput3d;
    THCudaTensor *wslice = THCudaTensor_new(state);
    THCudaTensor *gislice = THCudaTensor_new(state);
    THCudaTensor *goslice = THCudaTensor_new(state);

    // Matrix mulitply per sample:
    THCudaTensor_select(state, gradInput_n, gradInput, 0, elt);
    THCudaTensor_select(state, fgradInput_n, fgradInput, 0, elt);
    THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);

    gradOutput3d = THCudaTensor_newWithStorage3d(state, gradOutput_n->storage, gradOutput_n->storageOffset,
                                               outputHeight*outputWidth, 1,
                                               nOutputPlane, outputHeight*outputWidth,
                                               1, nOutputPlane*outputHeight*outputWidth);
    fgradInput3d = THCudaTensor_newWithStorage3d(state, fgradInput_n->storage, fgradInput_n->storageOffset,
                                               outputHeight*outputWidth, 1,
                                               kW*kH*nInputPlane, outputHeight*outputWidth,
                                               1, kW*kH*nInputPlane*outputHeight*outputWidth);

    for (int i = 0; i < outputHeight; i++) {
      for(int j = 0; j < outputWidth; j++) {
        int sliceidx = i * outputWidth + j;
        THCudaTensor_select(state, wslice, weight, 0, sliceidx);
        THCudaTensor_select(state, gislice, fgradInput3d, 0, sliceidx);
        THCudaTensor_select(state, goslice, gradOutput3d, 0, sliceidx);
        THCudaTensor_addmm(state, gislice, 0.0, gislice, 1.0, wslice, goslice);
      }
    }

    // weight:        oH*oW x nInputPlane*kH*kW x nOutputPlane
    // gradOutput3d:  oH*oW x nOutputPlane x 1
    //THCudaTensor_baddbmm(state, fgradInput3d, 0.0, fgradInput3d, 1.0, weight, gradOutput3d);
    // fgradInput3d:  oH*oW x nInputPlane*kH*kW x 1

    // Unpack columns back into input:
    col2im(
      THCState_getCurrentStream(state),
      THCudaTensor_data(state, fgradInput_n),
      nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
      THCudaTensor_data(state, gradInput_n)
    );

    THCudaTensor_free(state, gradOutput3d);
    THCudaTensor_free(state, fgradInput3d);
    THCudaTensor_free(state, wslice);
    THCudaTensor_free(state, gislice);
    THCudaTensor_free(state, goslice);
  }

  // Free
  THCudaTensor_free(state, gradInput_n);
  THCudaTensor_free(state, fgradInput_n);
  THCudaTensor_free(state, gradOutput_n);

  // Resize output
  if (batch == 0) {
    THCudaTensor_resize3d(state, gradOutput, nOutputPlane, outputHeight, outputWidth);
    THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
    THCudaTensor_resize3d(state, gradInput, nInputPlane, inputHeight, inputWidth);
  }

  THCudaTensor_transpose(state, weight, weight, 1, 2);
}

void THNN_CudaSpatialConvolutionLocal_accGradParameters(
    THCState *state,
    THCudaTensor *input,
    THCudaTensor *gradOutput,
    THCudaTensor *gradWeight,
    THCudaTensor *gradBias,
    THCudaTensor *finput,
    THCudaTensor *fgradInput,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    long inputWidth, long inputHeight,
    long outputWidth, long outputHeight,
    float scale)
{
  THCUNN_assertSameGPU(state, 5, input, gradOutput, gradWeight,
                                 gradBias, finput);

  long nInputPlane = THCudaTensor_size(state,gradWeight,2)/(kW*kH);
  long nOutputPlane = THCudaTensor_size(state,gradWeight,1);

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(state, input, 1, nInputPlane, inputHeight, inputWidth);
    THCudaTensor_resize4d(state, gradOutput, 1, nOutputPlane, outputHeight, outputWidth);
  }

  // Batch size + input planes
  long batchSize = input->size[0];

  // Helpers
  THCudaTensor *input_n = THCudaTensor_new(state);
  THCudaTensor *finput_n = THCudaTensor_new(state);
  THCudaTensor *gradOutput_n = THCudaTensor_new(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    THCudaTensor *gradOutput3d, *finput3d;
    THCudaTensor *gwslice = THCudaTensor_new(state);
    THCudaTensor *islice = THCudaTensor_new(state);
    THCudaTensor *goslice = THCudaTensor_new(state);

    // Matrix mulitply per output:
    THCudaTensor_select(state, input_n, input, 0, elt);
    THCudaTensor_select(state, finput_n, finput, 0, elt);
    THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);

    gradOutput3d = THCudaTensor_newWithStorage3d(state, gradOutput_n->storage, gradOutput_n->storageOffset,
                                                 outputHeight*outputWidth, 1,
                                                 nOutputPlane, outputHeight*outputWidth,
                                                 1, nOutputPlane*outputHeight*outputWidth);
    finput3d = THCudaTensor_newWithStorage3d(state, finput_n->storage, finput_n->storageOffset,
                                             outputHeight*outputWidth, 1,
                                             1, kW*kH*nInputPlane*outputHeight*outputWidth,
                                             kW*kH*nInputPlane, outputHeight*outputWidth);

    // Extract columns:
    im2col(
      THCState_getCurrentStream(state),
      THCudaTensor_data(state, input_n),
      nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
      THCudaTensor_data(state, finput_n)
    );

    for (int i = 0; i < outputHeight; i++) {
      for(int j = 0; j < outputWidth; j++) {
        int sliceidx = i * outputWidth + j;
        THCudaTensor_select(state, gwslice, gradWeight, 0, sliceidx);
        THCudaTensor_select(state, goslice, gradOutput3d, 0, sliceidx);
        THCudaTensor_select(state, islice, finput3d, 0, sliceidx);
        THCudaTensor_addmm(state, gwslice, 1.0, gwslice, scale, goslice, islice);
      }
    }
    // gradOutput3d:  oH*oW x nOutputPlane x 1
    // finput3d:      oH*oW x 1 x kW*kH*nInputPlane
    //THCudaTensor_baddbmm(state, gradWeight, 1.0, gradWeight, scale, gradOutput3d, finput3d);
    // gradWeight:    oH*oW x nOutputPlane x kW*kH*nInputPlane

    THCudaTensor_cadd(state, gradBias, gradBias, scale, gradOutput_n);

    THCudaTensor_free(state, gradOutput3d);
    THCudaTensor_free(state, finput3d);
    THCudaTensor_free(state, gwslice);
    THCudaTensor_free(state, goslice);
    THCudaTensor_free(state, islice);
  }

  // Free
  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, finput_n);
  THCudaTensor_free(state, gradOutput_n);

  // Resize
  if (batch == 0) {
    THCudaTensor_resize3d(state, gradOutput, nOutputPlane, outputHeight, outputWidth);
    THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
  }
}
