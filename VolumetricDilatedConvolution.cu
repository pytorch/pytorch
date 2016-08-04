#include "THCUNN.h"
#include "common.h"
#include "vol2col.h"


void THNN_CudaVolumetricDilatedConvolution_updateOutput(
  THCState *state,
  THCudaTensor *input,
  THCudaTensor *output,
  THCudaTensor *weight,
  THCudaTensor *bias,
  THCudaTensor *columns,
  THCudaTensor *ones,
  int kT, int kW, int kH,
  int dT, int dW, int dH,
  int padT, int padW, int padH,
  int dilationT, int dilationW, int dilationH) {

  THCUNN_assertSameGPU(state, 5, input, output, weight, columns, ones);
  if (bias) {
    THCUNN_assertSameGPU(state, 2, weight, bias);
  }
  THArgCheck(input->nDimension == 4 || input->nDimension == 5, 2, "4D or 5D (batch mode) tensor is expected, but got: %d", input->nDimension);
  THArgCheck(weight->nDimension == 5, 4, "weight tensor must be 5D (nOutputPlane,nInputPlane,kT,kH,kW)");
  THArgCheck(!bias || weight->size[0] == bias->size[0], 4, "nOutputPlane mismatch in weight and bias");
  THArgCheck(kT > 0 && kW > 0 && kH > 0, 8, "kernel size should be greater than zero");
  THArgCheck(dT > 0 && dW > 0 && dH > 0, 10, "stride should be greater than zero");

  // Params:
  int nInputPlane = weight->size[1];
  int nOutputPlane = weight->size[0];

  int batch = 1;
  if (input->nDimension == 4) {
    THArgCheck(input->size[0] == nInputPlane, 2, "input channels and nInputPlane dont match");
    // Force batch
    batch = 0;
    THCudaTensor_resize5d(state, input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
  } else {
    THArgCheck(input->size[1] == nInputPlane, 2, "input channels and nInputPlane dont match");
  }

  long inputDepth  = input->size[2];
  long inputHeight  = input->size[3];
  long inputWidth   = input->size[4];
  long outputDepth  = (inputDepth  + 2*padT - (dilationT * (kT - 1) + 1)) / dT + 1;
  long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  long outputWidth  = (inputWidth  + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;

  if (outputDepth < 1 || outputWidth < 1 || outputHeight < 1)
    THError("Given input size: (%dx%dx%dx%d). Calculated output size: (%dx%dx%dx%d). Output size is too small",
            nInputPlane,inputDepth,inputHeight,inputWidth,nOutputPlane,outputDepth,outputHeight,outputWidth);

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCudaTensor_resize5d(state, output, batchSize, nOutputPlane, outputDepth, outputHeight, outputWidth);

  // Resize temporary columns
  THCudaTensor_resize2d(state, columns, nInputPlane*kT*kW*kH, outputDepth*outputHeight*outputWidth);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1]*ones->size[2] < outputDepth*outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THCudaTensor_resize3d(state, ones, outputDepth, outputHeight, outputWidth);
    THCudaTensor_fill(state, ones, 1);
  }

  // Helpers
  THCudaTensor *input_n = THCudaTensor_new(state);
  THCudaTensor *output_n = THCudaTensor_new(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THCudaTensor_select(state, input_n, input, 0, elt);
    THCudaTensor_select(state, output_n, output, 0, elt);

    // Do Bias first:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m_ = nOutputPlane;
    long n_ = outputDepth * outputHeight * outputWidth;
    long k_ = 1;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    if (bias) {
      THCudaBlas_Sgemm(
          state,
          't', 'n',
          n_, m_, k_,
          1,
          THCudaTensor_data(state, ones), k_,
          THCudaTensor_data(state, bias), k_,
          0,
          THCudaTensor_data(state, output_n), n_
      );
    } else {
      THCudaTensor_zero(state, output_n);
    }

    // Extract columns:
    vol2col(
      THCState_getCurrentStream(state),
      THCudaTensor_data(state, input_n),
      nInputPlane, inputDepth, inputHeight, inputWidth,
      kT, kH, kW, padT, padH, padW, dT, dH, dW,
      dilationT, dilationH, dilationW,
      THCudaTensor_data(state, columns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = nOutputPlane;
    long n = columns->size[1];
    long k = nInputPlane*kT*kH*kW;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THCudaBlas_Sgemm(
        state,
        'n', 'n',
        n, m, k,
        1,
        THCudaTensor_data(state, columns), n,
        THCudaTensor_data(state, weight), k,
        1,
        THCudaTensor_data(state, output_n), n
    );
  }

  // Free
  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, output_n);

  // Resize output
  if (batch == 0) {
    THCudaTensor_resize4d(state, output, nOutputPlane, outputDepth, outputHeight, outputWidth);
    THCudaTensor_resize4d(state, input, nInputPlane, inputDepth, inputHeight, inputWidth);
  }
}

void THNN_CudaVolumetricDilatedConvolution_updateGradInput(
  THCState *state,
  THCudaTensor *input,
  THCudaTensor *gradOutput,
  THCudaTensor *gradInput,
  THCudaTensor *weight,
  THCudaTensor *gradColumns,
  int kT, int kW, int kH,
  int dT, int dW, int dH,
  int padT, int padW, int padH,
  int dilationT, int dilationW, int dilationH) {

  THCUNN_assertSameGPU(state, 5, input, gradOutput, weight,
                                 gradColumns, gradInput);
  THArgCheck(input->nDimension == 4 || input->nDimension == 5, 2, "4D or 5D (batch mode) tensor is expected");
  THArgCheck(gradOutput->nDimension == 4 || gradOutput->nDimension == 5, 3, "4D or 5D (batch mode) tensor is expected");
  THArgCheck(weight->nDimension == 5, 4, "weight tensor must be 5D (nOutputPlane,nInputPlane,kT,kH,kW)");
  THArgCheck(kT > 0 && kW > 0 && kH > 0, 8, "kernel size should be greater than zero");
  THArgCheck(dT > 0 && dW > 0 && dH > 0, 10, "stride should be greater than zero");

  // Params
  int nInputPlane = weight->size[1];
  int nOutputPlane = weight->size[0];

  int batch = 1;
  if (input->nDimension == 4) {
    // Force batch
    batch = 0;
    THCudaTensor_resize5d(state, input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
    THCudaTensor_resize5d(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  long inputDepth  = input->size[2];
  long inputWidth   = input->size[4];
  long inputHeight  = input->size[3];
  long outputDepth  = (inputDepth + 2*padT - (dilationT * (kT - 1) + 1)) / dT + 1;
  long outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCudaTensor_resize5d(state, gradInput, batchSize, nInputPlane, inputDepth, inputHeight, inputWidth);

  // Resize temporary columns
  THCudaTensor_resize2d(state, gradColumns, nInputPlane*kT*kW*kH, outputDepth*outputHeight*outputWidth);

  // Helpers
  THCudaTensor *gradInput_n = THCudaTensor_new(state);
  THCudaTensor *gradOutput_n = THCudaTensor_new(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per sample:
    THCudaTensor_select(state, gradInput_n, gradInput, 0, elt);
    THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = nInputPlane*kT*kW*kH;
    long n = gradColumns->size[1];
    long k = nOutputPlane;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THCudaBlas_Sgemm(
        state,
        'n', 't',
        n, m, k,
        1,
        THCudaTensor_data(state, gradOutput_n), n,
        THCudaTensor_data(state, weight), m,
        0,
        THCudaTensor_data(state, gradColumns), n
    );

    // Unpack columns back into input:
    col2vol(
      THCState_getCurrentStream(state),
      THCudaTensor_data(state, gradColumns),
      nInputPlane, inputDepth, inputHeight, inputWidth,
      kT, kH, kW, padT, padH, padW, dT, dH, dW,
      dilationT, dilationH, dilationW,
      THCudaTensor_data(state, gradInput_n)
    );
  }

  // Free
  THCudaTensor_free(state, gradInput_n);
  THCudaTensor_free(state, gradOutput_n);

  // Resize output
  if (batch == 0) {
    THCudaTensor_resize4d(state, gradOutput, nOutputPlane, outputDepth, outputHeight, outputWidth);
    THCudaTensor_resize4d(state, input, nInputPlane, inputDepth, inputHeight, inputWidth);
    THCudaTensor_resize4d(state, gradInput, nInputPlane, inputDepth, inputHeight, inputWidth);
  }
}

void THNN_CudaVolumetricDilatedConvolution_accGradParameters(
  THCState *state,
  THCudaTensor *input,
  THCudaTensor *gradOutput,
  THCudaTensor *gradWeight,
  THCudaTensor *gradBias,
  THCudaTensor *columns,
  THCudaTensor *ones,
  int kT, int kW, int kH,
  int dT, int dW, int dH,
  int padT, int padW, int padH,
  int dilationT, int dilationW, int dilationH,
  float scale) {

  THCUNN_assertSameGPU(state, 5, input, gradOutput, gradWeight, columns, ones);
  if (gradBias) {
   THCUNN_assertSameGPU(state, 2, gradWeight, gradBias);
  }
  THArgCheck(input->nDimension == 4 || input->nDimension == 5, 2, "4D or 5D (batch mode) tensor is expected");
  THArgCheck(gradOutput->nDimension == 4 || gradOutput->nDimension == 5, 3, "4D or 5D (batch mode) tensor is expected");
  THArgCheck(gradWeight->nDimension == 5, 4, "gradWeight tensor must be 5D (nOutputPlane,nInputPlane,kT,kH,kW)");
  THArgCheck(kT > 0 && kW > 0 && kH > 0, 8, "kernel size should be greater than zero");
  THArgCheck(dT > 0 && dW > 0 && dH > 0, 10, "stride should be greater than zero");

  // Params
  int nInputPlane = gradWeight->size[1];
  int nOutputPlane = gradWeight->size[0];

  int batch = 1;
  if (input->nDimension == 4) {
    // Force batch
    batch = 0;
    THCudaTensor_resize5d(state, input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
    THCudaTensor_resize5d(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  long inputDepth  = input->size[2];
  long inputWidth   = input->size[4];
  long inputHeight  = input->size[3];
  long outputDepth  = (inputDepth + 2*padT - (dilationT * (kT - 1) + 1)) / dT + 1;
  long outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 3 || ones->size[0]*ones->size[1]*ones->size[2] < outputDepth*outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THCudaTensor_resize3d(state, ones, outputDepth, outputHeight, outputWidth);
    THCudaTensor_fill(state, ones, 1);
  }

  // Resize temporary columns
  THCudaTensor_resize2d(state, columns, nInputPlane*kT*kW*kH, outputDepth*outputHeight*outputWidth);

  // Helpers
  THCudaTensor *input_n = THCudaTensor_new(state);
  THCudaTensor *gradOutput_n = THCudaTensor_new(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THCudaTensor_select(state, input_n, input, 0, elt);
    THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    vol2col(
      THCState_getCurrentStream(state),
      THCudaTensor_data(state, input_n),
      nInputPlane, inputDepth, inputHeight, inputWidth, kT, kH, kW, padT, padH, padW, dT, dH, dW,
      dilationT, dilationH, dilationW,
      THCudaTensor_data(state, columns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = nOutputPlane;
    long n = nInputPlane*kT*kW*kH;
    long k = columns->size[1];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THCudaBlas_Sgemm(
        state,
        't', 'n',
        n, m, k,
        scale,
        THCudaTensor_data(state, columns), k,
        THCudaTensor_data(state, gradOutput_n), k,
        1,
        THCudaTensor_data(state, gradWeight), n
    );

    // Do Bias:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m_ = nOutputPlane;
    long k_ = outputDepth * outputHeight * outputWidth;

    // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
    if (gradBias) {
      THCudaBlas_Sgemv(
          state,
          't',
          k_, m_,
          scale,
          THCudaTensor_data(state, gradOutput_n), k_,
          THCudaTensor_data(state, ones), 1,
          1,
          THCudaTensor_data(state, gradBias), 1
      );
    }
  }

  // Free
  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, gradOutput_n);

    // Resize output
  if (batch == 0) {
    THCudaTensor_resize4d(state, gradOutput, nOutputPlane, outputDepth, outputHeight, outputWidth);
    THCudaTensor_resize4d(state, input, nInputPlane, inputDepth, inputHeight, inputWidth);
  }
}
