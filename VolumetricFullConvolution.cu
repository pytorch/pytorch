#include "THCUNN.h"
#include "common.h"
#include "vol2col.h"


void THNN_CudaVolumetricFullConvolution_updateOutput(
    THCState *state,
    THCudaTensor *input,
    THCudaTensor *output,
    THCudaTensor *weight,
    THCudaTensor *bias,
    THCudaTensor *finput,
    THCudaTensor *fgradInput,
    int dT, int dW, int dH,
    int padT, int padW, int padH,
    int adjT, int adjW, int adjH)
{

  THCudaTensor *columns = finput;
  THCudaTensor *ones    = fgradInput;

  int nInputPlane = THCudaTensor_size(state, weight, 0);
  int nOutputPlane = THCudaTensor_size(state, weight, 1);
  const int kT           = (int)weight->size[2];
  const int kH           = (int)weight->size[3];
  const int kW           = (int)weight->size[4];

  THCUNN_assertSameGPU(state, 6, input, output, weight,
                                 bias, columns, ones);
  THArgCheck(input->nDimension == 4 || input->nDimension == 5, 2, "4D or 5D (batch mode) tensor is expected");

  int batch = 1;
  if (input->nDimension == 4) {
    THArgCheck(input->size[0] == nInputPlane, 2, "input channels and nInputPlane dont match");
    // Force batch
    batch = 0;
    THCudaTensor_resize5d(state, input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
  } else {
    THArgCheck(input->size[1] == nInputPlane, 2, "input channels and nInputPlane dont match");
  }

  long inputWidth   = input->size[4];
  long inputHeight  = input->size[3];
  long inputDepth  = input->size[2];
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + kW + adjW;
  long outputHeight = (inputHeight - 1) * dH - 2*padH + kH + adjH;
  long outputDepth = (inputDepth - 1) * dT - 2*padT + kT + adjT;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCudaTensor_resize5d(state, output, batchSize, nOutputPlane, outputDepth, outputHeight, outputWidth);

  // Resize temporary columns
  THCudaTensor_resize2d(state, columns, nOutputPlane*kW*kH*kT, inputDepth*inputHeight*inputWidth);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  if (ones->nDimension != 3 || ones->size[0]*ones->size[1]*ones->size[2] < outputDepth*outputHeight*outputWidth) {
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

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = weight->size[1] * weight->size[2] * weight->size[3] * weight->size[4];
    long n = columns->size[1];
    long k = weight->size[0];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THCudaBlas_gemm(
        state,
        'n', 't',
        n, m, k,
        1,
        THCudaTensor_data(state, input_n), n,
        THCudaTensor_data(state, weight), m,
        0,
        THCudaTensor_data(state, columns), n
    );

    // Unpack columns back into input:
    col2vol(
      THCState_getCurrentStream(state),
      THCudaTensor_data(state, columns),
      nOutputPlane, outputDepth, outputHeight, outputWidth, kT, kH, kW, padT, padH, padW, dT, dH, dW,
      THCudaTensor_data(state, output_n)
    );

    // Do Bias after:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m_ = nOutputPlane;
    long n_ = outputDepth * outputHeight * outputWidth;
    long k_ = 1;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THCudaBlas_gemm(
        state,
        't', 'n',
        n_, m_, k_,
        1,
        THCudaTensor_data(state, ones), k_,
        THCudaTensor_data(state, bias), k_,
        1,
        THCudaTensor_data(state, output_n), n_
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

void THNN_CudaVolumetricFullConvolution_updateGradInput(
    THCState *state,
    THCudaTensor *input,
    THCudaTensor *gradOutput,
    THCudaTensor *gradInput,
    THCudaTensor *weight,
    THCudaTensor *finput,
    THCudaTensor *fgradInput,
    int dT, int dW, int dH,
    int padT, int padW, int padH,
    int adjT, int adjW, int adjH)
{
  THCudaTensor *gradColumns = finput;

  int nInputPlane = THCudaTensor_size(state, weight, 0);
  int nOutputPlane = THCudaTensor_size(state, weight, 1);
  const int kT           = (int)weight->size[2];
  const int kH           = (int)weight->size[3];
  const int kW           = (int)weight->size[4];

  THCUNN_assertSameGPU(state, 5, input, gradOutput, weight,
                                 gradColumns, gradInput);
  THArgCheck(input->nDimension == 4 || input->nDimension == 5, 2, "4D or 5D (batch mode) tensor is expected");

  int batch = 1;
  if (input->nDimension == 4) {
    // Force batch
    batch = 0;
    THCudaTensor_resize5d(state, input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
    THCudaTensor_resize5d(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  long inputWidth   = input->size[4];
  long inputHeight  = input->size[3];
  long inputDepth   = input->size[2];
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + kW + adjW;
  long outputHeight = (inputHeight - 1) * dH - 2*padH + kH + adjH;
  long outputDepth = (inputDepth - 1) * dT - 2*padT + kT + adjT;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCudaTensor_resize5d(state, gradInput, batchSize, nInputPlane, inputDepth, inputHeight, inputWidth);

  // Resize temporary columns
  THCudaTensor_resize2d(state, gradColumns, nOutputPlane*kW*kH*kT, inputDepth*inputHeight*inputWidth);

  // Helpers
  THCudaTensor *gradInput_n = THCudaTensor_new(state);
  THCudaTensor *gradOutput_n = THCudaTensor_new(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per sample:
    THCudaTensor_select(state, gradInput_n, gradInput, 0, elt);
    THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    vol2col(
      THCState_getCurrentStream(state),
      THCudaTensor_data(state, gradOutput_n),
      nOutputPlane, outputDepth, outputHeight, outputWidth, kT, kH, kW, padT, padH, padW, dT, dH, dW,
      THCudaTensor_data(state, gradColumns)
    );


    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = weight->size[0];
    long n = gradColumns->size[1];
    long k = weight->size[1] * weight->size[2] * weight->size[3] * weight->size[4];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THCudaBlas_gemm(
        state,
        'n', 'n',
        n, m, k,
        1,
        THCudaTensor_data(state, gradColumns), n,
        THCudaTensor_data(state, weight), k,
        0,
        THCudaTensor_data(state, gradInput_n), n
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


void THNN_CudaVolumetricFullConvolution_accGradParameters(
    THCState *state,
    THCudaTensor *input,
    THCudaTensor *gradOutput,
    THCudaTensor *gradWeight,
    THCudaTensor *gradBias,
    THCudaTensor *finput,
    THCudaTensor *fgradInput,
    int dT, int dW, int dH,
    int padT, int padW, int padH,
    int adjT, int adjW, int adjH,
    float scale)
{
  THCudaTensor *columns = finput;
  THCudaTensor *ones = fgradInput;

  int nInputPlane = THCudaTensor_size(state, gradWeight, 0);
  int nOutputPlane = THCudaTensor_size(state, gradWeight, 1);
  const int kT           = (int)gradWeight->size[2];
  const int kH           = (int)gradWeight->size[3];
  const int kW           = (int)gradWeight->size[4];

  THCUNN_assertSameGPU(state, 6, input, gradOutput, gradWeight,
                                 gradBias, columns, ones);
  THArgCheck(input->nDimension == 4 || input->nDimension == 5, 2, "4D or 5D (batch mode) tensor is expected");

  int batch = 1;
  if (input->nDimension == 4) {
    // Force batch
    batch = 0;
    THCudaTensor_resize5d(state, input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
    THCudaTensor_resize5d(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  long inputWidth   = input->size[4];
  long inputHeight  = input->size[3];
  long inputDepth   = input->size[2];
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + kW + adjW;
  long outputHeight = (inputHeight - 1) * dH - 2*padH + kH + adjH;
  long outputDepth  = (inputDepth - 1) * dT - 2*padT + kT + adjT;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 3 || ones->size[0]*ones->size[1]*ones->size[2] < outputDepth*outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THCudaTensor_resize3d(state, ones, outputDepth, outputHeight, outputWidth);
    THCudaTensor_fill(state, ones, 1);
  }

  // Resize temporary columns
  THCudaTensor_resize2d(state, columns, nOutputPlane*kW*kH*kT, inputDepth*inputHeight*inputWidth);

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
      THCudaTensor_data(state, gradOutput_n),
      nOutputPlane, outputDepth, outputHeight, outputWidth, kT, kH, kW, padT, padH, padW, dT, dH, dW,
      THCudaTensor_data(state, columns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long n = columns->size[0];   // nOutputPlane * kt * kh * kw
    long m = input_n->size[0];   // nInputPlane
    long k = columns->size[1];   // inputHeight * inputWidth

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THCudaBlas_gemm(
        state,
        't', 'n',
        n, m, k,
        scale,
        THCudaTensor_data(state, columns), k,
        THCudaTensor_data(state, input_n), k,
        1,
        THCudaTensor_data(state, gradWeight), n
    );

    // Do Bias:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m_ = nOutputPlane;
    long k_ = outputDepth * outputHeight * outputWidth;

    // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
    THCudaBlas_gemv(
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

  // Free
  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, gradOutput_n);

  // Resize
  if (batch == 0) {
    THCudaTensor_resize4d(state, gradOutput, nOutputPlane, outputDepth, outputHeight, outputWidth);
    THCudaTensor_resize4d(state, input, nInputPlane, inputDepth, inputHeight, inputWidth);
  }
}
