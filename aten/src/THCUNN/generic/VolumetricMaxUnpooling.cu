#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/VolumetricMaxUnpooling.cu"
#else

static inline void THNN_(VolumetricMaxUnpooling_shapeCheck)(
                         THCState *state,
                         THCTensor *input,
                         THCTensor *gradOutput,
                         THCIndexTensor *indices,
                         int oT,
                         int oW,
                         int oH,
                         int dT,
                         int dW,
                         int dH,
                         int pT,
                         int pW,
                         int pH) {
  int inputSlices = 0;

  THCUNN_check_shape_indices(state, indices, input);

  THArgCheck(dT > 0 && dW > 0 && dH > 0, 10,
             "stride should be greater than zero, but got dT: %d dH: %d dW: %d",
             dT, dH, dW);

  if (THCTensor_(nDimensionLegacyNoScalars)(state, input) == 4)
  {
    inputSlices = THCTensor_(size)(state, input, 0);
  }
  else if (THCTensor_(nDimensionLegacyNoScalars)(state, input) == 5)
  {
    inputSlices = THCTensor_(size)(state, input, 1);
  }
  else
  {
    AT_ERROR("non-empty 4D or 5D tensor expected, got size: ",
             input->sizes());
  }

  int dimw = 3;
  int dimh = 2;
  int dimt = 1;
  int dimn = 0;
  if (input->dim() == 5)
  {
    dimt++;
    dimw++;
    dimh++;
    dimn++;
  }

  if (gradOutput != NULL) {
    if (oT != gradOutput->size(dimt) || oW != gradOutput->size(dimw) || oH != gradOutput->size(dimh))
    {
      THError(
        "Inconsistent gradOutput size. oT= %d, oH= %d, oW= %d, gradOutput: %dx%dx%d",
        oT, oH, oW, gradOutput->size(dimt), gradOutput->size(dimh), gradOutput->size(dimw));
    }

    THCUNN_check_dim_size(state, gradOutput, input->dim(), dimn, inputSlices);
  }
}

void THNN_(VolumetricMaxUnpooling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCIndexTensor *indices,
           int outputTime, int outputWidth, int outputHeight,
           int dT, int dW, int dH,
           int padT, int padW, int padH)
{
  AT_ERROR("Deprecated");
}

void THNN_(VolumetricMaxUnpooling_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCIndexTensor *indices,
           int outputTime, int outputWidth, int outputHeight,
           int dT, int dW, int dH,
           int padT, int padW, int padH)
{
  int batchSize = 0;
  int inputSlices = 0;
  int inputTime = 0;
  int inputHeight = 0;
  int inputWidth = 0;

  THNN_(VolumetricMaxUnpooling_shapeCheck)(
        state, input, gradOutput, indices,
        outputTime, outputWidth, outputHeight,
        dT, dW, dH, padT, padW, padH);
  THCUNN_assertSameGPU(state, 4, input, indices, gradOutput, gradInput);

  int fiveDimensionalInput = THCTensor_(nDimensionLegacyNoScalars)(state, input) == 5;
  if (!fiveDimensionalInput) /* 4D */
  {
    batchSize = 1;
    inputSlices  = THCTensor_(size)(state, input, 0);
    inputTime   = THCTensor_(size)(state, input, 1);
    inputHeight = THCTensor_(size)(state, input, 2);
    inputWidth  = THCTensor_(size)(state, input, 3);
  }
  else
  {
    batchSize    = THCTensor_(size)(state, input, 0);
    inputSlices  = THCTensor_(size)(state, input, 1);
    inputTime   = THCTensor_(size)(state, input, 2);
    inputHeight = THCTensor_(size)(state, input, 3);
    inputWidth  = THCTensor_(size)(state, input, 4);
  }

  input = THCTensor_(newContiguous)(state, input);
  THCTensor_(resizeAs)(state, gradInput, input);
  THCTensor_(zero)(state, gradInput);
  indices = THCIndexTensor_(newContiguous)(state, indices);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  // Collapse batch and feature dimensions
  if (fiveDimensionalInput) {
    gradInput = THCTensor_(newFoldBatchDim)(state, gradInput);

    THCIndexTensor *old_indices = indices;
    indices = THCIndexTensor_(newFoldBatchDim)(state, indices);
    THCIndexTensor_(free)(state, old_indices);

    THCTensor *old_gradOutput = gradOutput;
    gradOutput = THCTensor_(newFoldBatchDim)(state, gradOutput);
    THCTensor_(free)(state, old_gradOutput);
  } else {
    THCTensor_(retain)(state, gradInput);
  }

  scalar_t* gradOutputData = THCTensor_(data)(state, gradOutput);

  THCDeviceTensor<scalar_t, 4> cudaGradInput;
  THCDeviceTensor<THCIndex_t, 4> cudaIndices;

  cudaGradInput  = toDeviceTensor<scalar_t, 4>(state, gradInput);
  cudaIndices = toDeviceTensor<THCIndex_t, 4>(state, indices);

  int totalZ = inputTime * inputSlices * batchSize;
  int offsetZ = 0;
  dim3 block(32, 8);

  while (totalZ > 0) {
    dim3 grid(THCCeilDiv(inputWidth, static_cast<int>(block.x)),
              THCCeilDiv(inputHeight, static_cast<int>(block.y)),
              totalZ > 65535 ? 65535 : totalZ);

    cuda_VolumetricMaxUnpooling_updateGradInput<<<grid, block,
      0, THCState_getCurrentStream(state)>>>(
                                             gradOutputData,
                                             outputTime, outputHeight, outputWidth,
                                             cudaIndices,
                                             cudaGradInput,
                                             dT, dH, dW,
                                             padT, padH, padW, offsetZ);
    THCudaCheck(cudaGetLastError());
    totalZ -= 65535;
    offsetZ += 65535;
  }

  // cleanup
  THCTensor_(free)(state, gradOutput);
  THCTensor_(free)(state, gradInput);
  THCIndexTensor_(free)(state, indices);
  THCTensor_(free)(state, input);
}

#endif
