#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/VolumetricDilatedMaxPooling.cu"
#else

#define UPDATE_OUTPUT_KERNEL_WIDTH(KW) case KW:                         \
  cuda_VolumetricDilatedMaxPooling_updateOutput<KW><<<grid, block,             \
    0, THCState_getCurrentStream(state)>>>(                             \
    inputData, inputTime, inputHeight, inputWidth, \
    cudaIndices, cudaOutput, kT, kH, dT, dH, dW, padT, padH, padW,\
    dilationT, dilationH, dilationW, offsetZ); \
    break

static inline void THNN_(VolumetricDilatedMaxPooling_shapeCheck)(
                         THCState *state,
                         THCTensor *input,
                         THCTensor *gradOutput,
                         THCIndexTensor *indices,
                         int kT, int kW, int kH,
                         int dT, int dW, int dH,
                         int padT, int padW, int padH,
                         int dilationT, int dilationW, int dilationH,
                         bool ceilMode) {
  int ndim = input->nDimension;
  int inputSlices;
  int inputTime;
  int inputHeight;
  int inputWidth;
  int outputTime;
  int outputHeight;
  int outputWidth;
  int dimf = 0;
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  THArgCheck(kT > 0 && kW > 0 && kH > 0, 7,
             "kernel size should be greater than zero, but got kT: %d kH: %d kW: %d",
             kT, kH, kW);
  THArgCheck(dT > 0 && dW > 0 && dH > 0, 10,
             "stride should be greater than zero, but got dT: %d dH: %d dW: %d",
             dT, dH, dW);
  THArgCheck(dilationT > 0 && dilationW > 0 && dilationH > 0, 16,
             "dilation should be greater than 0, but got dilationT: %d dilationH: %d dilationW: %d",
             dilationT, dilationH, dilationW);

  if (input->nDimension == 5)
  {
    dimf++;
    dimt++;
    dimh++;
    dimw++;
  }

  if (THCTensor_(nDimension)(state, input) == 4)
  {
    /* sizes */
    inputSlices = THCTensor_(size)(state, input, 0);
    inputTime   = THCTensor_(size)(state, input, 1);
    inputHeight = THCTensor_(size)(state, input, 2);
    inputWidth  = THCTensor_(size)(state, input, 3);
  }
  else if (THCTensor_(nDimension)(state, input) == 5)
  {
    /* sizes */
    inputSlices = THCTensor_(size)(state, input, 1);
    inputTime   = THCTensor_(size)(state, input, 2);
    inputHeight = THCTensor_(size)(state, input, 3);
    inputWidth  = THCTensor_(size)(state, input, 4);
  }
  else
  {
    THArgError(2, "4D or 5D tensor expected, got %d", THCTensor_(nDimension)(state, input));
  }

  THArgCheck(kT/2 >= padT && kW/2 >= padW && kH/2 >= padH, 13,
             "pad should be smaller than half of kernel size, but got "
             "kT: %d kW: %d, kH: %d, padT: %d, padW: %d, padH: %d",
             kT, kW, kH, padT, padW, padH);

  if (ceilMode)
  {
    outputTime   = (int)(ceil((float)(inputTime - (dilationT * (kT - 1) + 1) + 2*padT) / dT)) + 1;
    outputHeight = (int)(ceil((float)(inputHeight - (dilationH * (kH - 1) + 1) + 2*padH) / dH)) + 1;
    outputWidth  = (int)(ceil((float)(inputWidth  - (dilationW * (kW - 1) + 1) + 2*padW) / dW)) + 1;
  }
  else
  {
    outputTime   = (int)(floor((float)(inputTime - (dilationT * (kT - 1) + 1) + 2*padT) / dT)) + 1;
    outputHeight = (int)(floor((float)(inputHeight - (dilationH * (kH - 1) + 1) + 2*padH) / dH)) + 1;
    outputWidth  = (int)(floor((float)(inputWidth  - (dilationW * (kW - 1) + 1) + 2*padW) / dW)) + 1;
  }

  if (padT || padW || padH)
  {
    if ((outputTime - 1)*dT >= inputTime + padT)
      --outputTime;
    if ((outputHeight - 1)*dH >= inputHeight + padH)
      --outputHeight;
    if ((outputWidth  - 1)*dW >= inputWidth  + padW)
      --outputWidth;
  }

  if (outputTime < 1 || outputHeight < 1 || outputWidth < 1)
    THError("Given input size: (%dx%dx%dx%d). Calculated output size: (%dx%dx%dx%d). Output size is too small",
            inputSlices,inputTime,inputHeight,inputWidth,inputSlices,outputTime,outputHeight,outputWidth);

   if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, ndim, dimf, inputSlices);
    THCUNN_check_dim_size(state, gradOutput, ndim, dimt, outputTime);
    THCUNN_check_dim_size(state, gradOutput, ndim, dimh, outputHeight);
    THCUNN_check_dim_size(state, gradOutput, ndim, dimw, outputWidth);
  }
  if (indices != NULL) {
    THCUNN_check_dim_size_indices(state, indices, ndim, dimf, inputSlices);
    THCUNN_check_dim_size_indices(state, indices, ndim, dimt, outputTime);
    THCUNN_check_dim_size_indices(state, indices, ndim, dimh, outputHeight);
    THCUNN_check_dim_size_indices(state, indices, ndim, dimw, outputWidth);
  }
}

void THNN_(VolumetricDilatedMaxPooling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCIndexTensor *indices,
           int kT, int kW, int kH,
           int dT, int dW, int dH,
           int padT, int padW, int padH,
           int dilationT, int dilationW, int dilationH,
           bool ceilMode)
{
  int batchSize;
  int inputSlices;
  int inputTime;
  int inputHeight;
  int inputWidth;
  int outputTime;
  int outputHeight;
  int outputWidth;

  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  int fiveDimensionalInput = THCTensor_(nDimension)(state, input) == 5;

  if (fiveDimensionalInput)
  {
    dimt++;
    dimh++;
    dimw++;
  }

  THCUNN_assertSameGPU(state, 3, input, indices, output);
  THNN_(VolumetricDilatedMaxPooling_shapeCheck)(
        state, input, NULL, NULL, kT, kW, kH,
        dT, dW, dH, padT, padW, padH,
        dilationT, dilationW, dilationH, ceilMode);

  if (THCTensor_(nDimension)(state, input) == 4)
  {
    /* sizes */
    batchSize   = 1;
    inputSlices = THCTensor_(size)(state, input, 0);
    inputTime   = THCTensor_(size)(state, input, 1);
    inputHeight = THCTensor_(size)(state, input, 2);
    inputWidth  = THCTensor_(size)(state, input, 3);
  }
  else if (fiveDimensionalInput)
  {
    /* sizes */
    batchSize   = THCTensor_(size)(state, input, 0);
    inputSlices = THCTensor_(size)(state, input, 1);
    inputTime   = THCTensor_(size)(state, input, 2);
    inputHeight = THCTensor_(size)(state, input, 3);
    inputWidth  = THCTensor_(size)(state, input, 4);
  }
  else
  {
    THArgError(2, "4D or 5D tensor expected, got %d", THCTensor_(nDimension)(state, input));
  }

  if (ceilMode)
  {
    outputTime   = (int)(ceil((float)(inputTime - (dilationT * (kT - 1) + 1) + 2*padT) / dT)) + 1;
    outputHeight = (int)(ceil((float)(inputHeight - (dilationH * (kH - 1) + 1) + 2*padH) / dH)) + 1;
    outputWidth  = (int)(ceil((float)(inputWidth  - (dilationW * (kW - 1) + 1) + 2*padW) / dW)) + 1;
  }
  else
  {
    outputTime   = (int)(floor((float)(inputTime - (dilationT * (kT - 1) + 1) + 2*padT) / dT)) + 1;
    outputHeight = (int)(floor((float)(inputHeight - (dilationH * (kH - 1) + 1) + 2*padH) / dH)) + 1;
    outputWidth  = (int)(floor((float)(inputWidth  - (dilationW * (kW - 1) + 1) + 2*padW) / dW)) + 1;
  }

  if (padT || padW || padH)
  {
    if ((outputTime - 1)*dT >= inputTime + padT)
      --outputTime;
    if ((outputHeight - 1)*dH >= inputHeight + padH)
      --outputHeight;
    if ((outputWidth  - 1)*dW >= inputWidth  + padW)
      --outputWidth;
  }

  if (!fiveDimensionalInput) /* 4D */
  {
    /* resize output */
    THCTensor_(resize4d)(state, output, inputSlices,
                          outputTime, outputHeight, outputWidth);
    /* indices pack ti,i,j locations for each output point as uchar into
     each float of the tensor */
    THCIndexTensor_(resize4d)(state, indices, inputSlices,
                          outputTime, outputHeight, outputWidth);
  }
  else
  { /* 5D */
    THCTensor_(resize5d)(state, output, batchSize, inputSlices,
                          outputTime, outputHeight, outputWidth);
    // Index tensor packs index offsets as uchars into floats
    THCIndexTensor_(resize5d)(state, indices, batchSize, inputSlices,
                          outputTime, outputHeight, outputWidth);
    fiveDimensionalInput = 1;
  }

  input = THCTensor_(newContiguous)(state, input);
  if (fiveDimensionalInput) {
    // Collapse batch and feature dimensions
    output = THCTensor_(newFoldBatchDim)(state, output);

    THCTensor *old_input = input;
    input = THCTensor_(newFoldBatchDim)(state, input);
    THCTensor_(free)(state, old_input);
  } else {
    THCTensor_(retain)(state, output);
  }
  
  real* inputData = THCTensor_(data)(state, input);

  THCDeviceTensor<real, 4> cudaOutput;
  cudaOutput = toDeviceTensor<real, 4>(state, output);

  THLongStorage *indicesSize = THLongStorage_newWithSize(4);
  int64_t indicesSizeRaw[4] = { batchSize * inputSlices,
                            outputTime, outputHeight, outputWidth };
  THLongStorage_rawCopy(indicesSize, indicesSizeRaw);

  THCIndexTensor *indices1 = THCIndexTensor_(newWithStorage)(
    state, THCIndexTensor_(storage)(state, indices),
    THCIndexTensor_(storageOffset)(state, indices),
    indicesSize, NULL);

  THLongStorage_free(indicesSize);

  THCDeviceTensor<THCIndex_t, 4> cudaIndices =
    toDeviceTensor<THCIndex_t, 4>(state, indices1);

  int totalZ = outputTime * inputSlices * batchSize;
  int offsetZ = 0;
  dim3 block(32, 8);

  while (totalZ > 0) {
    dim3 grid(THCCeilDiv(outputWidth, static_cast<int>(block.x)),
              THCCeilDiv(outputHeight, static_cast<int>(block.y)),
              totalZ > 65535 ? 65535 : totalZ);

    switch (kW)
      {
        UPDATE_OUTPUT_KERNEL_WIDTH(1);
        UPDATE_OUTPUT_KERNEL_WIDTH(2);
        UPDATE_OUTPUT_KERNEL_WIDTH(3);
        UPDATE_OUTPUT_KERNEL_WIDTH(4);
        UPDATE_OUTPUT_KERNEL_WIDTH(5);
        UPDATE_OUTPUT_KERNEL_WIDTH(6);
        UPDATE_OUTPUT_KERNEL_WIDTH(7);
      default:
        cuda_VolumetricDilatedMaxPooling_updateOutput<<<grid, block,
          0, THCState_getCurrentStream(state)>>>(
                             inputData, inputTime, inputHeight, inputWidth,
                             cudaIndices, cudaOutput,
                             kT, kH, kW, dT, dH, dW,
                             padT, padH, padW, dilationT, dilationH, dilationW, offsetZ);
      }
    THCudaCheck(cudaGetLastError());
    totalZ -= 65535;
    offsetZ += 65535;
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, output);
  THCIndexTensor_(free)(state, indices1);
}

#undef UPDATE_OUTPUT_KERNEL_WIDTH

void THNN_(VolumetricDilatedMaxPooling_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCIndexTensor *indices,
           int kT, int kW, int kH,
           int dT, int dW, int dH,
           int padT, int padW, int padH,
           int dilationT, int dilationW, int dilationH,
           bool ceilMode)
{
  // TODO: gradOutput shape check
  // Resize and initialize result tensor.
  THCTensor_(resizeAs)(state, gradInput, input);
  THCTensor_(zero)(state, gradInput);

  int batchSize;
  int inputSlices;

  int outputTime, outputHeight, outputWidth;
  int inputTime, inputHeight, inputWidth;

  int fiveDimensionalInput = THCTensor_(nDimension)(state, input) == 5;

  THCUNN_assertSameGPU(state, 4, input, indices, gradOutput, gradInput);
  THNN_(VolumetricDilatedMaxPooling_shapeCheck)(
        state, input, gradOutput, indices, kT, kW, kH,
        dT, dW, dH, padT, padW, padH,
        dilationT, dilationW, dilationH, ceilMode);

  if (!fiveDimensionalInput) /* 4D */
  {
    batchSize = 1;
    inputSlices  = THCTensor_(size)(state, input, 0);

    outputTime   = THCTensor_(size)(state, gradOutput, 1);
    outputHeight = THCTensor_(size)(state, gradOutput, 2);
    outputWidth  = THCTensor_(size)(state, gradOutput, 3);
    inputTime   = THCTensor_(size)(state, gradInput, 1);
    inputHeight = THCTensor_(size)(state, gradInput, 2);
    inputWidth  = THCTensor_(size)(state, gradInput, 3);
  }
  else
  {
    batchSize    = THCTensor_(size)(state, input, 0);
    inputSlices  = THCTensor_(size)(state, input, 1);

    outputTime   = THCTensor_(size)(state, gradOutput, 2);
    outputHeight = THCTensor_(size)(state, gradOutput, 3);
    outputWidth  = THCTensor_(size)(state, gradOutput, 4);
    inputTime   = THCTensor_(size)(state, gradInput, 2);
    inputHeight = THCTensor_(size)(state, gradInput, 3);
    inputWidth  = THCTensor_(size)(state, gradInput, 4);
  }

  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  if (fiveDimensionalInput) {
    // Collapse batch and feature dimensions
    gradInput = THCTensor_(newFoldBatchDim)(state, gradInput);

    THCTensor *old_gradOutput = gradOutput;
    gradOutput = THCTensor_(newFoldBatchDim)(state, gradOutput);
    THCTensor_(free)(state, old_gradOutput);
  } else {
    THCTensor_(retain)(state, gradInput);
  }

  THCDeviceTensor<real, 4> cudaGradOutput;
  cudaGradOutput = toDeviceTensor<real, 4>(state, gradOutput);
  real* gradInputData = THCTensor_(data)(state, gradInput);

  THLongStorage *indicesSize = THLongStorage_newWithSize(4);
  int64_t indicesSizeRaw[4] = { batchSize * inputSlices,
                           outputTime, outputHeight, outputWidth };
  THLongStorage_rawCopy(indicesSize, indicesSizeRaw);
  THCIndexTensor *indices1 = THCIndexTensor_(newWithStorage)(
    state, THCIndexTensor_(storage)(state, indices),
    THCIndexTensor_(storageOffset)(state, indices), indicesSize, NULL);
  THLongStorage_free(indicesSize);

  THCDeviceTensor<THCIndex_t, 4> cudaIndices =
    toDeviceTensor<THCIndex_t, 4>(state, indices1);

  int totalZ = outputTime * inputSlices * batchSize;
  int offsetZ = 0;
  dim3 block(32, 8);

  while (totalZ > 0) {
    dim3 grid(THCCeilDiv(outputWidth, static_cast<int>(block.x)),
              THCCeilDiv(outputHeight, static_cast<int>(block.y)),
              totalZ > 65535 ? 65535 : totalZ);

    cuda_VolumetricDilatedMaxPooling_updateGradInput<<<grid, block,
      0, THCState_getCurrentStream(state)>>>(
                                             cudaGradOutput,
                                             cudaIndices,
                                             gradInputData,
                                             inputTime, inputHeight, inputWidth,
                                             dT, dH, dW,
                                             padT, padH, padW,
                                             dilationT, dilationH, dilationW, offsetZ);
    THCudaCheck(cudaGetLastError());
    totalZ -= 65535;
    offsetZ += 65535;
  }

  // cleanup
  THCTensor_(free)(state, gradInput);
  THCTensor_(free)(state, gradOutput);
  THCIndexTensor_(free)(state, indices1);
}

#endif
