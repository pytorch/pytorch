#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/VolumetricAveragePooling.cu"
#else

static inline void THNN_(VolumetricAveragePooling_shapeCheck)(
                         THCState *state,
                         THCTensor *input,
                         THCTensor *gradOutput,
                         int kT,
                         int kW,
                         int kH,
                         int dT,
                         int dW,
                         int dH) {
  int inputSlices;
  int inputTime;
  int inputHeight;
  int inputWidth;

  int ndim = input->nDimension;
  int dimN = 0;
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  if (input->nDimension == 5)
  {
    dimN++;
    dimt++;
    dimh++;
    dimw++;
  }

  if (THCTensor_(nDimension)(state, input) == 4)
  {
    THArgCheck(input->size[dimw] >= kW && input->size[dimh] >= kH
               && input->size[dimt] >= kT, 2,
               "input image (T: %d H: %d W: %d) smaller than "
               "kernel size (kT: %d kH: %d kW: %d)",
               input->size[dimt], input->size[dimh], input->size[dimw],
               kT, kH, kW);

    /* sizes */
    inputSlices = THCTensor_(size)(state, input, 0);
    inputTime   = THCTensor_(size)(state, input, 1);
    inputHeight = THCTensor_(size)(state, input, 2);
    inputWidth  = THCTensor_(size)(state, input, 3);
  }
  else if (THCTensor_(nDimension)(state, input) == 5)
  {
    THArgCheck(input->size[dimw] >= kW && input->size[dimh] >= kH
               && input->size[dimt] >= kT, 2,
               "input image (T: %d H: %d W: %d) smaller than "
               "kernel size (kT: %d kH: %d kW: %d)",
               input->size[dimt], input->size[dimh], input->size[dimw],
               kT, kH, kW);

    /* sizes */
    inputSlices = THCTensor_(size)(state, input, 1);
    inputTime   = THCTensor_(size)(state, input, 2);
    inputHeight = THCTensor_(size)(state, input, 3);
    inputWidth  = THCTensor_(size)(state, input, 4);
  }
  else
  {
    THArgCheck(false, 2, "4D or 5D tensor expected, but got: %d", input->nDimension);
  }

  int outputTime   = (inputTime   - kT) / dT + 1;
  int outputHeight = (inputHeight - kH) / dH + 1;
  int outputWidth  = (inputWidth  - kW) / dW + 1;

  if (gradOutput != NULL) {
     THCUNN_check_dim_size(state, gradOutput, ndim, dimN, inputSlices);
     THCUNN_check_dim_size(state, gradOutput, ndim, dimt, outputTime);
     THCUNN_check_dim_size(state, gradOutput, ndim, dimh, outputHeight);
     THCUNN_check_dim_size(state, gradOutput, ndim, dimw, outputWidth);
  }
}

void THNN_(VolumetricAveragePooling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int kT, int kW, int kH,
           int dT, int dW, int dH)
{
  int batchSize;
  int inputSlices;
  int inputTime;
  int inputHeight;
  int inputWidth;

  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  if (input->nDimension == 5)
  {
    dimt++;
    dimh++;
    dimw++;
  }

  THNN_(VolumetricAveragePooling_shapeCheck)
       (state, input, NULL, kT, kW, kH, dT, dW, dH);

  if (THCTensor_(nDimension)(state, input) == 4)
  {
    /* sizes */
    batchSize   = 1;
    inputSlices = THCTensor_(size)(state, input, 0);
    inputTime   = THCTensor_(size)(state, input, 1);
    inputHeight = THCTensor_(size)(state, input, 2);
    inputWidth  = THCTensor_(size)(state, input, 3);
  }
  else if (THCTensor_(nDimension)(state, input) == 5)
  {
    /* sizes */
    batchSize   = THCTensor_(size)(state, input, 0);
    inputSlices = THCTensor_(size)(state, input, 1);
    inputTime   = THCTensor_(size)(state, input, 2);
    inputHeight = THCTensor_(size)(state, input, 3);
    inputWidth  = THCTensor_(size)(state, input, 4);
  }

  int outputTime   = (inputTime   - kT) / dT + 1;
  int outputHeight = (inputHeight - kH) / dH + 1;
  int outputWidth  = (inputWidth  - kW) / dW + 1;

  if (input->nDimension == 4) /* 4D */
  {
    /* resize output */
    THCTensor_(resize4d)(state, output, inputSlices,
                         outputTime, outputHeight, outputWidth);
  }
  else /* 5D */
  {
    THCTensor_(resize5d)(state, output, batchSize, inputSlices,
                         outputTime, outputHeight, outputWidth);
  }

  input = THCTensor_(newContiguous)(state, input);

  // Collapse batch and feature dimensions
  THCDeviceTensor<real, 4> cudaInput;
  THCDeviceTensor<real, 4> cudaOutput;
  if (THCTensor_(nDimension)(state, input) == 4)
  {
    cudaInput  = toDeviceTensor<real, 4>(state, input);
    cudaOutput = toDeviceTensor<real, 4>(state, output);
  }
  else
  {
    cudaInput  = toDeviceTensor<real, 5>(state, input).downcastOuter<4>();
    cudaOutput = toDeviceTensor<real, 5>(state, output).downcastOuter<4>();
  }

  int totalZ = outputTime * inputSlices * batchSize;
  int offsetZ = 0;
  dim3 block(32, 8);
  while (totalZ > 0) {
    dim3 grid(THCCeilDiv(outputWidth, static_cast<int>(block.x)),
              THCCeilDiv(outputHeight, static_cast<int>(block.y)),
              totalZ > 65535 ? 65535 : totalZ);

    accreal normFactor = ScalarConvert<int, accreal>::to(1) / static_cast<accreal>(kT * kH * kW);
    switch (kW)
      {
        LAUNCH_UPDATE_OUTPUT_KERNEL_WIDTH(1);
        LAUNCH_UPDATE_OUTPUT_KERNEL_WIDTH(2);
        LAUNCH_UPDATE_OUTPUT_KERNEL_WIDTH(3);
        LAUNCH_UPDATE_OUTPUT_KERNEL_WIDTH(4);
        LAUNCH_UPDATE_OUTPUT_KERNEL_WIDTH(5);
        LAUNCH_UPDATE_OUTPUT_KERNEL_WIDTH(6);
        LAUNCH_UPDATE_OUTPUT_KERNEL_WIDTH(7);
      default:
        cuda_VolumetricAveragePooling_updateOutput<real, accreal><<<grid, block>>>(
                                                                    cudaInput,
                                                                    cudaOutput,
                                                                    kT, kH, kW,
                                                                    dT, dH, dW,
                                                                    normFactor,
                                                                    offsetZ
                                                                    );
        break;
      }
    totalZ -= 65535;
    offsetZ += 65535;
    THCudaCheck(cudaGetLastError());
  }
  THCTensor_(free)(state, input);
}

void THNN_(VolumetricAveragePooling_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int kT, int kW, int kH,
           int dT, int dW, int dH)
{

  THNN_(VolumetricAveragePooling_shapeCheck)
       (state, input, gradOutput, kT, kW, kH, dT, dW, dH);
  bool kernelsOverlap = (dT < kT) || (dH < kH) || (dW < kW);

  // Resize and initialize result tensor.
  THCTensor_(resizeAs)(state, gradInput, input);
  THCTensor_(zero)(state, gradInput);

  int batchSize;
  int inputSlices;
  int inputTime;
  int inputHeight;
  int inputWidth;

  int outputTime;
  int outputHeight;
  int outputWidth;

  if (THCTensor_(nDimension)(state, input) == 4) /* 4D */
  {
    batchSize = 1;
    inputSlices  = THCTensor_(size)(state, input, 0);
    inputTime    = THCTensor_(size)(state, input, 1);
    inputHeight  = THCTensor_(size)(state, input, 2);
    inputWidth   = THCTensor_(size)(state, input, 3);

    outputTime   = THCTensor_(size)(state, gradOutput, 1);
    outputHeight = THCTensor_(size)(state, gradOutput, 2);
    outputWidth  = THCTensor_(size)(state, gradOutput, 3);
  }
  else
  {
    batchSize    = THCTensor_(size)(state, input, 0);
    inputSlices  = THCTensor_(size)(state, input, 1);
    inputTime    = THCTensor_(size)(state, input, 2);
    inputHeight  = THCTensor_(size)(state, input, 3);
    inputWidth   = THCTensor_(size)(state, input, 4);

    outputTime   = THCTensor_(size)(state, gradOutput, 2);
    outputHeight = THCTensor_(size)(state, gradOutput, 3);
    outputWidth  = THCTensor_(size)(state, gradOutput, 4);
  }

  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  // Collapse batch and feature dimensions
  THCDeviceTensor<real, 4> cudaGradInput;
  THCDeviceTensor<real, 4> cudaGradOutput;
  if (THCTensor_(nDimension)(state, input) == 4)
  {
    cudaGradInput  = toDeviceTensor<real, 4>(state, gradInput);
    cudaGradOutput = toDeviceTensor<real, 4>(state, gradOutput);
  }
  else
  {
    cudaGradInput =
      toDeviceTensor<real, 5>(state, gradInput).downcastOuter<4>();
    cudaGradOutput =
      toDeviceTensor<real, 5>(state, gradOutput).downcastOuter<4>();
  }

  dim3 block(32, 8);

  // Optimizing for stride 1 is probably only of limited value, but this
  // specialization yields 3x speedup over the atomicAdd implementation.
  if (dT == 1 && dH == 1 && dW == 1)
  {
    int totalZ = inputTime * inputSlices * batchSize;
    int offsetZ = 0;
    while (totalZ > 0) {
      dim3 grid(THCCeilDiv(inputWidth, static_cast<int>(block.x)),
                THCCeilDiv(inputHeight, static_cast<int>(block.y)),
                totalZ > 65535 ? 65535 : totalZ);
      cuda_VolumetricAveragePooling_updateGradInput_Stride1<real, accreal><<<grid, block>>>(
         cudaGradOutput, cudaGradInput, kT, kH, kW, 1.0f/(kT * kH * kW), offsetZ);
      THCudaCheck(cudaGetLastError());
      totalZ -= 65535;
      offsetZ += 65535;
    }
  }
  else
  {
    int totalZ = outputTime * inputSlices * batchSize;
    int offsetZ = 0;
    while (totalZ > 0) {

      dim3 grid(THCCeilDiv(outputWidth, static_cast<int>(block.x)),
                THCCeilDiv(outputHeight, static_cast<int>(block.y)),
                totalZ > 65535 ? 65535 : totalZ);
      if (kernelsOverlap)
        {
          cuda_VolumetricAveragePooling_updateGradInput_atomicAdd<real, accreal><<<grid, block>>>(
            cudaGradOutput, cudaGradInput, kT, kH, kW, dT, dH, dW, offsetZ);
        }
      else
        {
          cuda_VolumetricAveragePooling_updateGradInput<real, accreal><<<grid, block>>>(
             cudaGradOutput, cudaGradInput, kT, kH, kW, dT, dH, dW, offsetZ);
        }
      THCudaCheck(cudaGetLastError());
      totalZ -= 65535;
      offsetZ += 65535;
    }
  }

  THCTensor_(free)(state, gradOutput);
}

#endif
