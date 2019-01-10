#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/VolumetricFractionalMaxPooling.cu"
#else

void THNN_(VolumetricFractionalMaxPooling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int outputT, int outputW, int outputH,
           int poolSizeT, int poolSizeW, int poolSizeH,
           THCIndexTensor *indices,
           THCTensor *randomSamples)
{
  int planeDim = 0;
  int dimh = 1;
  int dimw = 2;
  int dimt = 3;
  int64_t numBatch = 1;

  int64_t numInputDims = THCTensor_(nDimension)(state, input);
  THCUNN_argCheck(state, numInputDims == 4 || numInputDims == 5, 2, input,
                  "4D or 5D (batch mode) tensor expected for input, but got: %s");

  if (numInputDims == 5) {
    numBatch = THCTensor_(size)(state, input, 0);
    planeDim++;
    dimh++;
    dimw++;
    dimt++;
  }

  /* sizes */
  int64_t numPlanes = THCTensor_(size)(state, input, planeDim);
  int64_t inputH = THCTensor_(size)(state, input, dimh);
  int64_t inputW = THCTensor_(size)(state, input, dimw);
  int64_t inputT = THCTensor_(size)(state, input, dimt);

  THArgCheck(outputH + poolSizeH - 1 < inputH, 7,
             "poolSizeH (%d) too large relative to input height (%d)",
             poolSizeH, inputH);
  THArgCheck(outputW + poolSizeW - 1 < inputW, 6,
             "poolSizeW (%d) too large relative to input width (%d)",
             poolSizeW, inputW);
  THArgCheck(outputT + poolSizeT - 1 < inputW, 5,
             "poolSizeT (%d) too large relative to input time (%d)",
             poolSizeT, inputT);

  THCDeviceTensor<real, 5> devInput;
  THCDeviceTensor<real, 5> devOutput;
  THCDeviceTensor<THCIndex_t, 5> devIndices;
  THCDeviceTensor<real, 3> devSamples =
    toDeviceTensor<real, 3>(state, randomSamples);

  if (numInputDims == 4) {
    /* resize output */
    THCTensor_(resize4d)(state, output, numPlanes, outputH, outputW, outputT);
    /* indices will contain the locations for each output point */
    THCIndexTensor_(resize4d)(state, indices, numPlanes, outputH, outputW, outputT);

    devInput = toDeviceTensor<real, 4>(state, input).upcastOuter<5>();
    devOutput = toDeviceTensor<real, 4>(state, output).upcastOuter<5>();
    devIndices = toDeviceTensor<THCIndex_t, 4>(state, indices).upcastOuter<5>();
  } else {
    THCTensor_(resize5d)(state, output, numBatch, numPlanes, outputH, outputW, outputT);
    /* indices will contain the locations for each output point */
    THCIndexTensor_(resize5d)(state, indices, numBatch, numPlanes, outputH, outputW, outputT);

    devInput = toDeviceTensor<real, 5>(state, input);
    devOutput = toDeviceTensor<real, 5>(state, output);
    devIndices = toDeviceTensor<THCIndex_t, 5>(state, indices);
  }

  // block is limited to 4 warps
  // grid handles overflow per each plane
  int outputPlaneSize = devOutput.getSize(2) * devOutput.getSize(3) * devOutput.getSize(4);
  dim3 grid(THCCeilDiv(outputPlaneSize, 128),
            devInput.getSize(1),
            devInput.getSize(0));
  dim3 block(outputPlaneSize > 128 ? 128 : outputPlaneSize);

#define SFMP_UPDATE_OUTPUT(POOL_W)                                      \
  VolumetricFractionalMaxPooling_updateOutput<POOL_W, real, accreal>       \
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(             \
      devInput, devOutput, devIndices, devSamples, poolSizeT, poolSizeW, poolSizeH);

#define SFMP_UPDATE_OUTPUT_CASE(POOL_W)                 \
  case POOL_W: SFMP_UPDATE_OUTPUT(POOL_W); break

  switch (poolSizeW) {
    SFMP_UPDATE_OUTPUT_CASE(2);
    SFMP_UPDATE_OUTPUT_CASE(3);
    SFMP_UPDATE_OUTPUT_CASE(4);
    SFMP_UPDATE_OUTPUT_CASE(5);
    SFMP_UPDATE_OUTPUT_CASE(6);
    SFMP_UPDATE_OUTPUT_CASE(7);
    default:
      // dynamic pool width
      SFMP_UPDATE_OUTPUT_CASE(-1);
  }
  THCudaCheck(cudaGetLastError());
}

void THNN_(VolumetricFractionalMaxPooling_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int outputT, int outputW, int outputH,
           int poolSizeT, int poolSizeW, int poolSizeH,
           THCIndexTensor *indices)
{
  int dimh = 1;
  int dimw = 2;
  int dimt = 3;

  int64_t numInputDims = THCTensor_(nDimension)(state, input);
  if (numInputDims == 5) {
    dimh++;
    dimw++;
    dimt++;
  }

  /* sizes */
  int64_t inputH = THCTensor_(size)(state, input, dimh);
  int64_t inputW = THCTensor_(size)(state, input, dimw);
  int64_t inputT = THCTensor_(size)(state, input, dimt);

  THArgCheck(outputH == THCTensor_(size)(state, gradOutput, dimh), 3,
                "gradOutput height unexpected");
  THArgCheck(outputW == THCTensor_(size)(state, gradOutput, dimw), 3,
                "gradOutput width unexpected");
  THArgCheck(outputT == THCTensor_(size)(state, gradOutput, dimt), 3,
                "gradOutput time unexpected");

  /* resize */
  THCTensor_(resizeAs)(state, gradInput, input);
  THCTensor_(zero)(state, gradInput);

  THCDeviceTensor<real, 5> devGradInput;
  THCDeviceTensor<real, 5> devGradOutput;
  THCDeviceTensor<THCIndex_t, 5> devIndices;

  /* backprop */
  if (numInputDims == 4) {
    devGradInput = toDeviceTensor<real, 4>(state, gradInput).upcastOuter<5>();
    devGradOutput = toDeviceTensor<real, 4>(state, gradOutput).upcastOuter<5>();
    devIndices = toDeviceTensor<THCIndex_t, 4>(state, indices).upcastOuter<5>();
  } else {
    devGradInput = toDeviceTensor<real, 5>(state, gradInput);
    devGradOutput = toDeviceTensor<real, 5>(state, gradOutput);
    devIndices = toDeviceTensor<THCIndex_t, 5>(state, indices);
  }

  // block is limited to 4 warps
  // grid handles overflow per each plane
  int outputPlaneSize = devGradOutput.getSize(2) * devGradOutput.getSize(3) * devGradOutput.getSize(4);
  dim3 grid(THCCeilDiv(outputPlaneSize, 128),
            devGradInput.getSize(1),
            devGradInput.getSize(0));
  dim3 block(outputPlaneSize > 128 ? 128 : outputPlaneSize);

  VolumetricFractionalMaxPooling_updateGradInput
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      devGradInput, devGradOutput, devIndices);
  THCudaCheck(cudaGetLastError());
}

#endif
