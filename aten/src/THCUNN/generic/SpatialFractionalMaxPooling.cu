#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialFractionalMaxPooling.cu"
#else

void THNN_(SpatialFractionalMaxPooling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int outputW, int outputH,
           int poolSizeW, int poolSizeH,
           THCIndexTensor *indices,
           THCTensor *randomSamples)
{
  int planeDim = 0;
  int dimh = 1;
  int dimw = 2;
  int64_t numBatch = 1;

  int numInputDims = THCTensor_(nDimension)(state, input);
  THCUNN_argCheck(state, numInputDims == 3 || numInputDims == 4, 2, input,
                  "3D or 4D (batch mode) tensor expected for input, but got: %s");

  if (numInputDims == 4) {
    numBatch = THCTensor_(size)(state, input, 0);
    planeDim++;
    dimh++;
    dimw++;
  }

  /* sizes */
  int64_t numPlanes = THCTensor_(size)(state, input, planeDim);
  int64_t inputH = THCTensor_(size)(state, input, dimh);
  int64_t inputW = THCTensor_(size)(state, input, dimw);

  THArgCheck(outputH + poolSizeH - 1 <= inputH, 6,
             "poolSizeH (%d) too large relative to input height (%d)",
             poolSizeH, inputH);
  THArgCheck(outputW + poolSizeW - 1 <= inputW, 5,
             "poolSizeW (%d) too large relative to input width (%d)",
             poolSizeW, inputW);

  THCDeviceTensor<real, 4> devInput;
  THCDeviceTensor<real, 4> devOutput;
  THCDeviceTensor<THCIndex_t, 4> devIndices;
  THCDeviceTensor<real, 3> devSamples =
    toDeviceTensor<real, 3>(state, randomSamples);

  if (numInputDims == 3) {
    /* resize output */
    THCTensor_(resize3d)(state, output, numPlanes, outputH, outputW);
    /* indices will contain the locations for each output point */
    THCIndexTensor_(resize3d)(state, indices, numPlanes, outputH, outputW);

    devInput = toDeviceTensor<real, 3>(state, input).upcastOuter<4>();
    devOutput = toDeviceTensor<real, 3>(state, output).upcastOuter<4>();
    devIndices = toDeviceTensor<THCIndex_t, 3>(state, indices).upcastOuter<4>();
  } else {
    THCTensor_(resize4d)(state, output, numBatch, numPlanes, outputH, outputW);
    /* indices will contain the locations for each output point */
    THCIndexTensor_(resize4d)(state, indices, numBatch, numPlanes, outputH, outputW);

    devInput = toDeviceTensor<real, 4>(state, input);
    devOutput = toDeviceTensor<real, 4>(state, output);
    devIndices = toDeviceTensor<THCIndex_t, 4>(state, indices);
  }

  // block is limited to 4 warps
  // grid handles overflow per each plane
  int outputPlaneSize = devOutput.getSize(2) * devOutput.getSize(3);
  dim3 grid(THCCeilDiv(outputPlaneSize, 128),
            devInput.getSize(1),
            devInput.getSize(0));
  dim3 block(outputPlaneSize > 128 ? 128 : outputPlaneSize);

#define SFMP_UPDATE_OUTPUT(POOL_W)                                      \
  SpatialFractionalMaxPooling_updateOutput<POOL_W, real, accreal>       \
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(             \
      devInput, devOutput, devIndices, devSamples, poolSizeW, poolSizeH);

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

void THNN_(SpatialFractionalMaxPooling_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int outputW, int outputH,
           int poolSizeW, int poolSizeH,
           THCIndexTensor *indices)
{
  int dimh = 1;
  int dimw = 2;

  int64_t numInputDims = THCTensor_(nDimension)(state, input);
  if (numInputDims == 4) {
    dimh++;
    dimw++;
  }

  /* sizes */
  int64_t inputH = THCTensor_(size)(state, input, dimh);
  int64_t inputW = THCTensor_(size)(state, input, dimw);

  THArgCheck(outputH == THCTensor_(size)(state, gradOutput, dimh), 3,
                "gradOutput height unexpected");
  THArgCheck(outputW == THCTensor_(size)(state, gradOutput, dimw), 3,
                "gradOutput width unexpected");

  /* resize */
  THCTensor_(resizeAs)(state, gradInput, input);
  THCTensor_(zero)(state, gradInput);

  THCDeviceTensor<real, 4> devGradInput;
  THCDeviceTensor<real, 4> devGradOutput;
  THCDeviceTensor<THCIndex_t, 4> devIndices;

  /* backprop */
  if (numInputDims == 3) {
    devGradInput = toDeviceTensor<real, 3>(state, gradInput).upcastOuter<4>();
    devGradOutput = toDeviceTensor<real, 3>(state, gradOutput).upcastOuter<4>();
    devIndices = toDeviceTensor<THCIndex_t, 3>(state, indices).upcastOuter<4>();
  } else {
    devGradInput = toDeviceTensor<real, 4>(state, gradInput);
    devGradOutput = toDeviceTensor<real, 4>(state, gradOutput);
    devIndices = toDeviceTensor<THCIndex_t, 4>(state, indices);
  }

  // block is limited to 4 warps
  // grid handles overflow per each plane
  int outputPlaneSize = devGradOutput.getSize(2) * devGradOutput.getSize(3);
  dim3 grid(THCCeilDiv(outputPlaneSize, 128),
            devGradInput.getSize(1),
            devGradInput.getSize(0));
  dim3 block(outputPlaneSize > 128 ? 128 : outputPlaneSize);

  SpatialFractionalMaxPooling_updateGradInput
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      devGradInput, devGradOutput, devIndices);
  THCudaCheck(cudaGetLastError());
}

#endif
