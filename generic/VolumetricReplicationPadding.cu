#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/VolumetricReplicationPadding.cu"
#else

void THNN_(VolumetricReplicationPadding_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int pleft, int pright,
           int ptop, int pbottom,
           int pfront, int pback) {
  THArgCheck(TensorUtils<THCTensor>::canUse32BitIndexMath(state, input), 2,
             "input tensor must fit into 32-bit index math");

  int planeDim = 0;
  int dimd = 1;
  int dimh = 2;
  int dimw = 3;
  int numBatch = 1;

  int numInputDims = THCTensor_(nDimension)(state, input);
  THArgCheck(numInputDims == 4 || numInputDims == 5, 2,
             "input must be 4 or 5-dimensional");

  if (numInputDims == 5) {
    numBatch = THCTensor_(size)(state, input, 0);
    planeDim++;
    dimd++;
    dimh++;
    dimw++;
  }

  int numPlanes = THCTensor_(size)(state, input, planeDim);
  int inputD = THCTensor_(size)(state, input, dimd);
  int inputH = THCTensor_(size)(state, input, dimh);
  int inputW = THCTensor_(size)(state, input, dimw);
  int outputD = inputD + pfront + pback;
  int outputH = inputH + ptop + pbottom;
  int outputW  = inputW + pleft + pright;

  THCDeviceTensor<real, 5> devInput;
  THCDeviceTensor<real, 5> devOutput;

  if (numInputDims == 4) {
    THCTensor_(resize4d)(state, output, numPlanes, outputD, outputH, outputW);

    devInput = toDeviceTensor<real, 4>(state, input).upcastOuter<5>();
    devOutput = toDeviceTensor<real, 4>(state, output).upcastOuter<5>();
  } else {
    THCTensor_(resize5d)(state, output, numBatch, numPlanes, outputD, outputH,
                          outputW);

    devInput = toDeviceTensor<real, 5>(state, input);
    devOutput = toDeviceTensor<real, 5>(state, output);
  }

  int outputPlaneSize = devOutput.getSize(2) * devOutput.getSize(3) *
      devOutput.getSize(4);
  dim3 gridSize(THCCeilDiv(outputPlaneSize, 256),
            devOutput.getSize(1),
            devOutput.getSize(0));
  dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

  VolumetricReplicationPadding_updateOutput<real><<<gridSize, blockSize, 0, THCState_getCurrentStream(state)>>>(
    devInput, devOutput, pfront, pback, ptop, pbottom, pleft, pright);
}

void THNN_(VolumetricReplicationPadding_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int pleft, int pright,
           int ptop, int pbottom,
           int pfront, int pback) {
  THArgCheck(TensorUtils<THCTensor>::canUse32BitIndexMath(state, input), 2,
             "input tensor must fit into 32-bit index math");
  THArgCheck(TensorUtils<THCTensor>::canUse32BitIndexMath(state, gradOutput),
             3, "output gradient tensor must fit into 32-bit index math");

  int planeDim = 0;
  int dimd = 1;
  int dimh = 2;
  int dimw = 3;

  int numInputDims = THCTensor_(nDimension)(state, input);
  if (numInputDims == 5) {
    planeDim++;
    dimd++;
    dimh++;
    dimw++;
  }

  THCTensor_(resizeAs)(state, gradInput, input);
  THCTensor_(zero)(state, gradInput);

  THCDeviceTensor<real, 5> devGradInput;
  THCDeviceTensor<real, 5> devGradOutput;

  if (numInputDims == 4) {
    devGradInput = toDeviceTensor<real, 4>(state, gradInput).upcastOuter<5>();
    devGradOutput =
        toDeviceTensor<real, 4>(state, gradOutput).upcastOuter<5>();
  } else {
    devGradInput = toDeviceTensor<real, 5>(state, gradInput);
    devGradOutput = toDeviceTensor<real, 5>(state, gradOutput);
  }

  int outputPlaneSize = devGradOutput.getSize(2) * devGradOutput.getSize(3) *
      devGradOutput.getSize(4);
  dim3 gridSize(THCCeilDiv(outputPlaneSize, 256),
            devGradOutput.getSize(1),
            devGradOutput.getSize(0));
  dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

  VolumetricReplicationPadding_updateGradInput<<<gridSize, blockSize, 0, THCState_getCurrentStream(state)>>>(
    devGradInput, devGradOutput, pfront, pback, ptop, pbottom, pleft, pright);
}

#endif
