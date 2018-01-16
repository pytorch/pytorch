#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/VolumetricReplicationPadding.cu"
#else

static inline void THNN_(VolumetricReplicationPadding_shapeCheck)(
                         THCState *state,
                         THCTensor *input,
                         THCTensor *gradOutput,
                         int pleft, int pright,
                         int ptop, int pbottom,
                         int pfront, int pback) {
  THArgCheck(TensorUtils<THCTensor>::canUse32BitIndexMath(state, input), 2,
             "input tensor must fit into 32-bit index math");
  int numInputDims = THCTensor_(nDimension)(state, input);

  THCUNN_argCheck(state, numInputDims == 4 || numInputDims == 5, 2, input,
    "4D or 5D (batch mode) tensor expected for input, but got: %s");

  int planeDim = 0;
  int dimd = 1;
  int dimh = 2;
  int dimw = 3;
  if (numInputDims == 5) {
    planeDim++;
    dimd++;
    dimh++;
    dimw++;
    }

  int numPlanes = THCTensor_(size)(state, input, planeDim);
  int idepth = input->size[dimd];
  int iheight = input->size[dimh];
  int iwidth = input->size[dimw];
  int odepth = idepth + pfront + pback;
  int oheight = iheight + ptop + pbottom;
  int owidth  = iwidth + pleft + pright;
  THArgCheck(owidth >= 1 || oheight >= 1 || odepth >= 1, 2,
             "input (D: %d H: %d, W: %d) is too small."
             " Calculated output D: %d H: %d W: %d",
             idepth, iheight, iwidth, odepth, oheight, owidth);

  if (gradOutput != NULL) {
    THArgCheck(TensorUtils<THCTensor>::canUse32BitIndexMath(state, gradOutput),
               3, "output gradient tensor must fit into 32-bit index math");

    THArgCheck(numPlanes == THCTensor_(size)(state, gradOutput, planeDim), 3,
               "gradOutput width unexpected. Expected: %d, Got: %d",
               numPlanes, THCTensor_(size)(state, gradOutput, planeDim));
    THArgCheck(owidth == THCTensor_(size)(state, gradOutput, dimw), 3,
               "gradOutput width unexpected. Expected: %d, Got: %d",
               owidth, THCTensor_(size)(state, gradOutput, dimw));
    THArgCheck(oheight == THCTensor_(size)(state, gradOutput, dimh), 3,
               "gradOutput height unexpected. Expected: %d, Got: %d",
               oheight, THCTensor_(size)(state, gradOutput, dimh));
    THArgCheck(odepth == THCTensor_(size)(state, gradOutput, dimd), 3,
               "gradOutput depth unexpected. Expected: %d, Got: %d",
               odepth, THCTensor_(size)(state, gradOutput, dimd));
  }
}

void THNN_(VolumetricReplicationPadding_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int pleft, int pright,
           int ptop, int pbottom,
           int pfront, int pback) {
  THNN_(VolumetricReplicationPadding_shapeCheck)(
        state, input, NULL, pleft, pright, ptop,
        pbottom, pfront, pback);

  int planeDim = 0;
  int dimd = 1;
  int dimh = 2;
  int dimw = 3;
  int numBatch = 1;

  int numInputDims = THCTensor_(nDimension)(state, input);

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
  THNN_(VolumetricReplicationPadding_shapeCheck)(
        state, input, gradOutput, pleft, pright, ptop,
        pbottom, pfront, pback);

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
