#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/TemporalReflectionPadding.cu"
#else

void THNN_(TemporalReflectionPadding_updateOutput)(THCState *state,
           THCTensor *input,
           THCTensor *output,
           int padL, int padR) {
  THArgCheck(TensorUtils<THCTensor>::canUse32BitIndexMath(state, input), 2,
             "input tensor must fit into 32-bit index math");

  int planeDim = 0;
  int dimw = 1;
  int numBatch = 1;

  int numInputDims = THCTensor_(nDimension)(state, input);
  THCUNN_argCheck(state, numInputDims == 2 || numInputDims == 3, 2, input,
                  "2D or 3D (batch mode) tensor expected for input, but got: %s")

  if (numInputDims == 3) {
    numBatch = THCTensor_(size)(state, input, 0);
    planeDim++;
    dimw++;
  }

  int numPlanes = THCTensor_(size)(state, input, planeDim);
  int inputW = THCTensor_(size)(state, input, dimw);

  THArgCheck(padL < inputW && padR < inputW, 4,
             "Padding size should be less than the corresponding input dimension, "
             "but got: padding (%d, %d) at dimension %d of input %s",
             padL, padR, dimw, THCTensor_(sizeDesc)(state, input).str);

  int outputW  = inputW + padL + padR;

  THArgCheck(outputW >= 1 , 2,
             "input (W: %d)is too small."
             " Calculated output W: %d",
             inputW, outputW);

  THCDeviceTensor<real, 3> devInput;
  THCDeviceTensor<real, 3> devOutput;

  if (numInputDims == 2) {
    THCTensor_(resize2d)(state, output, numPlanes, outputW);

    devInput = toDeviceTensor<real, 2>(state, input).upcastOuter<3>();
    devOutput = toDeviceTensor<real, 2>(state, output).upcastOuter<3>();
  } else {
    THCTensor_(resize3d)(state, output, numBatch, numPlanes, outputW);

    devInput = toDeviceTensor<real, 3>(state, input);
    devOutput = toDeviceTensor<real, 3>(state, output);
  }

  int outputPlaneSize = devOutput.getSize(2);
  dim3 gridSize(THCCeilDiv(outputPlaneSize, 256),
            devOutput.getSize(1),
            devOutput.getSize(0));
  dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

  TemporalReflectionPadding_updateOutput<<<gridSize, blockSize, 0, THCState_getCurrentStream(state)>>>(
    devInput, devOutput, padL, padR);
  THCudaCheck(cudaGetLastError());
}

void THNN_(TemporalReflectionPadding_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int padL, int padR) {

  THArgCheck(TensorUtils<THCTensor>::canUse32BitIndexMath(state, input), 2,
                "input tensor must fit into 32-bit index math");
  THArgCheck(TensorUtils<THCTensor>::canUse32BitIndexMath(state, gradOutput), 3,
                "output gradient tensor must fit into 32-bit index math");

  int planeDim = 0;
  int dimw = 1;

  int numInputDims = THCTensor_(nDimension)(state, input);
  if (numInputDims == 3) {
    planeDim++;
    dimw++;
  }
  int iwidth = input->size[dimw];
  int owidth  = iwidth + padL + padR;

  THArgCheck(owidth == THCTensor_(size)(state, gradOutput, dimw), 3,
             "gradOutput width unexpected. Expected: %d, Got: %d",
             owidth, THCTensor_(size)(state, gradOutput, dimw));

  THCTensor_(resizeAs)(state, gradInput, input);
  THCTensor_(zero)(state, gradInput);

  THCDeviceTensor<real, 3> devGradInput;
  THCDeviceTensor<real, 3> devGradOutput;

  if (numInputDims == 2) {
    devGradInput = toDeviceTensor<real, 2>(state, gradInput).upcastOuter<3>();
    devGradOutput = toDeviceTensor<real, 2>(state, gradOutput).upcastOuter<3>();
  } else {
    devGradInput = toDeviceTensor<real, 3>(state, gradInput);
    devGradOutput = toDeviceTensor<real, 3>(state, gradOutput);
  }

  int outputPlaneSize = devGradOutput.getSize(2);
  dim3 gridSize(THCCeilDiv(outputPlaneSize, 256),
            devGradOutput.getSize(1),
            devGradOutput.getSize(0));
  dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

  TemporalReflectionPadding_updateGradInput<<<gridSize, blockSize, 0, THCState_getCurrentStream(state)>>>(
    devGradInput, devGradOutput, padL, padR);
  THCudaCheck(cudaGetLastError());
}

#endif
