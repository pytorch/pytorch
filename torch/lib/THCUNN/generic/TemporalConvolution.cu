#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/TemporalConvolution.cu"
#else

static inline void THNN_(TemporalConvolution_shapeCheck)(
                         THCState *state,
                         THCTensor *input,
                         int kW,
                         int dW,
                         int *inputFrameSize) {

  THArgCheck(kW > 0, 9,
             "kernel size should be greater than zero, but got kW: %d", kW);
  THArgCheck(dW > 0, 11,
             "stride should be greater than zero, but got dW: %d", dW);

  int dimS = 0; // sequence dimension
  int dimF = 1; // feature dimension

  if (input->nDimension == 3)
  {
    dimS = 1;
    dimF = 2;
  }
  THCUNN_argCheck(state, input->nDimension == 2 || input->nDimension == 3, 2, input,
                  "2D or 3D (batch mode) tensor expected for input, but got: %s");
  if (inputFrameSize != NULL) {
    THArgCheck(input->size[dimF] == *inputFrameSize, 2,
               "invalid input frame size. Got: %d, Expected: %d",
               input->size[dimF], *inputFrameSize);
  }
  THArgCheck(input->size[dimS] >= kW, 2,
             "input sequence smaller than kernel size. Got: %d, Expected: %d",
             input->size[dimS], kW);
}

void THNN_(TemporalConvolution_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *weight,
           THCTensor *bias,
           int kW, int dW,
           int inputFrameSize,
           int outputFrameSize) {

  THCTensor *outputWindow, *inputWindow;
  int nInputFrame, nOutputFrame;
  int64_t k, i;

  int dimS = 0; // sequence dimension

  THCUNN_assertSameGPU(state, 4, input, output, weight, bias);
  THNN_(TemporalConvolution_shapeCheck)
       (state, input, kW, dW, &inputFrameSize);
  THArgCheck(THCTensor_(isContiguous)(state, weight), 4, "weight must be contiguous");
  THArgCheck(!bias || THCTensor_(isContiguous)(state, bias), 5, "bias must be contiguous");

  if (input->nDimension == 3)
  {
    dimS = 1;
  }

  input = THCTensor_(newContiguous)(state, input);
  outputWindow = THCTensor_(new)(state);
  inputWindow = THCTensor_(new)(state);

  nInputFrame = input->size[dimS];
  nOutputFrame = (nInputFrame - kW) / dW + 1;

  if (input->nDimension == 2)
  {
    THCTensor_(resize2d)(state, output,
                          nOutputFrame,
                          outputFrameSize);

    /* bias first */
    for(k = 0; k < nOutputFrame; k++)
    {
      THCTensor_(select)(state, outputWindow, output, 0, k);
      THCTensor_(copy)(state, outputWindow, bias);
    }


    /* ouch */
    for(k = 0; nOutputFrame > 0; k++)
    {
      int64_t outputFrameStride = (kW-1)/dW+1;
      int64_t inputFrameStride = outputFrameStride*dW;
      int64_t nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
      nOutputFrame -= nFrame;

      THCTensor_(setStorage2d)(state, inputWindow, input->storage,
                              input->storageOffset+k*dW*input->size[1],
                              nFrame, inputFrameStride*input->size[1],
                              kW*input->size[1], 1);

      THCTensor_(setStorage2d)(state, outputWindow, output->storage,
                              output->storageOffset + k*output->size[1],
                              nFrame, outputFrameStride*output->size[1],
                              output->size[1], 1);

      THCTensor *tweight = THCTensor_(new)(state);
      THCTensor_(transpose)(state, tweight, weight, 0, 1);
      THCTensor_(addmm)(state, outputWindow, ScalarConvert<int, real>::to(1), outputWindow, ScalarConvert<int, real>::to(1), inputWindow, tweight);
      THCTensor_(free)(state, tweight);
    }
  }
  else
  {
    THCTensor *outputSample = THCTensor_(new)(state);
    THCTensor *inputSample = THCTensor_(new)(state);
    int nBatchFrame = input->size[0];

    THCTensor_(resize3d)(state, output,
                          nBatchFrame,
                          nOutputFrame,
                          outputFrameSize);

    for(i = 0; i < nBatchFrame; i++)
    {
      THCTensor_(select)(state, outputSample, output, 0, i);
      THCTensor_(select)(state, inputSample, input, 0, i);
      int64_t nOutputSampleFrame = nOutputFrame;

      /* bias first */
      for(k = 0; k < nOutputFrame; k++)
      {
        THCTensor_(select)(state, outputWindow, outputSample, 0, k);
        THCTensor_(copy)(state, outputWindow, bias);
      }

      /* ouch */
      for(k = 0; nOutputSampleFrame > 0; k++)
      {
        int64_t outputFrameStride = (kW-1)/dW+1;
        int64_t inputFrameStride = outputFrameStride*dW;
        int64_t nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
        nOutputSampleFrame -= nFrame;

        THCTensor_(setStorage2d)(state, inputWindow, inputSample->storage,
                                inputSample->storageOffset+k*dW*inputSample->size[1],
                                nFrame, inputFrameStride*inputSample->size[1],
                                kW*inputSample->size[1], 1);

        THCTensor_(setStorage2d)(state, outputWindow, outputSample->storage,
                                outputSample->storageOffset + k*outputSample->size[1],
                                nFrame, outputFrameStride*outputSample->size[1],
                                outputSample->size[1], 1);

        THCTensor *tweight = THCTensor_(new)(state);
        THCTensor_(transpose)(state, tweight, weight, 0, 1);
        THCTensor_(addmm)(state, outputWindow, ScalarConvert<int, real>::to(1), outputWindow, ScalarConvert<int, real>::to(1), inputWindow, tweight);
        THCTensor_(free)(state, tweight);
      }
    }
    THCTensor_(free)(state, outputSample);
    THCTensor_(free)(state, inputSample);
  }

  THCTensor_(free)(state, outputWindow);
  THCTensor_(free)(state, inputWindow);
  THCTensor_(free)(state, input);

}

void THNN_(TemporalConvolution_updateGradInput)(
           THCState* state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *weight,
           int kW, int dW) {

  int64_t nInputFrame;
  int64_t nOutputFrame;

  THCTensor *gradOutputWindow;
  THCTensor *gradInputWindow;
  int64_t k, i;

  int dimS = 0; // sequence dimension

  THCUNN_assertSameGPU(state, 4, input, gradOutput, weight, gradInput);
  THArgCheck(THCTensor_(isContiguous)(state, weight), 4, "weight must be contiguous");
  input = THCTensor_(newContiguous)(state, input);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  THNN_(TemporalConvolution_shapeCheck)
       (state, input, kW, dW, NULL);

  if (gradOutput->nDimension == 3)
  {
    dimS = 1;
  }

  nInputFrame = input->size[dimS];
  nOutputFrame = gradOutput->size[dimS];


  /* Not necessary with partial backprop: */
  gradOutputWindow = THCTensor_(new)(state);
  gradInputWindow = THCTensor_(new)(state);

  THCTensor_(resizeAs)(state, gradInput, input);
  THCTensor_(zero)(state, gradInput);

  if (gradOutput->nDimension == 2)
  {
    /* ouch */
    for(k = 0; nOutputFrame > 0; k++)
    {
      int64_t outputFrameStride = (kW-1)/dW+1;
      int64_t inputFrameStride = outputFrameStride*dW;
      int64_t nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
      nOutputFrame -= nFrame;

      THCTensor_(setStorage2d)(state, gradOutputWindow, gradOutput->storage,
                              gradOutput->storageOffset + k*gradOutput->size[1],
                              nFrame, outputFrameStride*gradOutput->size[1],
                              gradOutput->size[1], 1);

      THCTensor_(setStorage2d)(state, gradInputWindow, gradInput->storage,
                              gradInput->storageOffset+k*dW*gradInput->size[1],
                              nFrame, inputFrameStride*gradInput->size[1],
                              kW*gradInput->size[1], 1);

      THCTensor_(addmm)(state, gradInputWindow, ScalarConvert<int, real>::to(1), gradInputWindow, ScalarConvert<int, real>::to(1), gradOutputWindow, weight);
    }
  }
  else
  {
    THCTensor *gradOutputSample = THCTensor_(new)(state);
    THCTensor *gradInputSample = THCTensor_(new)(state);
    int64_t nBatchFrame = input->size[0];
    for(i = 0; i < nBatchFrame; i++)
    {
      THCTensor_(select)(state, gradOutputSample, gradOutput, 0, i);
      THCTensor_(select)(state, gradInputSample, gradInput, 0, i);
      int64_t nOutputSampleFrame = nOutputFrame;

      /* ouch */
      for(k = 0; nOutputSampleFrame > 0; k++)
      {
        int64_t outputFrameStride = (kW-1)/dW+1;
        int64_t inputFrameStride = outputFrameStride*dW;
        int64_t nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
        nOutputSampleFrame -= nFrame;

        THCTensor_(setStorage2d)(state, gradOutputWindow, gradOutputSample->storage,
                                gradOutputSample->storageOffset + k*gradOutputSample->size[1],
                                nFrame, outputFrameStride*gradOutputSample->size[1],
                                gradOutputSample->size[1], 1);

        THCTensor_(setStorage2d)(state, gradInputWindow, gradInputSample->storage,
                                gradInputSample->storageOffset+k*dW*gradInputSample->size[1],
                                nFrame, inputFrameStride*gradInputSample->size[1],
                                kW*gradInputSample->size[1], 1);

        THCTensor_(addmm)(state, gradInputWindow, ScalarConvert<int, real>::to(1), gradInputWindow, ScalarConvert<int, real>::to(1), gradOutputWindow, weight);
      }
    }
    THCTensor_(free)(state, gradOutputSample);
    THCTensor_(free)(state, gradInputSample);
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
  THCTensor_(free)(state, gradOutputWindow);
  THCTensor_(free)(state, gradInputWindow);

}

void THNN_(TemporalConvolution_accGradParameters)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradWeight,
           THCTensor *gradBias,
           int kW, int dW,
           accreal scale_) {

  real scale = ScalarConvert<accreal, real>::to(scale_);
  int64_t nInputFrame;
  int64_t nOutputFrame;

  THCTensor *gradOutputWindow;
  THCTensor *inputWindow;
  int64_t k, i;

  THNN_(TemporalConvolution_shapeCheck)
       (state, input, kW, dW, NULL);

  int dimS = 0; // sequence dimension

  if (gradOutput->nDimension == 3)
  {
    dimS = 1;
  }

  nInputFrame = input->size[dimS];
  nOutputFrame = gradOutput->size[dimS];

  /* Not necessary with partial backprop: */
  input = THCTensor_(newContiguous)(state, input);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  gradOutputWindow = THCTensor_(new)(state);
  inputWindow = THCTensor_(new)(state);

  if (input->nDimension == 2)
  {
    /* bias first */
    for(k = 0; k < nOutputFrame; k++)
    {
      THCTensor_(select)(state, gradOutputWindow, gradOutput, 0, k);
      THCTensor_(cadd)(state, gradBias, gradBias, scale, gradOutputWindow);
    }

    /* ouch */
    for(k = 0; nOutputFrame > 0; k++)
    {
      int64_t outputFrameStride = (kW-1)/dW+1;
      int64_t inputFrameStride = outputFrameStride*dW;
      int64_t nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
      nOutputFrame -= nFrame;

      THCTensor_(setStorage2d)(state, inputWindow, input->storage,
                              input->storageOffset+k*dW*input->size[1],
                              nFrame, inputFrameStride*input->size[1],
                              kW*input->size[1], 1);

      THCTensor_(setStorage2d)(state, gradOutputWindow, gradOutput->storage,
                              gradOutput->storageOffset + k*gradOutput->size[1],
                              nFrame, outputFrameStride*gradOutput->size[1],
                              gradOutput->size[1], 1);

      THCTensor *tgradOutputWindow = THCTensor_(new)(state);
      THCTensor_(transpose)(state, tgradOutputWindow, gradOutputWindow, 0, 1);
      THCTensor_(addmm)(state, gradWeight, ScalarConvert<int, real>::to(1), gradWeight, scale, tgradOutputWindow, inputWindow);
      THCTensor_(free)(state, tgradOutputWindow);
    }
  }
  else
  {
    THCTensor *gradOutputSample = THCTensor_(new)(state);
    THCTensor *inputSample = THCTensor_(new)(state);
    int64_t nBatchFrame = input->size[0];

    for(i = 0; i < nBatchFrame; i++)
    {
      THCTensor_(select)(state, gradOutputSample, gradOutput, 0, i);
      THCTensor_(select)(state, inputSample, input, 0, i);
      int64_t nOutputSampleFrame = nOutputFrame;

      /* bias first */
      for(k = 0; k < nOutputFrame; k++)
      {
        THCTensor_(select)(state, gradOutputWindow, gradOutputSample, 0, k);
        THCTensor_(cadd)(state, gradBias, gradBias, scale, gradOutputWindow);
      }

      /* ouch */
      for(k = 0; nOutputSampleFrame > 0; k++)
      {
        int64_t outputFrameStride = (kW-1)/dW+1;
        int64_t inputFrameStride = outputFrameStride*dW;
        int64_t nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
        nOutputSampleFrame -= nFrame;

        THCTensor_(setStorage2d)(state, inputWindow, inputSample->storage,
                                inputSample->storageOffset+k*dW*inputSample->size[1],
                                nFrame, inputFrameStride*inputSample->size[1],
                                kW*inputSample->size[1], 1);

        THCTensor_(setStorage2d)(state, gradOutputWindow, gradOutputSample->storage,
                                gradOutputSample->storageOffset + k*gradOutputSample->size[1],
                                nFrame, outputFrameStride*gradOutputSample->size[1],
                                gradOutputSample->size[1], 1);

        THCTensor *tgradOutputWindow = THCTensor_(new)(state);
        THCTensor_(transpose)(state, tgradOutputWindow, gradOutputWindow, 0, 1);
        THCTensor_(addmm)(state, gradWeight, ScalarConvert<int, real>::to(1), gradWeight, scale, tgradOutputWindow, inputWindow);
        THCTensor_(free)(state, tgradOutputWindow);
      }
    }
    THCTensor_(free)(state, gradOutputSample);
    THCTensor_(free)(state, inputSample);
  }

  THCTensor_(free)(state, gradOutputWindow);
  THCTensor_(free)(state, inputWindow);
  THCTensor_(free)(state, gradOutput);
  THCTensor_(free)(state, input);

}

#endif
