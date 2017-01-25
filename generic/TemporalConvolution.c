#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TemporalConvolution.c"
#else

static inline void THNN_(TemporalConvolution_shapeCheck)(
                         THNNState *state,
                         THTensor *input,
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
  THNN_ARGCHECK(input->nDimension == 2 || input->nDimension == 3, 2, input,
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
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          int kW,
          int dW,
          int inputFrameSize,
          int outputFrameSize)
{
  THTensor *outputWindow, *inputWindow;
  int nInputFrame, nOutputFrame;
  long k, i;
  
  int dimS = 0; // sequence dimension
  int dimF = 1; // feature dimension
  
  if (input->nDimension == 3) 
  {
    dimS = 1;
    dimF = 2;
  }

  THNN_(TemporalConvolution_shapeCheck)
       (state, input, kW, dW, &inputFrameSize);
  input = THTensor_(newContiguous)(input);
  outputWindow = THTensor_(new)();
  inputWindow = THTensor_(new)();

  nInputFrame = input->size[dimS];
  nOutputFrame = (nInputFrame - kW) / dW + 1;

  if (input->nDimension == 2)
  {
    THTensor_(resize2d)(output,
                        nOutputFrame,
                        outputFrameSize);

    /* bias first */
    for(k = 0; k < nOutputFrame; k++)
    {
      THTensor_(select)(outputWindow, output, 0, k);
      THTensor_(copy)(outputWindow, bias);
    }

    /* ouch */
    for(k = 0; nOutputFrame > 0; k++)
    {
      long outputFrameStride = (kW-1)/dW+1;
      long inputFrameStride = outputFrameStride*dW;
      long nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
      nOutputFrame -= nFrame;

      THTensor_(setStorage2d)(inputWindow, input->storage,
                              input->storageOffset+k*dW*input->size[1],
                              nFrame, inputFrameStride*input->size[1],
                              kW*input->size[1], 1);

      THTensor_(setStorage2d)(outputWindow, output->storage, 
                              output->storageOffset + k*output->size[1],
                              nFrame, outputFrameStride*output->size[1],
                              output->size[1], 1);

      THTensor_(transpose)(weight, NULL, 0, 1);
      THTensor_(addmm)(outputWindow, 1, outputWindow, 1, inputWindow, weight);
      THTensor_(transpose)(weight, NULL, 0, 1);
    }
  }
  else
  {
    THTensor *outputSample = THTensor_(new)();
    THTensor *inputSample = THTensor_(new)();
    int nBatchFrame = input->size[0];
    
    THTensor_(resize3d)(output,
                        nBatchFrame,
                        nOutputFrame,
                        outputFrameSize);
    
    for(i = 0; i < nBatchFrame; i++)
    {
      THTensor_(select)(outputSample, output, 0, i);
      THTensor_(select)(inputSample, input, 0, i);
      long nOutputSampleFrame = nOutputFrame;
      
      /* bias first */
      for(k = 0; k < nOutputFrame; k++)
      {
        THTensor_(select)(outputWindow, outputSample, 0, k);
        THTensor_(copy)(outputWindow, bias);
      }

      /* ouch */
      for(k = 0; nOutputSampleFrame > 0; k++)
      {
        long outputFrameStride = (kW-1)/dW+1;
        long inputFrameStride = outputFrameStride*dW;
        long nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
        nOutputSampleFrame -= nFrame;

        THTensor_(setStorage2d)(inputWindow, inputSample->storage,
                                inputSample->storageOffset+k*dW*inputSample->size[1],
                                nFrame, inputFrameStride*inputSample->size[1],
                                kW*inputSample->size[1], 1);

        THTensor_(setStorage2d)(outputWindow, outputSample->storage, 
                                outputSample->storageOffset + k*outputSample->size[1],
                                nFrame, outputFrameStride*outputSample->size[1],
                                outputSample->size[1], 1);

        THTensor_(transpose)(weight, NULL, 0, 1);
        THTensor_(addmm)(outputWindow, 1, outputWindow, 1, inputWindow, weight);
        THTensor_(transpose)(weight, NULL, 0, 1);
      }
    }
    THTensor_(free)(outputSample);
    THTensor_(free)(inputSample);
  }

  THTensor_(free)(outputWindow);
  THTensor_(free)(inputWindow);
  THTensor_(free)(input);

}

void THNN_(TemporalConvolution_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          int kW,
          int dW)
{
  long nInputFrame;
  long nOutputFrame;

  THTensor *gradOutputWindow;
  THTensor *gradInputWindow;
  long k, i;
  
  int dimS = 0; // sequence dimension
  int dimF = 1; // feature dimension
  
  if (gradOutput->nDimension == 3) 
  {
    dimS = 1;
    dimF = 2;
  }

  THNN_(TemporalConvolution_shapeCheck)(
        state, input, kW, dW, NULL);
  nInputFrame = input->size[dimS];
  nOutputFrame = gradOutput->size[dimS];

  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  gradOutputWindow = THTensor_(new)();
  gradInputWindow = THTensor_(new)();

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  if (gradOutput->nDimension == 2)
  {
    /* ouch */
    for(k = 0; nOutputFrame > 0; k++)
    {
      long outputFrameStride = (kW-1)/dW+1;
      long inputFrameStride = outputFrameStride*dW;
      long nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
      nOutputFrame -= nFrame;

      THTensor_(setStorage2d)(gradOutputWindow, gradOutput->storage,
                              gradOutput->storageOffset + k*gradOutput->size[1],
                              nFrame, outputFrameStride*gradOutput->size[1],
                              gradOutput->size[1], 1);

      THTensor_(setStorage2d)(gradInputWindow, gradInput->storage,
                              gradInput->storageOffset+k*dW*gradInput->size[1],
                              nFrame, inputFrameStride*gradInput->size[1],
                              kW*gradInput->size[1], 1);

      THTensor_(addmm)(gradInputWindow, 1, gradInputWindow, 1, gradOutputWindow, weight);
    }
  }
  else
  {
    THTensor *gradOutputSample = THTensor_(new)();
    THTensor *gradInputSample = THTensor_(new)();
    int nBatchFrame = input->size[0];
    
    for(i = 0; i < nBatchFrame; i++)
    {
      THTensor_(select)(gradOutputSample, gradOutput, 0, i);
      THTensor_(select)(gradInputSample, gradInput, 0, i);
      int nOutputSampleFrame = nOutputFrame;
      
      /* ouch */
      for(k = 0; nOutputSampleFrame > 0; k++)
      {
        long outputFrameStride = (kW-1)/dW+1;
        long inputFrameStride = outputFrameStride*dW;
        long nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
        nOutputSampleFrame -= nFrame;

        THTensor_(setStorage2d)(gradOutputWindow, gradOutputSample->storage,
                                gradOutputSample->storageOffset + k*gradOutputSample->size[1],
                                nFrame, outputFrameStride*gradOutputSample->size[1],
                                gradOutputSample->size[1], 1);

        THTensor_(setStorage2d)(gradInputWindow, gradInputSample->storage,
                                gradInputSample->storageOffset+k*dW*gradInputSample->size[1],
                                nFrame, inputFrameStride*gradInputSample->size[1],
                                kW*gradInputSample->size[1], 1);

        THTensor_(addmm)(gradInputWindow, 1, gradInputWindow, 1, gradOutputWindow, weight);
      }
    }
    THTensor_(free)(gradOutputSample);
    THTensor_(free)(gradInputSample);
  }

  THTensor_(free)(gradOutputWindow);
  THTensor_(free)(gradInputWindow);
  THTensor_(free)(gradOutput);
  THTensor_(free)(input);

}

void THNN_(TemporalConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          int kW,
          int dW,
          real scale)
{
  long nInputFrame;
  long nOutputFrame;

  THTensor *gradOutputWindow;
  THTensor *inputWindow;
  long k, i;
  
  int dimS = 0; // sequence dimension
  int dimF = 1; // feature dimension
  
  if (gradOutput->nDimension == 3) 
  {
    dimS = 1;
    dimF = 2;
  }

  THNN_(TemporalConvolution_shapeCheck)(
        state, input, kW, dW, NULL);
  nInputFrame = input->size[dimS];
  nOutputFrame = gradOutput->size[dimS];

  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  gradOutputWindow = THTensor_(new)();
  inputWindow = THTensor_(new)();
  
  if (input->nDimension == 2)
  {
    /* bias first */
    for(k = 0; k < nOutputFrame; k++)
    {
      THTensor_(select)(gradOutputWindow, gradOutput, 0, k);
      THTensor_(cadd)(gradBias, gradBias, scale, gradOutputWindow);
    }

    /* ouch */
    for(k = 0; nOutputFrame > 0; k++)
    {
      long outputFrameStride = (kW-1)/dW+1;
      long inputFrameStride = outputFrameStride*dW;
      long nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
      nOutputFrame -= nFrame;

      THTensor_(setStorage2d)(inputWindow, input->storage,
                              input->storageOffset+k*dW*input->size[1],
                              nFrame, inputFrameStride*input->size[1],
                              kW*input->size[1], 1);

      THTensor_(setStorage2d)(gradOutputWindow, gradOutput->storage, 
                              gradOutput->storageOffset + k*gradOutput->size[1],
                              nFrame, outputFrameStride*gradOutput->size[1],
                              gradOutput->size[1], 1);

      THTensor_(transpose)(gradOutputWindow, NULL, 0, 1);
      THTensor_(addmm)(gradWeight, 1, gradWeight, scale, gradOutputWindow, inputWindow);
      THTensor_(transpose)(gradOutputWindow, NULL, 0, 1);
    }
  }
  else
  {
    THTensor *gradOutputSample = THTensor_(new)();
    THTensor *inputSample = THTensor_(new)();
    int nBatchFrame = input->size[0];
    
    for(i = 0; i < nBatchFrame; i++)
    {
      THTensor_(select)(gradOutputSample, gradOutput, 0, i);
      THTensor_(select)(inputSample, input, 0, i);
      int nOutputSampleFrame = nOutputFrame;
      
      /* bias first */
      for(k = 0; k < nOutputFrame; k++)
      {
        THTensor_(select)(gradOutputWindow, gradOutputSample, 0, k);
        THTensor_(cadd)(gradBias, gradBias, scale, gradOutputWindow);
      }

      /* ouch */
      for(k = 0; nOutputSampleFrame > 0; k++)
      {
        long outputFrameStride = (kW-1)/dW+1;
        long inputFrameStride = outputFrameStride*dW;
        long nFrame = (nInputFrame-k*dW-kW)/inputFrameStride + 1;
        nOutputSampleFrame -= nFrame;

        THTensor_(setStorage2d)(inputWindow, inputSample->storage,
                                inputSample->storageOffset+k*dW*inputSample->size[1],
                                nFrame, inputFrameStride*inputSample->size[1],
                                kW*inputSample->size[1], 1);

        THTensor_(setStorage2d)(gradOutputWindow, gradOutputSample->storage, 
                                gradOutputSample->storageOffset + k*gradOutputSample->size[1],
                                nFrame, outputFrameStride*gradOutputSample->size[1],
                                gradOutputSample->size[1], 1);

        THTensor_(transpose)(gradOutputWindow, NULL, 0, 1);
        THTensor_(addmm)(gradWeight, 1, gradWeight, scale, gradOutputWindow, inputWindow);
        THTensor_(transpose)(gradOutputWindow, NULL, 0, 1);
      }
    }
    THTensor_(free)(gradOutputSample);
    THTensor_(free)(inputSample);
  }

  THTensor_(free)(gradOutputWindow);
  THTensor_(free)(inputWindow);
  THTensor_(free)(gradOutput);
  THTensor_(free)(input);

}

#endif
