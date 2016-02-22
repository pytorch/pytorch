#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TemporalSubSampling.c"
#else

void THNN_(TemporalSubSampling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          int kW,
          int dW,
          int inputFrameSize)
{
  THTensor *outputFrame, *inputWindow;
  int nInputFrame, nOutputFrame;
  long k;
  
  THArgCheck( input->nDimension == 2, 2, "2D tensor expected");
  THArgCheck( input->size[1] == inputFrameSize, 2, "invalid input frame size");
  THArgCheck( input->size[0] >= kW, 2, "input sequence smaller than kernel size");

  outputFrame = THTensor_(new)();
  inputWindow = THTensor_(new)();

  nInputFrame = input->size[0];
  nOutputFrame = (nInputFrame - kW) / dW + 1;

  THTensor_(resize2d)(output,
                      nOutputFrame,
                      inputFrameSize);
  
  for(k = 0; k < nOutputFrame; k++)
  {
    THTensor_(narrow)(inputWindow, input, 0, k*dW, kW);
    THTensor_(select)(outputFrame, output, 0, k);
    THTensor_(sum)(outputFrame, inputWindow, 0);
    THTensor_(cmul)(outputFrame, outputFrame, weight);
    THTensor_(cadd)(outputFrame, outputFrame, 1, bias);
  }

  THTensor_(free)(outputFrame);
  THTensor_(free)(inputWindow);
}

void THNN_(TemporalSubSampling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          int kW,
          int dW)
{

  THTensor *gradOutputFrame;
  THTensor *gradInputWindow, *buffer, *kwunit;
  long k;

  gradOutputFrame = THTensor_(new)();
  gradInputWindow = THTensor_(new)();
  buffer = THTensor_(new)();
  kwunit = THTensor_(newWithSize1d)(kW);

  THTensor_(fill)(kwunit, 1);
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  for(k = 0; k < gradOutput->size[0]; k++)
  {
    THTensor_(narrow)(gradInputWindow, gradInput, 0, k*dW, kW);
    THTensor_(select)(gradOutputFrame, gradOutput, 0, k);
    THTensor_(cmul)(buffer, weight, gradOutputFrame);
    THTensor_(addr)(gradInputWindow, 1, gradInputWindow, 1, kwunit, buffer);
  }

  THTensor_(free)(gradOutputFrame);
  THTensor_(free)(gradInputWindow);
  THTensor_(free)(buffer);
  THTensor_(free)(kwunit);
}

void THNN_(TemporalSubSampling_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          int kW,
          int dW,
          real scale)
{
  THTensor *gradOutputFrame;
  THTensor *inputWindow, *buffer;
  long k;


  gradOutputFrame = THTensor_(new)();
  inputWindow = THTensor_(new)();
  buffer = THTensor_(new)();

  for(k = 0; k < gradOutput->size[0]; k++)
  {
    THTensor_(narrow)(inputWindow, input, 0, k*dW, kW);
    THTensor_(select)(gradOutputFrame, gradOutput, 0, k);
    THTensor_(sum)(buffer, inputWindow, 0);
    THTensor_(addcmul)(gradWeight, gradWeight, scale, buffer, gradOutputFrame);
    THTensor_(cadd)(gradBias, gradBias, scale, gradOutputFrame);
  }

  THTensor_(free)(gradOutputFrame);
  THTensor_(free)(inputWindow);
  THTensor_(free)(buffer);
}

#endif
