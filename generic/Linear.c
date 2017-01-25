#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Linear.c"
#else

void THNN_(Linear_updateAddBuffer)(
          THNNState *state,
          THTensor *input,
          THTensor *addBuffer)
{
  long nframe = THTensor_(size)(input,0);
  long nElement = THTensor_(nElement)(addBuffer);
  if (nElement != nframe) {
    THTensor_(resize1d)(addBuffer,nframe);
    THTensor_(fill)(addBuffer,1.0);
  }
}

void THNN_(Linear_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *addBuffer)
{
  long dim = THTensor_(nDimension)(input);
  if (dim == 1) {
    THTensor_(resize1d)(output,THTensor_(size)(weight,0));
    if (bias) {
      THTensor_(copy)(output,bias);
    }
    else {
      THTensor_(zero)(output);
    }
    THTensor_(addmv)(output,1,output,1,weight,input);
  }
  else if (dim == 2) {
    long nframe = THTensor_(size)(input,0);
    long nElement = THTensor_(nElement)(output);
    THTensor_(resize2d)(output,nframe,THTensor_(size)(weight,0));
    if (THTensor_(nElement)(output) != nElement) {
      THTensor_(zero)(output);
    }
    THNN_(Linear_updateAddBuffer)(state,input,addBuffer);
    THTensor_(transpose)(weight,weight,0,1);
    THTensor_(addmm)(output,0,output,1,input,weight);
    THTensor_(transpose)(weight,weight,0,1);
    if (bias) {
      THTensor_(addr)(output,1,output,1,addBuffer,bias);
    }
  }
}

void THNN_(Linear_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight)
{
  if (gradInput) {
    long nElement = THTensor_(nElement)(gradInput);
    THTensor_(resizeAs)(gradInput,input);
    if (THTensor_(nElement)(gradInput) != nElement) {
      THTensor_(zero)(gradInput);
    }

    long dim = THTensor_(nDimension)(input);
    if (dim == 1) {
      THTensor_(transpose)(weight,weight,0,1);
      THTensor_(addmv)(gradInput,0,gradInput,1,weight,gradOutput);
      THTensor_(transpose)(weight,weight,0,1);
    }
    else if (dim == 2) {
      THTensor_(addmm)(gradInput,0,gradInput,1,gradOutput,weight);
    }
  }
}

void THNN_(Linear_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *bias,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *addBuffer,
          real scale)
{
  long dim = THTensor_(nDimension)(input);
  if (dim == 1) {
    THTensor_(addr)(gradWeight,1,gradWeight,scale,gradOutput,input);
    if (bias) {
      THTensor_(cadd)(gradBias,gradBias,scale,gradOutput);
    }
  }
  else if (dim == 2) {
    THTensor_(transpose)(gradOutput,gradOutput,0,1);
    THTensor_(addmm)(gradWeight,1,gradWeight,scale,gradOutput,input);
    if (bias) {
      THNN_(Linear_updateAddBuffer)(state,input,addBuffer);
      THTensor_(addmv)(gradBias,1,gradBias,scale,gradOutput,addBuffer);
    }
    THTensor_(transpose)(gradOutput,gradOutput,0,1);
  }
}

#endif
