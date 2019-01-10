#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Linear.c"
#else

void THNN_(Linear_updateAddBuffer)(
          THNNState *state,
          THTensor *input,
          THTensor *addBuffer)
{
  int64_t nframe = THTensor_(size)(input,0);
  int64_t nElement = THTensor_(nElement)(addBuffer);
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
  int64_t dim = THTensor_(nDimension)(input);
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
    int64_t nframe = THTensor_(size)(input,0);
    int64_t nElement = THTensor_(nElement)(output);
    THTensor_(resize2d)(output,nframe,THTensor_(size)(weight,0));
    if (THTensor_(nElement)(output) != nElement) {
      THTensor_(zero)(output);
    }
    THNN_(Linear_updateAddBuffer)(state,input,addBuffer);
    THTensor *tweight = THTensor_(new)();
    THTensor_(transpose)(tweight,weight,0,1);
    THTensor_(addmm)(output,0,output,1,input,tweight);
    THTensor_(free)(tweight);
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
    int64_t nElement = THTensor_(nElement)(gradInput);
    THTensor_(resizeAs)(gradInput,input);
    if (THTensor_(nElement)(gradInput) != nElement) {
      THTensor_(zero)(gradInput);
    }

    int64_t dim = THTensor_(nDimension)(input);
    if (dim == 1) {
      THTensor *tweight = THTensor_(new)();
      THTensor_(transpose)(tweight,weight,0,1);
      THTensor_(addmv)(gradInput,0,gradInput,1,tweight,gradOutput);
      THTensor_(free)(tweight);
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
          accreal scale_)
{
  real scale = TH_CONVERT_ACCREAL_TO_REAL(scale_);
  int64_t dim = THTensor_(nDimension)(input);
  if (dim == 1) {
    THTensor_(addr)(gradWeight,1,gradWeight,scale,gradOutput,input);
    if (bias) {
      THTensor_(cadd)(gradBias,gradBias,scale,gradOutput);
    }
  }
  else if (dim == 2) {
    THTensor *tgradOutput = THTensor_(new)();
    THTensor_(transpose)(tgradOutput,gradOutput,0,1);
    THTensor_(addmm)(gradWeight,1,gradWeight,scale,tgradOutput,input);
    if (bias) {
      THNN_(Linear_updateAddBuffer)(state,input,addBuffer);
      THTensor_(addmv)(gradBias,1,gradBias,scale,tgradOutput,addBuffer);
    }
    THTensor_(free)(tgradOutput);
  }
}

#endif
