// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/TemporalUpSamplingLinear.c"
#else

void THNN_(TemporalUpSamplingLinear_updateOutput)(
    THNNState* state,
    THTensor* input,
    THTensor* output,
    int outputWidth,
    bool align_corners) {
  return;
}

void THNN_(TemporalUpSamplingLinear_updateGradInput)(
    THNNState* state,
    THTensor* gradOutput,
    THTensor* gradInput,
    int nbatch,
    int channels,
    int inputWidth,
    int outputWidth,
    bool align_corners) {
  return;
}

#endif