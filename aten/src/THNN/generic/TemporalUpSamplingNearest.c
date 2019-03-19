#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/TemporalUpSamplingNearest.c"
#else

void THNN_(TemporalUpSamplingNearest_updateOutput)(
    THNNState* state,
    THTensor* input,
    THTensor* output,
    int outputWidth) {
  return;
}

void THNN_(TemporalUpSamplingNearest_updateGradInput)(
    THNNState* state,
    THTensor* gradOutput,
    THTensor* gradInput,
    int nbatch,
    int channels,
    int inputWidth,
    int outputWidth) {
  return;
}

#endif