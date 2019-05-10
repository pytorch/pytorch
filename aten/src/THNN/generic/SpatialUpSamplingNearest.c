#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/SpatialUpSamplingNearest.c"
#else

void THNN_(SpatialUpSamplingNearest_updateOutput)(
    THNNState* state,
    THTensor* input,
    THTensor* output,
    int outputHeight,
    int outputWidth) {
  AT_ERROR("This function is deprecated, please use it from ATen.");
}

void THNN_(SpatialUpSamplingNearest_updateGradInput)(
    THNNState* state,
    THTensor* gradOutput,
    THTensor* gradInput,
    int nbatch,
    int channels,
    int inputHeight,
    int inputWidth,
    int outputHeight,
    int outputWidth) {
  AT_ERROR("This function is deprecated, please use it from ATen.");
}

#endif
