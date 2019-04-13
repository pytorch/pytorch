#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/SpatialUpSamplingBilinear.c"
#else

void THNN_(SpatialUpSamplingBilinear_updateOutput)(
    THNNState* state,
    THTensor* input,
    THTensor* output,
    int outputHeight,
    int outputWidth,
    bool align_corners) {
  AT_ERROR("This function is deprecated, please use it from ATen.");
}

void THNN_(SpatialUpSamplingBilinear_updateGradInput)(
    THNNState* state,
    THTensor* gradOutput,
    THTensor* gradInput,
    int nbatch,
    int channels,
    int inputHeight,
    int inputWidth,
    int outputHeight,
    int outputWidth,
    bool align_corners) {
  AT_ERROR("This function is deprecated, please use it from ATen.");
}

#endif
