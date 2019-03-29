#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/VolumetricUpSamplingNearest.c"
#else

void THNN_(VolumetricUpSamplingNearest_updateOutput)(
    THNNState* state,
    THTensor* input,
    THTensor* output,
    int outputDepth,
    int outputHeight,
    int outputWidth) {
  AT_ERROR("This function is deprecated, please use it from ATen.");
}

void THNN_(VolumetricUpSamplingNearest_updateGradInput)(
    THNNState* state,
    THTensor* gradOutput,
    THTensor* gradInput,
    int nbatch,
    int channels,
    int inputDepth,
    int inputHeight,
    int inputWidth,
    int outputDepth,
    int outputHeight,
    int outputWidth) {
  AT_ERROR("This function is deprecated, please use it from ATen.");
}

#endif
