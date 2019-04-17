// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/VolumetricUpSamplingTrilinear.c"
#else

void THNN_(VolumetricUpSamplingTrilinear_updateOutput)(
    THNNState* state,
    THTensor* input,
    THTensor* output,
    int outputDepth,
    int outputHeight,
    int outputWidth,
    bool align_corners) {
  AT_ERROR("This function is deprecated, please use it from ATen.");
}

void THNN_(VolumetricUpSamplingTrilinear_updateGradInput)(
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
    int outputWidth,
    bool align_corners) {
  AT_ERROR("This function is deprecated, please use it from ATen.");
}

#endif
