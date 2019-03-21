#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/SpatialUpSamplingBicubic.c"
#else

void THNN_(SpatialUpSamplingBicubic_updateOutput)(
    THNNState* state,
    THTensor* input,
    THTensor* output,
    int output_height,
    int output_width,
    bool align_corners) {
  THError('This function is deprecated, please use it from ATen.');
}

void THNN_(SpatialUpSamplingBicubic_updateGradInput)(
    THNNState* state,
    THTensor* gradOutput,
    THTensor* gradInput,
    int nbatch,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    bool align_corners) {
  THError('This function is deprecated, please use it from ATen.');
}

#endif