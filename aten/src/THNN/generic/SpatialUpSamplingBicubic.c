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
  const int nbatch = THTensor_(size)(input, 0);
  const int channels = THTensor_(size)(input, 1);
  const int input_height = THTensor_(size)(input, 2);
  const int input_width = THTensor_(size)(input, 3);

  return;
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
  return;
}

#endif