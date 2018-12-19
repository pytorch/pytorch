#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/SpatialFractionalMaxPooling.c"
#else

void THNN_(SpatialFractionalMaxPooling_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    int outputW, int outputH,
    int poolSizeW, int poolSizeH,
    THIndexTensor *indices,
    THTensor *randomSamples) {
  perror("Error: CPU implementation for FractionalMaxPool2d using THNN is \
      deprecated.");

}

void THNN_(SpatialFractionalMaxPooling_updateGradInput)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradInput,
    int outputW, int outputH,
    int poolSizeW, int poolSizeH,
    THIndexTensor *indices) {
      perror("Error: CPU implementation for FractionalMaxPool2d using THNN is \
          deprecated.");
}

#endif
