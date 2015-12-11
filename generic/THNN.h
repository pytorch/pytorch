#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THNN.h"
#else

#ifndef THIndexTensor
#define THIndexTensor THLongTensor
#endif

TH_API void THNN_(Abs_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(Abs_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput);

#endif
