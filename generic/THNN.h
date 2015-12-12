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

TH_API void THNN_(AbsCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          real *output,
          bool sizeAverage);
TH_API void THNN_(AbsCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradInput,
          bool sizeAverage);

#endif
