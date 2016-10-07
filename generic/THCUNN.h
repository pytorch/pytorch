#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCUNN.h"
#else

TH_API void THNN_(SoftPlus_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  real beta,
                  real threshold);

TH_API void THNN_(SoftPlus_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *output,
                  real beta,
                  real threshold);

#endif
