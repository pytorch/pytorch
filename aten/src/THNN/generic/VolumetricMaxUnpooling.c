#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricMaxUnpooling.c"
#else

void THNN_(VolumetricMaxUnpooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          int oT,
          int oW,
          int oH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH)
{
    THError("VolumetricMaxUnpooling_updateOutput has been deprecated");
}

void THNN_(VolumetricMaxUnpooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          int oT,
          int oW,
          int oH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH)
{
    THError("VolumetricMaxUnpooling_shapeCheck has been deprecated");
}

#endif
