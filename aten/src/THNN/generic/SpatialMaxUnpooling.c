#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialMaxUnpooling.c"
#else

void THNN_(SpatialMaxUnpooling_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    THIndexTensor *indices,
    int owidth, int oheight)
{
  THError("SpatialMaxUnpooling_updateOutput has been deprecated");
}

void THNN_(SpatialMaxUnpooling_updateGradInput)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradInput,
    THIndexTensor *indices,
    int owidth, int oheight)
{
  THError("SpatialMaxUnpooling_updateGradInput has been deprecated");
}

#endif
