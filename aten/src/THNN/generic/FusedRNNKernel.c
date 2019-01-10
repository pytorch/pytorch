#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/FusedRNNKernel.c"
#else

void THNN_(GRUFused_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *hidden,
          THTensor *bias1,
          THTensor *bias2,
          THTensor *hx,
          THTensor *hy,
          THTensor *storage)
{
  THAssertMsg(false, "Not implemented for CPU");
}

void THNN_(GRUFused_updateGradInput)(
          THNNState *state,
          THTensor *gradInInput,
          THTensor *gradInHidden,
          THTensor *gradOutput,
          THTensor *gradInputHx,
          THTensor *storage)
{
  THAssertMsg(false, "Not implemented for CPU");
}

void THNN_(LSTMFused_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *hidden,
          THTensor *bias1,
          THTensor *bias2,
          THTensor *cx,
          THTensor *hy,
          THTensor *cy)
{
  THAssertMsg(false, "Not implemented for CPU");
}

void THNN_(LSTMFused_updateGradInput)(
          THNNState *state,
          THTensor *storage,
          THTensor *gradInGates,
          THTensor *prevC,
          THTensor *cy,
          THTensor *gradOutput,
          THTensor *gradOutputCell,
          THTensor *gradInputCx)
{
  THAssertMsg(false, "Not implemented for CPU");
}

#endif
