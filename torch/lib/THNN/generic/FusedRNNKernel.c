#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/FusedRNNKernel.c"
#else

//add then sigmoid
//returns sigmoid(A+B)

void THNN_(GRUFused_updateOutput)(
          THNNState *state,
          THTensor *input,
	  THTensor *hidden,
	  THTensor *bias1,
	  THTensor *bias2,
	  THTensor *prevHidden,
	  THTensor *output)
{
  THAssertMsg(false, "Not implemented for CPU");
}

void THNN_(GRUFused_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *hidden,
          THTensor *gradOutput,
          THTensor *gradInput)
{
  THAssertMsg(false, "Not implemented for CPU");
}

#endif
