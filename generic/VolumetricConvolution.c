#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricConvolution.c"
#else

void THNN_(VolumetricConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,     // only used by cuda impl
          THTensor *fgradInput, // only used by cuda impl
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH)
{
  THArgCheck(pT != 0 || pW != 0 || pH != 0, 9, "padding not supported by CPU backend");   // sharing signature with CUDA version

  THNN_ARGCHECK(input->nDimension == 4 || input->nDimension == 5, 2, input,
		"4D or 5D (batch mode) tensor expected for input, but got: %s");

  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  if (input->nDimension == 5)
  {
    dimt++;
    dimh++;
    dimw++;
  }

  long nOutputPlane = weight->size[0];
  long kT           = weight->size[2];
  long kH           = weight->size[3];
  long kW           = weight->size[4];
  long inputDepth   = input->size[dimt];
  long inputHeight  = input->size[dimh];
  long inputWidth   = input->size[dimw];
  long outputDepth  = (inputDepth - kT) / dT + 1;
  long outputWidth  = (inputWidth - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;
  THTensor *outn = THTensor_(new)();
  long i, j;
  if (input->nDimension == 4) /* non-batch mode */
  {
    THTensor_(resize4d)(output, nOutputPlane, outputDepth, outputHeight, outputWidth);

    /* add bias */
    for (i = 0; i < bias->size[0]; i++)
    {
      THTensor_(select)(outn, output, 0, i);
      THTensor_(fill)(outn, THTensor_(get1d)(bias, i));
    }

    /* do convolutions */
    THTensor_(conv3Dmv)(output, 1.0, 1.0, input, weight, dT, dH, dW, "V", "X");
  }
  else /* batch mode */
  {
    long nBatch = input->size[0];
    THTensor_(resize5d)(output, nBatch, nOutputPlane, outputDepth, outputHeight, outputWidth);
    THTensor *inb = THTensor_(new)();
    THTensor *outb = THTensor_(new)();

    /* loop over batches */
    for (j = 0; j < nBatch; j++)
    {
      THTensor_(select)(inb, input, 0, j);
      THTensor_(select)(outb, output, 0, j);

      /* add bias */
      for (i = 0; i < bias->size[0]; i++)
      {
        THTensor_(select)(outn, outb, 0, i);
        THTensor_(fill)(outn, THTensor_(get1d)(bias, i));
      }

      /* do convolutions */
      THTensor_(conv3Dmv)(outb, 1.0, 1.0, inb, weight, dT, dH, dW, "V", "X");
    }

    THTensor_(free)(inb);
    THTensor_(free)(outb);
  }
  THTensor_(free)(outn);
}

void THNN_(VolumetricConvolution_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput, // only used by cuda impl
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH)
{
  THArgCheck(pT != 0 || pW != 0 || pH != 0, 9, "padding not supported by CPU backend");   // sharing signature with CUDA version

  THNN_ARGCHECK(weight->nDimension == 5, 4, weight,
		"5D (nOutputPlane x nInputPlane x kT x kH x kW) tensor "
		"expected for weight, but got: %s");

  int nOutputPlane = (int)weight->size[0];

  THNN_ARGCHECK(gradOutput->nDimension == 4 || gradOutput->nDimension == 5, 3,
		gradOutput,
		"4D or 5D (batch mode) tensor expected for gradOutput, but got: %s");

  int dimPlane = 0;
  if (gradOutput->nDimension == 5)
  {
    dimPlane++;
  }

  THArgCheck(nOutputPlane == gradOutput->size[dimPlane], 1,
    "Number of output features is not equal to nOutputPlane"
  );

  /* gradient to input */
  THTensor *tweight = THTensor_(newTranspose)(weight, 0, 1);
  if (gradOutput->nDimension == 4) /* non-batch mode */
  {
    THTensor_(conv3Dmv)(gradInput, 0.0, 1.0, gradOutput, tweight, dT, dH, dW, "F", "C");
  }
  else /* batch mode */
  {
    long nBatch = gradOutput->size[0];
    THTensor *ginpb = THTensor_(new)();
    THTensor *goutb = THTensor_(new)();
    long j;

    THTensor_(resize5d)(gradInput,
      input->size[0], input->size[1], input->size[2], input->size[3], input->size[4]
    );

    /* loop over batches */
    for (j = 0; j < nBatch; j++)
    {
      THTensor_(select)(ginpb, gradInput, 0, j);
      THTensor_(select)(goutb, gradOutput, 0, j);
      THTensor_(conv3Dmv)(ginpb, 0.0, 1.0, goutb, tweight, dT, dH, dW, "F", "C");
    }
    THTensor_(free)(ginpb);
    THTensor_(free)(goutb);
  }

  THTensor_(free)(tweight);
}

void THNN_(VolumetricConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,     // only used by cuda impl
          THTensor *fgradInput, // only used by cuda impl
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH,
          real scale)
{
  THArgCheck(pT != 0 || pW != 0 || pH != 0, 9, "padding not supported by CPU backend");   // sharing signature with CUDA version

  THNN_ARGCHECK(gradWeight->nDimension == 5, 4, gradWeight,
		"5D (nOutputPlane x nInputPlane x kT x kH x kW) tensor "
		"expected for gradWeight, but got: %s");

  int nOutputPlane = (int)gradWeight->size[0];

  THArgCheck(gradBias->nDimension == 1 && gradBias->size[0] == nOutputPlane, 5,
    "gradBias tensor has wrong size"
  );

  long k;
  real *gradBias_data;
  THTensor *gradOutSlice;
  int dimPlane = 0;
  if (gradOutput->nDimension == 5)
  {
    dimPlane++;
  }

  THArgCheck(nOutputPlane == gradOutput->size[dimPlane], 1,
    "Number of output features is not equal to nOutputPlane"
  );

  if (gradOutput->nDimension == 4) /* non-batch mode */
  {
    /* gradient to bias */
    gradBias_data = THTensor_(data)(gradBias);
    gradOutSlice = THTensor_(new)();
    for (k = 0; k < nOutputPlane; k++)
    {
      THTensor_(select)(gradOutSlice, gradOutput, 0, k);
      gradBias_data[k] += scale * THTensor_(sumall)(gradOutSlice);
    }
    THTensor_(free)(gradOutSlice);

    /* gradient to kernels */
    THTensor_(conv3DRevger)(gradWeight, 1.0, scale, input, gradOutput, dT, dH, dW);
  }
  else /* batch mode */
  {
    long nBatch = gradOutput->size[0];
    THTensor *inpb = THTensor_(new)();
    THTensor *goutb = THTensor_(new)();
    long j;

    /* loop over batches */
    for (j = 0; j < nBatch; j++)
    {
      THTensor_(select)(inpb, input, 0, j);
      THTensor_(select)(goutb, gradOutput, 0, j);

      /* gradient to bias */
      gradBias_data = THTensor_(data)(gradBias);
      gradOutSlice = THTensor_(new)();
      for (k = 0; k < nOutputPlane; k++)
      {
        THTensor_(select)(gradOutSlice, goutb, 0, k);
        gradBias_data[k] += scale * THTensor_(sumall)(gradOutSlice);
      }
      THTensor_(free)(gradOutSlice);

      /* gradient to kernels */
      THTensor_(conv3DRevger)(gradWeight, 1.0, scale, inpb, goutb, dT, dH, dW);
    }
    THTensor_(free)(inpb);
    THTensor_(free)(goutb);
  }
}

#endif
