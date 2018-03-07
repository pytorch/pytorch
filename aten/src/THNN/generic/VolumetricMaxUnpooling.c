#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricMaxUnpooling.c"
#else

static inline void THNN_(VolumetricMaxUnpooling_shapeCheck)(
                         THNNState *state,
                         THTensor *input,
                         THTensor *gradOutput,
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
  THNN_ARGCHECK(input->nDimension == 4 || input->nDimension == 5, 2, input,
                "4D or 5D (batch mode) tensor expected for input, but got: %s");

  THNN_CHECK_SHAPE_INDICES(input, indices);

  THArgCheck(dT > 0 && dW > 0 && dH > 0, 10,
             "stride should be greater than zero, but got dT: %d dH: %d dW: %d",
             dT, dH, dW);

  int dimw = 3;
  int dimh = 2;
  int dimt = 1;
  int dimn = 0;

  if (input->nDimension == 5)
  {
    dimt++;
    dimw++;
    dimh++;
    dimn++;
  }
  int nslices = input->size[dimn];

  if (gradOutput != NULL) {
    if (oT != gradOutput->size[dimt] || oW != gradOutput->size[dimw] || oH != gradOutput->size[dimh])
    {
      THError(
        "Inconsistent gradOutput size. oT= %d, oH= %d, oW= %d, gradOutput: %dx%dx%d",
        oT, oH, oW, gradOutput->size[dimt], gradOutput->size[dimh], gradOutput->size[dimw]
      );
    }

    THNN_CHECK_DIM_SIZE(gradOutput, input->nDimension, dimn, nslices);
  }
}

static void THNN_(VolumetricMaxUnpooling_updateOutput_frame)(
          real *input_p,
          real *output_p,
          THIndex_t *ind_p,
          int nslices,
          int iT,
          int iW,
          int iH,
          int oT,
          int oW,
          int oH)
{
  int k;
  int has_error = 0;
  THIndex_t error_index;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    real *output_p_k = output_p + k * oT * oH * oW;
    real *input_p_k = input_p + k * iT * iH * iW;
    THIndex_t *ind_p_k = ind_p + k * iT * iH * iW;

    int t, i, j, index;
    THIndex_t maxp;
    for (t = 0; t < iT; t++)
    {
      for (i = 0; i < iH; i++)
      {
        for (j = 0; j < iW; j++)
        {
          index = t * iH * iW + i * iW + j;
          maxp = ind_p_k[index] - TH_INDEX_BASE;  /* retrieve position of max */
          if (maxp < 0 || maxp >= oT * oW * oH)
          {
#pragma omp critical
            {
              has_error = 1;
              error_index = maxp;
            }
          } else {
            output_p_k[maxp] = input_p_k[index]; /* update output */
          }
        }
      }
    }
  }
  if (has_error) {
    THError(
        "found an invalid max index %ld (output volumes are of size %dx%dx%d)",
        error_index, oT, oH, oW
    );
  }
}

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
  int dimw = 3;
  int dimh = 2;
  int dimt = 1;
  int nbatch = 1;
  int nslices;
  int iT;
  int iH;
  int iW;
  real *input_data;
  real *output_data;
  THIndex_t *indices_data;

  THNN_(VolumetricMaxUnpooling_shapeCheck)(
        state, input, NULL, indices,
        oT, oW, oH, dT, dW, dH, pT, pW, pH);

  if (input->nDimension == 5)
  {
    nbatch = input->size[0];
    dimt++;
    dimw++;
    dimh++;
  }

  /* sizes */
  nslices = input->size[dimt-1];
  iT = input->size[dimt];
  iH = input->size[dimh];
  iW = input->size[dimw];

  /* get contiguous input */
  input = THTensor_(newContiguous)(input);
  indices = THIndexTensor_(newContiguous)(indices);

  /* resize output */
  if (input->nDimension == 4)
  {
    THTensor_(resize4d)(output, nslices, oT, oH, oW);
    THTensor_(zero)(output);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THIndexTensor_(data)(indices);

    THNN_(VolumetricMaxUnpooling_updateOutput_frame)(
      input_data, output_data,
      indices_data,
      nslices,
      iT, iW, iH,
      oT, oW, oH
    );
  }
  else
  {
    int p;

    THTensor_(resize5d)(output, nbatch, nslices, oT, oH, oW);
    THTensor_(zero)(output);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THIndexTensor_(data)(indices);

    for (p = 0; p < nbatch; p++)
    {
      THNN_(VolumetricMaxUnpooling_updateOutput_frame)(
        input_data+p*nslices*iT*iW*iH,
        output_data+p*nslices*oT*oW*oH,
        indices_data+p*nslices*iT*iW*iH,
        nslices,
        iT, iW, iH,
        oT, oW, oH
      );
    }
  }

  /* cleanup */
  THTensor_(free)(input);
  THIndexTensor_(free)(indices);
}

static void THNN_(VolumetricMaxUnpooling_updateGradInput_frame)(
          real *gradInput_p,
          real *gradOutput_p,
          THIndex_t *ind_p,
          int nslices,
          int iT,
          int iW,
          int iH,
          int oT,
          int oW,
          int oH)
{
  int k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    real *gradInput_p_k = gradInput_p + k * iT * iH * iW;
    real *gradOutput_p_k = gradOutput_p + k * oT * oH * oW;
    THIndex_t *ind_p_k = ind_p + k * iT * iH * iW;

    int t, i, j, index;
    THIndex_t maxp;
    for (t = 0; t < iT; t++)
    {
      for (i = 0; i < iH; i++)
      {
        for (j = 0; j < iW; j++)
        {
          index = t * iH * iW + i * iW  + j;
          maxp = ind_p_k[index] - TH_INDEX_BASE;  /* retrieve position of max */
          if (maxp < 0 || maxp >= oT * oH * oW)
          {
            THError("invalid max index %ld, oT= %d, oW= %d, oH= %d", maxp, oT, oW, oH);
          }
          gradInput_p_k[index] = gradOutput_p_k[maxp];  /* update gradient */
        }
      }
    }
  }
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
  int dimw = 3;
  int dimh = 2;
  int dimt = 1;
  int nbatch = 1;
  int nslices;
  int iT;
  int iH;
  int iW;
  real *gradInput_data;
  real *gradOutput_data;
  THIndex_t *indices_data;

  THNN_(VolumetricMaxUnpooling_shapeCheck)(
        state, input, gradOutput, indices,
        oT, oW, oH, dT, dW, dH, pT, pW, pH);

  // TODO: check gradOutput shape
  /* get contiguous gradOutput */
  gradOutput = THTensor_(newContiguous)(gradOutput);
  indices = THIndexTensor_(newContiguous)(indices);

  /* resize */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  if (input->nDimension == 5)
  {
    nbatch = input->size[0];
    dimt++;
    dimw++;
    dimh++;
  }

  /* sizes */
  nslices = input->size[dimt-1];
  iT = input->size[dimt];
  iH = input->size[dimh];
  iW = input->size[dimw];

  /* get raw pointers */
  gradInput_data = THTensor_(data)(gradInput);
  gradOutput_data = THTensor_(data)(gradOutput);
  indices_data = THIndexTensor_(data)(indices);

  /* backprop */
  if (input->nDimension == 4)
  {
    THNN_(VolumetricMaxUnpooling_updateGradInput_frame)(
      gradInput_data, gradOutput_data,
      indices_data,
      nslices,
      iT, iW, iH,
      oT, oW, oH
    );
  }
  else
  {
    int p;
    for (p = 0; p < nbatch; p++)
    {
      THNN_(VolumetricMaxUnpooling_updateGradInput_frame)(
        gradInput_data+p*nslices*iT*iW*iH,
        gradOutput_data+p*nslices*oT*oW*oH,
        indices_data+p*nslices*iT*iW*iH,
        nslices,
        iT, iW, iH,
        oT, oW, oH
      );
    }
  }

  /* cleanup */
  THTensor_(free)(gradOutput);
  THIndexTensor_(free)(indices);
}

#endif
