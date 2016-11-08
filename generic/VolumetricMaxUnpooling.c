#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricMaxUnpooling.c"
#else

static void THNN_(VolumetricMaxUnpooling_updateOutput_frame)(
          real *input_p,
          real *output_p,
          THIndex_t *ind_p,
          long nslices,
          long iT,
          long iW,
          long iH,
          long oT,
          long oW,
          long oH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH)
{
  long k;
  int has_error = 0;
  long error_index;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    long ti, i, j, maxz, maxy, maxx;
    for (ti = 0; ti < iT; ti++)
    {
      for (i = 0; i < iH; i++)
      {
        for (j = 0; j < iW; j++)
        {
          long start_t = ti * dT - pT;
          long start_h = i * dH - pH;
          long start_w = j * dW - pW;

          //real *output_p_k = output_p + k*oT*oW*oH + ti*oW*oH*dT + i*oW*dH + j*dW;
          real *input_p_k = input_p + k*iT*iW*iH + ti*iW*iH + i*iW + j;
          THIndex_t *ind_p_k = ind_p + k*iT*iW*iH + ti*iW*iH + i*iW + j;

          maxz = ((unsigned char*)(ind_p_k))[0]; /* retrieve position of max */
          maxy = ((unsigned char*)(ind_p_k))[1];
          maxx = ((unsigned char*)(ind_p_k))[2];

          size_t idx = k*oT*oW*oH + oH*oW*(start_t+maxz) + oW*(start_h+maxy) + (start_w+maxx);
          if (start_t+maxz<0 || start_h+maxy<0 || start_w+maxx<0 || start_t+maxz>=oT || start_h+maxy>=oH || start_w+maxx>=oW)
          {
#pragma omp critical
            {
              has_error = 1;
              error_index = idx;
            }
          } else {
            output_p[idx] = *input_p_k; /* update output */
          }
        }
      }
    }
  }
  if (has_error) {
    THError(
        "found an invalid max index %ld (output volumes are of size %ldx%ldx%ld)",
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

  THNN_ARGCHECK(input->nDimension == 4 || input->nDimension == 5, 2, input,
		"4D or 5D (batch mode) tensor expected for input, but got: %s");

  THNN_CHECK_SHAPE_INDICES(input, indices);

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
      oT, oW, oH,
      dT, dW, dH, pT, pW, pH
    );
  }
  else
  {
    long p;

    THTensor_(resize5d)(output, nbatch, nslices, oT, oH, oW);
    THTensor_(zero)(output);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THIndexTensor_(data)(indices);

#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      THNN_(VolumetricMaxUnpooling_updateOutput_frame)(
        input_data+p*nslices*iT*iW*iH,
        output_data+p*nslices*oT*oW*oH,
        indices_data+p*nslices*iT*iW*iH,
        nslices,
        iT, iW, iH,
        oT, oW, oH,
        dT, dW, dH,
        pT, pW, pH
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
          long nslices,
          long iT,
          long iW,
          long iH,
          long oT,
          long oW,
          long oH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH)
{
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    long ti, i, j, maxz, maxy, maxx;
    for (ti = 0; ti < iT; ti++)
    {
      for (i = 0; i < iH; i++)
      {
        for (j = 0; j < iW; j++)
        {
          long start_t = ti * dT - pT;
          long start_h = i * dH - pH;
          long start_w = j * dW - pW;

          real *gradInput_p_k = gradInput_p + k*iT*iW*iH + ti*iW*iH + i*iW + j;
          //real *gradOutput_p_k = gradOutput_p + k*oT*oW*oH + ti*oW*oH*dT + i*oW*dH + j*dW;
          THIndex_t *ind_p_k = ind_p + k*iT*iW*iH + ti*iW*iH + i*iW + j;

          maxz = ((unsigned char*)(ind_p_k))[0]; /* retrieve position of max */
          maxy = ((unsigned char*)(ind_p_k))[1];
          maxx = ((unsigned char*)(ind_p_k))[2];

          if (start_t+maxz<0 || start_h+maxy<0 || start_w+maxx<0 || start_t+maxz>=oT || start_h+maxy>=oH || start_w+maxx>=oW)
          {
            THError(
              "invalid max index z= %d, y= %d, x= %d, oT= %d, oW= %d, oH= %d",
              start_t+maxz, start_h+maxy, start_w+maxx, oT, oW, oH
            );
          }
          *gradInput_p_k = gradOutput_p[k*oT*oW*oH + oH*oW*(start_t+maxz) + oW*(start_h+maxy) + (start_w+maxx)]; /* update gradient */
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

  THNN_CHECK_SHAPE_INDICES(input, indices);

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

  if (oT != gradOutput->size[dimt] || oW != gradOutput->size[dimw] || oH != gradOutput->size[dimh])
  {
    THError(
      "Inconsistent gradOutput size. oT= %d, oH= %d, oW= %d, gradOutput: %dx%d",
      oT, oH, oW,gradOutput->size[dimh], gradOutput->size[dimw]
    );
  }

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
      oT, oW, oH,
      dT, dW, dH,
      pT, pW, pH
    );
  }
  else
  {
    long p;
#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      THNN_(VolumetricMaxUnpooling_updateGradInput_frame)(
        gradInput_data+p*nslices*iT*iW*iH,
        gradOutput_data+p*nslices*oT*oW*oH,
        indices_data+p*nslices*iT*iW*iH,
        nslices,
        iT, iW, iH,
        oT, oW, oH,
        dT, dW, dH,
        pT, pW, pH
      );
    }
  }

  /* cleanup */
  THTensor_(free)(gradOutput);
  THIndexTensor_(free)(indices);
}

#endif
