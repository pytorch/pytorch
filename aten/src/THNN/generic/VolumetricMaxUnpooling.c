#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/VolumetricMaxUnpooling.c"
#else

#include <ATen/Parallel.h>
#include <mutex>

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
  THNN_ARGCHECK(!input->is_empty() && (input->dim() == 4 || input->dim() == 5), 2, input,
                "non-empty 4D or 5D (batch mode) tensor expected for input, but got: %s");

  THNN_CHECK_SHAPE_INDICES(input, indices);

  THArgCheck(dT > 0 && dW > 0 && dH > 0, 10,
             "stride should be greater than zero, but got dT: %d dH: %d dW: %d",
             dT, dH, dW);

  int dimw = 3;
  int dimh = 2;
  int dimt = 1;
  int dimn = 0;

  if (input->dim() == 5)
  {
    dimt++;
    dimw++;
    dimh++;
    dimn++;
  }
  int nslices = input->size(dimn);

  if (gradOutput != NULL) {
    if (oT != gradOutput->size(dimt) || oW != gradOutput->size(dimw) || oH != gradOutput->size(dimh))
    {
      THError(
        "Inconsistent gradOutput size. oT= %d, oH= %d, oW= %d, gradOutput: %dx%dx%d",
        oT, oH, oW, gradOutput->size(dimt), gradOutput->size(dimh), gradOutput->size(dimw)
      );
    }

    THNN_CHECK_DIM_SIZE(gradOutput, input->dim(), dimn, nslices);
  }
}

static void THNN_(VolumetricMaxUnpooling_updateOutput_frame)(
          scalar_t *input_p,
          scalar_t *output_p,
          THIndex_t *ind_p,
          int nslices,
          int iT,
          int iW,
          int iH,
          int oT,
          int oW,
          int oH)
{
  int has_error = 0;
  THIndex_t error_index = 0;
  std::mutex mutex;
  at::parallel_for(0, nslices, 0, [&](int64_t start, int64_t end) {
    for (auto k = start; k < end; k++)
    {
      scalar_t *output_p_k = output_p + k * oT * oH * oW;
      scalar_t *input_p_k = input_p + k * iT * iH * iW;
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
            maxp = ind_p_k[index];  /* retrieve position of max */
            if (maxp < 0 || maxp >= oT * oW * oH)
            {
              std::unique_lock<std::mutex> lock(mutex);
              has_error = 1;
              error_index = maxp;
            } else {
              output_p_k[maxp] = input_p_k[index]; /* update output */
            }
          }
        }
      }
    }
  });
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
  scalar_t *input_data;
  scalar_t *output_data;
  THIndex_t *indices_data;

  THNN_(VolumetricMaxUnpooling_shapeCheck)(
        state, input, NULL, indices,
        oT, oW, oH, dT, dW, dH, pT, pW, pH);

  if (input->dim() == 5)
  {
    nbatch = input->size(0);
    dimt++;
    dimw++;
    dimh++;
  }

  /* sizes */
  nslices = input->size(dimt-1);
  iT = input->size(dimt);
  iH = input->size(dimh);
  iW = input->size(dimw);

  /* get contiguous input */
  input = THTensor_(newContiguous)(input);
  indices = THIndexTensor_(newContiguous)(indices);

  /* resize output */
  if (input->dim() == 4)
  {
    THTensor_(resize4d)(output, nslices, oT, oH, oW);
    THTensor_(zero)(output);

    input_data = input->data<scalar_t>();
    output_data = output->data<scalar_t>();
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

    input_data = input->data<scalar_t>();
    output_data = output->data<scalar_t>();
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
  c10::raw::intrusive_ptr::decref(input);
  THIndexTensor_(free)(indices);
}

static void THNN_(VolumetricMaxUnpooling_updateGradInput_frame)(
          scalar_t *gradInput_p,
          scalar_t *gradOutput_p,
          THIndex_t *ind_p,
          int nslices,
          int iT,
          int iW,
          int iH,
          int oT,
          int oW,
          int oH)
{
  at::parallel_for(0, nslices, 0, [&](int64_t start, int64_t end) {
    for (auto k = start; k < end; k++)
    {
      scalar_t *gradInput_p_k = gradInput_p + k * iT * iH * iW;
      scalar_t *gradOutput_p_k = gradOutput_p + k * oT * oH * oW;
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
            maxp = ind_p_k[index];  /* retrieve position of max */
            if (maxp < 0 || maxp >= oT * oH * oW)
            {
              THError("invalid max index %ld, oT= %d, oW= %d, oH= %d", maxp, oT, oW, oH);
            }
            gradInput_p_k[index] = gradOutput_p_k[maxp];  /* update gradient */
          }
        }
      }
    }
  });
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
  scalar_t *gradInput_data;
  scalar_t *gradOutput_data;
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

  if (input->dim() == 5)
  {
    nbatch = input->size(0);
    dimt++;
    dimw++;
    dimh++;
  }

  /* sizes */
  nslices = input->size(dimt-1);
  iT = input->size(dimt);
  iH = input->size(dimh);
  iW = input->size(dimw);

  /* get raw pointers */
  gradInput_data = gradInput->data<scalar_t>();
  gradOutput_data = gradOutput->data<scalar_t>();
  indices_data = THIndexTensor_(data)(indices);

  /* backprop */
  if (input->dim() == 4)
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
  c10::raw::intrusive_ptr::decref(gradOutput);
  THIndexTensor_(free)(indices);
}

#endif
