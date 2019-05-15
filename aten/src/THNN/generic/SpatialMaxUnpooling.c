#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/SpatialMaxUnpooling.c"
#else

#include <ATen/Parallel.h>
#include <mutex>

static void THNN_(SpatialMaxUnpooling_updateOutput_frame)(scalar_t *input_p, scalar_t *output_p,
                                                      THIndex_t *ind_p,
                                                      int nslices,
                                                      int iwidth, int iheight,
                                                      int owidth, int oheight)
{
  int has_error = 0;
  THIndex_t error_index = 0;
  std::mutex mutex;
  at::parallel_for(0, nslices, 0, [&](int64_t start, int64_t end) {
    for (auto k = start; k < end; k++)
    {
      scalar_t *output_p_k = output_p + k*owidth*oheight;
      scalar_t *input_p_k = input_p + k*iwidth*iheight;
      THIndex_t *ind_p_k = ind_p + k*iwidth*iheight;

      int i, j;
      THIndex_t maxp;
      for(i = 0; i < iheight; i++)
      {
        for(j = 0; j < iwidth; j++)
        {
          maxp = ind_p_k[i*iwidth + j];  /* retrieve position of max */
          if(maxp<0 || maxp>=owidth*oheight) {
            std::unique_lock<std::mutex> lock(mutex);
            has_error = 1;
            error_index = maxp;
          } else {
            output_p_k[maxp] = input_p_k[i*iwidth + j]; /* update output */
          }
        }
      }
    }
  });
  if (has_error) {
    THError("found an invalid max index %ld (output volumes are of size %dx%d)",
        error_index, oheight, owidth);
  }
}

void THNN_(SpatialMaxUnpooling_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    THIndexTensor *indices,
    int owidth, int oheight)
{
  int dimw = 2;
  int dimh = 1;
  int nbatch = 1;
  int nslices;
  int iheight;
  int iwidth;
  scalar_t *input_data;
  scalar_t *output_data;
  THIndex_t *indices_data;


  TORCH_CHECK(!input->is_empty() && (input->dim() == 3 || input->dim() == 4),
           "non-empty 3D or 4D (batch mode) tensor expected for input, but got sizes: ", input->sizes());
  THNN_CHECK_SHAPE_INDICES(input, indices);

  if (input->dim() == 4)
  {
    nbatch = input->size(0);
    dimw++;
    dimh++;
  }

  /* sizes */
  nslices = input->size(dimh-1);
  iheight = input->size(dimh);
  iwidth = input->size(dimw);

  /* get contiguous input and indices */
  input = THTensor_(newContiguous)(input);
  indices = THIndexTensor_(newContiguous)(indices);

  /* resize output */
  if (input->dim() == 3)
  {
    THTensor_(resize3d)(output, nslices, oheight, owidth);
    THTensor_(zero)(output);

    input_data = input->data<scalar_t>();
    output_data = output->data<scalar_t>();
    indices_data = THIndexTensor_(data)(indices);

    THNN_(SpatialMaxUnpooling_updateOutput_frame)(input_data, output_data,
                                              indices_data,
                                              nslices,
                                              iwidth, iheight,
                                              owidth, oheight);
  }
  else
  {
    int p;

    THTensor_(resize4d)(output, nbatch, nslices, oheight, owidth);
    THTensor_(zero)(output);

    input_data = input->data<scalar_t>();
    output_data = output->data<scalar_t>();
    indices_data = THIndexTensor_(data)(indices);

    for (p = 0; p < nbatch; p++)
    {
      THNN_(SpatialMaxUnpooling_updateOutput_frame)(
                                                    input_data+p*nslices*iwidth*iheight,
                                                    output_data+p*nslices*owidth*oheight,
                                                    indices_data+p*nslices*iwidth*iheight,
                                                    nslices,
                                                    iwidth, iheight,
                                                    owidth, oheight);
    }
  }

  /* cleanup */
  c10::raw::intrusive_ptr::decref(input);
  THIndexTensor_(free)(indices);
}

static void THNN_(SpatialMaxUnpooling_updateGradInput_frame)(scalar_t *gradInput_p, scalar_t *gradOutput_p,
                                                         THIndex_t *ind_p,
                                                         int nslices,
                                                         int iwidth, int iheight,
                                                         int owidth, int oheight)
{
  at::parallel_for(0, nslices, 0, [&](int64_t start, int64_t end) {
    for (auto k = start; k < end; k++)
    {
      scalar_t *gradInput_p_k = gradInput_p + k*iwidth*iheight;
      scalar_t *gradOutput_p_k = gradOutput_p + k*owidth*oheight;
      THIndex_t *ind_p_k = ind_p + k*iwidth*iheight;

      int i, j;
      THIndex_t maxp;
      for(i = 0; i < iheight; i++)
      {
        for(j = 0; j < iwidth; j++)
        {
          maxp = ind_p_k[i*iwidth + j]; /* retrieve position of max */
          if(maxp < 0 || maxp >= owidth * oheight) {
              THError("invalid max index %ld, owidth= %d, oheight= %d", maxp, owidth, oheight);
          }
          gradInput_p_k[i*iwidth + j] = gradOutput_p_k[maxp]; /* update gradient */
        }
      }
    }
  });
}

void THNN_(SpatialMaxUnpooling_updateGradInput)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradInput,
    THIndexTensor *indices,
    int owidth, int oheight)
{
  int dimw = 2;
  int dimh = 1;
  int nbatch = 1;
  int nslices;
  int iheight;
  int iwidth;
  scalar_t *gradInput_data;
  scalar_t *gradOutput_data;
  THIndex_t *indices_data;

  THNN_CHECK_SHAPE_INDICES(input, indices);

  /* get contiguous gradOutput and indices */
  gradOutput = THTensor_(newContiguous)(gradOutput);
  indices = THIndexTensor_(newContiguous)(indices);

  /* resize */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  if (input->dim() == 4) {
    nbatch = input->size(0);
    dimw++;
    dimh++;
  }

  /* sizes */
  nslices = input->size(dimh-1);
  iheight = input->size(dimh);
  iwidth = input->size(dimw);

  if(owidth!=gradOutput->size(dimw) || oheight!=gradOutput->size(dimh)){
    THError("Inconsistent gradOutput size. oheight= %d, owidth= %d, gradOutput: %dx%d",
            oheight, owidth, gradOutput->size(dimh), gradOutput->size(dimw));
  }

  /* get raw pointers */
  gradInput_data = gradInput->data<scalar_t>();
  gradOutput_data = gradOutput->data<scalar_t>();
  indices_data = THIndexTensor_(data)(indices);

  /* backprop */
  if (input->dim() == 3)
  {
    THNN_(SpatialMaxUnpooling_updateGradInput_frame)(gradInput_data, gradOutput_data,
                                                 indices_data,
                                                 nslices,
                                                 iwidth, iheight,
                                                 owidth, oheight);
  }
  else
  {
    int p;
    for (p = 0; p < nbatch; p++)
    {
      THNN_(SpatialMaxUnpooling_updateGradInput_frame)(gradInput_data+p*nslices*iwidth*iheight, gradOutput_data+p*nslices*owidth*oheight,
                                                   indices_data+p*nslices*iwidth*iheight,
                                                   nslices,
                                                   iwidth, iheight,
                                                   owidth, oheight);
    }
  }

  /* cleanup */
  c10::raw::intrusive_ptr::decref(gradOutput);
  THIndexTensor_(free)(indices);
}

#endif
