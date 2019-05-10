#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/VolumetricAdaptiveAveragePooling.c"
#else

#include <ATen/Parallel.h>

#define START_IND(a,b,c) (int)floor((float)(a * c) / b)
#define END_IND(a,b,c) (int)ceil((float)((a + 1) * c) / b)
// #define START_IND(a,b,c) a * c / b
// #define END_IND(a,b,c)  (a + 1) * c / b + ((a + 1) * c % b > 0)?1:0

// 5d tensor B x D x T x H x W

static void THNN_(VolumetricAdaptiveAveragePooling_updateOutput_frame)(
          scalar_t *input_p,
          scalar_t *output_p,
          int64_t sizeD,
          int64_t isizeT,
          int64_t isizeH,
          int64_t isizeW,
          int64_t osizeT,
          int64_t osizeH,
          int64_t osizeW,
          int64_t istrideD,
          int64_t istrideT,
          int64_t istrideH,
          int64_t istrideW)
{
  at::parallel_for(0, sizeD, 0, [&](int64_t start, int64_t end) {
    for (auto d = start; d < end; d++)
    {
      /* loop over output */
      int64_t ot, oh, ow;
      for(ot = 0; ot < osizeT; ot++)
      {
        int istartT = START_IND(ot, osizeT, isizeT);
        int iendT   = END_IND(ot, osizeT, isizeT);
        int kT = iendT - istartT;

        for(oh = 0; oh < osizeH; oh++)
        {
          int istartH = START_IND(oh, osizeH, isizeH);
          int iendH   = END_IND(oh, osizeH, isizeH);
          int kH = iendH - istartH;

          for(ow = 0; ow < osizeW; ow++)
          {

            int istartW = START_IND(ow, osizeW, isizeW);
            int iendW   = END_IND(ow, osizeW, isizeW);
            int kW = iendW - istartW;

            /* local pointers */
            scalar_t *ip = input_p  + d*istrideD + istartT*istrideT + istartH*istrideH + istartW*istrideW;
            scalar_t *op = output_p + d*osizeT*osizeH*osizeW + ot*osizeH*osizeW + oh*osizeW + ow;

            /* compute local average: */
            scalar_t sum = 0;
            int it, ih, iw;
            for(it = 0; it < kT; it++)
            {
              for(ih = 0; ih < kH; ih++)
              {
                for(iw = 0; iw < kW; iw++)
                {
                  scalar_t val = *(ip + it*istrideT + ih*istrideH + iw*istrideW);
                  sum += val;
                }
              }
            }

            /* set output to local average */
            *op = sum / kT / kH / kW;
          }
        }
      }
    }
  });
}

void THNN_(VolumetricAdaptiveAveragePooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int osizeT,
          int osizeW,
          int osizeH)
{
  int dimD = 0;
  int dimT = 1;
  int dimH = 2;
  int dimW = 3;
  int64_t sizeB = 1;
  int64_t sizeD = 0;
  int64_t isizeT = 0;
  int64_t isizeH = 0;
  int64_t isizeW = 0;

  int64_t istrideB = 0;
  int64_t istrideD = 0;
  int64_t istrideT = 0;
  int64_t istrideH = 0;
  int64_t istrideW = 0;

  scalar_t *input_data = nullptr;
  scalar_t *output_data = nullptr;


  THNN_ARGCHECK(!input->is_empty() && (input->dim() == 4 || input->dim() == 5), 2, input,
                "non-empty 4D or 5D (batch mode) tensor expected for input, but got: %s");

  if (input->dim() == 5)
  {
    istrideB = input->stride(0);
    sizeB = input->size(0);
    dimD++;
    dimT++;
    dimH++;
    dimW++;
  }

  /* sizes */
  sizeD  = input->size(dimD);
  isizeT = input->size(dimT);
  isizeH = input->size(dimH);
  isizeW = input->size(dimW);
  /* strides */
  istrideD = input->stride(dimD);
  istrideT = input->stride(dimT);
  istrideH = input->stride(dimH);
  istrideW = input->stride(dimW);

  /* resize output */
  if (input->dim() == 4)
  {
    THTensor_(resize4d)(output, sizeD, osizeT, osizeH, osizeW);

    input_data = input->data<scalar_t>();
    output_data = output->data<scalar_t>();

    THNN_(VolumetricAdaptiveAveragePooling_updateOutput_frame)(input_data, output_data,
                                                      sizeD,
                                                      isizeT, isizeH, isizeW,
                                                      osizeT, osizeH, osizeW,
                                                      istrideD, istrideT,
                                                      istrideH, istrideW);
  }
  else
  {
    THTensor_(resize5d)(output, sizeB, sizeD, osizeT, osizeH, osizeW);

    input_data = input->data<scalar_t>();
    output_data = output->data<scalar_t>();

    at::parallel_for(0, sizeB, 0, [&](int64_t start, int64_t end) {
      for (auto b = start; b < end; b++)
      {
        THNN_(VolumetricAdaptiveAveragePooling_updateOutput_frame)(input_data+b*istrideB, output_data+b*sizeD*osizeT*osizeH*osizeW,
                                                          sizeD,
                                                          isizeT, isizeH, isizeW,
                                                          osizeT, osizeH, osizeW,
                                                          istrideD, istrideT,
                                                          istrideH, istrideW);
      }
    });
  }
}

static void THNN_(VolumetricAdaptiveAveragePooling_updateGradInput_frame)(
          scalar_t *gradInput_p,
          scalar_t *gradOutput_p,
          int64_t sizeD,
          int64_t isizeT,
          int64_t isizeH,
          int64_t isizeW,
          int64_t osizeT,
          int64_t osizeH,
          int64_t osizeW)
{
  at::parallel_for(0, sizeD, 0, [&](int64_t start, int64_t end) {
    for (auto d = start; d < end; d++)
    {
      scalar_t *gradInput_p_d  = gradInput_p + d*isizeT*isizeW*isizeH;
      scalar_t *gradOutput_p_d = gradOutput_p + d*osizeT*osizeW*osizeH;

      /* calculate average */
      int64_t ot, oh, ow;
      for(ot = 0; ot < osizeT; ot++)
      {
        int istartT = START_IND(ot, osizeT, isizeT);
        int iendT   = END_IND(ot, osizeT, isizeT);
        int kT = iendT - istartT;

        for(oh = 0; oh < osizeH; oh++)
        {
          int istartH = START_IND(oh, osizeH, isizeH);
          int iendH   = END_IND(oh, osizeH, isizeH);
          int kH = iendH - istartH;

          for(ow = 0; ow < osizeW; ow++)
          {

            int istartW = START_IND(ow, osizeW, isizeW);
            int iendW   = END_IND(ow, osizeW, isizeW);
            int kW = iendW - istartW;

            scalar_t grad_delta = gradOutput_p_d[ot*osizeH*osizeW + oh*osizeW + ow] / kT / kH / kW;

            int it, ih, iw;
            for(it = istartT; it < iendT; it++)
            {
              for(ih = istartH; ih < iendH; ih++)
              {
                for(iw = istartW; iw < iendW; iw++)
                {
                  /* update gradient */
                  gradInput_p_d[it*isizeH*isizeW + ih*isizeW + iw] += grad_delta;
                }
              }
            }
          }
        }
      }
    }
  });
}

void THNN_(VolumetricAdaptiveAveragePooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput)
{
  int dimD = 0;
  int dimT = 1;
  int dimH = 2;
  int dimW = 3;
  int64_t sizeB = 1;
  int64_t sizeD;
  int64_t isizeT;
  int64_t isizeH;
  int64_t isizeW;
  int64_t osizeT;
  int64_t osizeH;
  int64_t osizeW;
  scalar_t *gradInput_data;
  scalar_t *gradOutput_data;

  /* get contiguous gradOutput */
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* resize */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  if (input->dim() == 5) {
    sizeB = input->size(0);
    dimD++;
    dimT++;
    dimH++;
    dimW++;
  }

  /* sizes */
  sizeD  = input->size(dimD);
  isizeT = input->size(dimT);
  isizeH = input->size(dimH);
  isizeW = input->size(dimW);
  osizeT = gradOutput->size(dimT);
  osizeH = gradOutput->size(dimH);
  osizeW = gradOutput->size(dimW);

  /* get raw pointers */
  gradInput_data = gradInput->data<scalar_t>();
  gradOutput_data = gradOutput->data<scalar_t>();

  /* backprop */
  if (input->dim() == 4)
  {
    THNN_(VolumetricAdaptiveAveragePooling_updateGradInput_frame)(gradInput_data, gradOutput_data,
                                                         sizeD,
                                                         isizeT, isizeH, isizeW,
                                                         osizeT, osizeH, osizeW);
  }
  else
  {
    at::parallel_for(0, sizeB, 0, [&](int64_t start, int64_t end) {
      for (auto b = start; b < end; b++)
      {
        THNN_(VolumetricAdaptiveAveragePooling_updateGradInput_frame)(gradInput_data+b*sizeD*isizeT*isizeH*isizeW, gradOutput_data+b*sizeD*osizeT*osizeH*osizeW,
                                                             sizeD,
                                                             isizeT, isizeH, isizeW,
                                                             osizeT, osizeH, osizeW);
      }
    });
  }

  /* cleanup */
  c10::raw::intrusive_ptr::decref(gradOutput);
}

#endif

#undef START_IND
#undef END_IND
