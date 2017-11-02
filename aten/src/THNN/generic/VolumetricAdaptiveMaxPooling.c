#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricAdaptiveMaxPooling.c"
#else

#define START_IND(a,b,c) (int)floor((float)(a * c) / b)
#define END_IND(a,b,c) (int)ceil((float)((a + 1) * c) / b)
// #define START_IND(a,b,c) a * c / b
// #define END_IND(a,b,c)  (a + 1) * c / b + ((a + 1) * c % b > 0)?1:0

// 5d tensor B x D x T x H x W

static void THNN_(VolumetricAdaptiveMaxPooling_updateOutput_frame)(
          real *input_p,
          real *output_p,
          THIndex_t *ind_p,
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
  int64_t d;
#pragma omp parallel for private(d)
  for (d = 0; d < sizeD; d++)
  {
    /* loop over output */
    int64_t ot, oh, ow;
    for(ot = 0; ot < osizeT; ot++)
    {
      int64_t istartT = START_IND(ot, osizeT, isizeT);
      int64_t iendT   = END_IND(ot, osizeT, isizeT);
      int64_t kT = iendT - istartT;

      for(oh = 0; oh < osizeH; oh++)
      {
        int64_t istartH = START_IND(oh, osizeH, isizeH);
        int64_t iendH   = END_IND(oh, osizeH, isizeH);
        int64_t kH = iendH - istartH;

        for(ow = 0; ow < osizeW; ow++)
        {

          int64_t istartW = START_IND(ow, osizeW, isizeW);
          int64_t iendW   = END_IND(ow, osizeW, isizeW);
          int64_t kW = iendW - istartW;

          /* local pointers */
          real *ip = input_p   + d*istrideD + istartT *istrideT + istartH*istrideH + istartW*istrideW;
          real *op = output_p  + d*osizeT*osizeH*osizeW + ot*osizeH*osizeW + oh*osizeW + ow;
          THIndex_t *indp = ind_p   + d*osizeT*osizeH*osizeW + ot*osizeH*osizeW + oh*osizeW + ow;

          /* compute local max: */
          int64_t maxindex = -1;
          real maxval = -FLT_MAX;
          int64_t it, ih, iw;
          for(it = 0; it < kT; it++)
          {
            for(ih = 0; ih < kH; ih++)
            {
              for(iw = 0; iw < kW; iw++)
              {
                real val = *(ip + it*istrideT + ih*istrideH + iw*istrideW);
                if (val > maxval)
                {
                  maxval = val;
                  maxindex = (it+istartT)*isizeH*isizeW + (ih+istartH)*isizeW + (iw+istartW);
                }
              }
            }
          }

          /* set output to local max */
          *op = maxval;

          /* store location of max */
          *indp = maxindex + TH_INDEX_BASE;
        }
      }
    }
  }
}

void THNN_(VolumetricAdaptiveMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          int osizeT,
          int osizeW,
          int osizeH)
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

  int64_t istrideB;
  int64_t istrideD;
  int64_t istrideT;
  int64_t istrideH;
  int64_t istrideW;

  real *input_data;
  real *output_data;
  THIndex_t *indices_data;

  THNN_ARGCHECK(input->nDimension == 4 || input->nDimension == 5, 2, input,
    "4D or 5D (batch mode) tensor expected for input, but got: %s");

  if (input->nDimension == 5)
  {
    istrideB = input->stride[0];
    sizeB = input->size[0];
    dimD++;
    dimT++;
    dimH++;
    dimW++;
  }

  /* sizes */
  sizeD  = input->size[dimD];
  isizeT = input->size[dimT];
  isizeH = input->size[dimH];
  isizeW = input->size[dimW];
  /* strides */
  istrideD = input->stride[dimD];
  istrideT = input->stride[dimT];
  istrideH = input->stride[dimH];
  istrideW = input->stride[dimW];

  /* resize output */
  if (input->nDimension == 4)
  {
    THTensor_(resize4d)(output, sizeD, osizeT, osizeH, osizeW);
    /* indices will contain max input locations for each output point */
    THIndexTensor_(resize4d)(indices, sizeD, osizeT, osizeH, osizeW);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THIndexTensor_(data)(indices);

    THNN_(VolumetricAdaptiveMaxPooling_updateOutput_frame)(input_data, output_data,
                                                      indices_data,
                                                      sizeD,
                                                      isizeT, isizeH, isizeW,
                                                      osizeT, osizeH, osizeW,
                                                      istrideD, istrideT,
                                                      istrideH, istrideW);
  }
  else
  {
    int64_t b;

    THTensor_(resize5d)(output, sizeB, sizeD, osizeT, osizeH, osizeW);
    /* indices will contain max input locations for each output point */
    THIndexTensor_(resize5d)(indices, sizeB, sizeD, osizeT, osizeH, osizeW);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THIndexTensor_(data)(indices);

#pragma omp parallel for private(b)
    for (b = 0; b < sizeB; b++)
    {
      THNN_(VolumetricAdaptiveMaxPooling_updateOutput_frame)(input_data+b*istrideB, output_data+b*sizeD*osizeT*osizeH*osizeW,
                                                        indices_data+b*sizeD*osizeT*osizeH*osizeW,
                                                        sizeD,
                                                        isizeT, isizeH, isizeW,
                                                        osizeT, osizeH, osizeW,
                                                        istrideD, istrideT,
                                                        istrideH, istrideW);
    }
  }
}

static void THNN_(VolumetricAdaptiveMaxPooling_updateGradInput_frame)(
          real *gradInput_p,
          real *gradOutput_p,
          THIndex_t *ind_p,
          int64_t sizeD,
          int64_t isizeT,
          int64_t isizeH,
          int64_t isizeW,
          int64_t osizeT,
          int64_t osizeH,
          int64_t osizeW)
{
  int64_t d;
#pragma omp parallel for private(d)
  for (d = 0; d < sizeD; d++)
  {
    real *gradInput_p_d = gradInput_p + d*isizeT*isizeH*isizeW;
    real *gradOutput_p_d = gradOutput_p + d*osizeT*osizeH*osizeW;
    THIndex_t *ind_p_d = ind_p + d*osizeT*osizeH*osizeW;

    /* calculate max points */
    int64_t ot, oh, ow;
    for(ot = 0; ot < osizeT; ot++)
    {
      for(oh = 0; oh < osizeH; oh++)
      {
        for(ow = 0; ow < osizeW; ow++)
        {
          /* retrieve position of max */
          int64_t maxp = ind_p_d[ot*osizeH*osizeW + oh*osizeW + ow] - TH_INDEX_BASE;

          /* update gradient */
          gradInput_p_d[maxp] += gradOutput_p_d[ot*osizeH*osizeW + oh*osizeW + ow];
        }
      }
    }
  }
}

void THNN_(VolumetricAdaptiveMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices)
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
  real *gradInput_data;
  real *gradOutput_data;
  THIndex_t *indices_data;

  /* get contiguous gradOutput */
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* resize */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  if (input->nDimension == 5) {
    sizeB = input->size[0];
    dimD++;
    dimT++;
    dimH++;
    dimW++;
  }

  /* sizes */
  sizeD  = input->size[dimD];
  isizeT = input->size[dimT];
  isizeH = input->size[dimH];
  isizeW = input->size[dimW];
  osizeT = gradOutput->size[dimT];
  osizeH = gradOutput->size[dimH];
  osizeW = gradOutput->size[dimW];

  /* get raw pointers */
  gradInput_data = THTensor_(data)(gradInput);
  gradOutput_data = THTensor_(data)(gradOutput);
  indices_data = THIndexTensor_(data)(indices);

  /* backprop */
  if (input->nDimension == 4)
  {
    THNN_(VolumetricAdaptiveMaxPooling_updateGradInput_frame)(gradInput_data, gradOutput_data,
                                                         indices_data,
                                                         sizeD,
                                                         isizeT, isizeH, isizeW,
                                                         osizeT, osizeH, osizeW);
  }
  else
  {
    int64_t b;
#pragma omp parallel for private(b)
    for (b = 0; b < sizeB; b++)
    {
      THNN_(VolumetricAdaptiveMaxPooling_updateGradInput_frame)(gradInput_data+b*sizeD*isizeT*isizeH*isizeW, gradOutput_data+b*sizeD*osizeT*osizeH*osizeW,
                                                           indices_data+b*sizeD*osizeT*osizeH*osizeW,
                                                           sizeD,
                                                           isizeT, isizeH, isizeW,
                                                           osizeT, osizeH, osizeW);
    }
  }

  /* cleanup */
  THTensor_(free)(gradOutput);
}

#endif
