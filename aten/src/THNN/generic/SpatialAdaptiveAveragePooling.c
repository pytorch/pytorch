#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialAdaptiveAveragePooling.c"
#else

#define START_IND(a,b,c) (int)floor((float)(a * c) / b)
#define END_IND(a,b,c) (int)ceil((float)((a + 1) * c) / b)
// #define START_IND(a,b,c) a * c / b
// #define END_IND(a,b,c)  (a + 1) * c / b + ((a + 1) * c % b > 0)?1:0

// 4d tensor B x D x H x W

static void THNN_(SpatialAdaptiveAveragePooling_updateOutput_frame)(
          scalar_t *input_p,
          scalar_t *output_p,
          int64_t sizeD,
          int64_t isizeH,
          int64_t isizeW,
          int64_t osizeH,
          int64_t osizeW,
          int64_t istrideD,
          int64_t istrideH,
          int64_t istrideW)
{
  int64_t d;
#pragma omp parallel for private(d)
  for (d = 0; d < sizeD; d++)
  {
    /* loop over output */
    int64_t oh, ow;
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
        scalar_t *ip = input_p   + d*istrideD + istartH*istrideH + istartW*istrideW;
        scalar_t *op = output_p  + d*osizeH*osizeW + oh*osizeW + ow;

        /* compute local average: */
        scalar_t sum = 0;
        int ih, iw;
        for(ih = 0; ih < kH; ih++)
        {
          for(iw = 0; iw < kW; iw++)
          {
            scalar_t val = *(ip + ih*istrideH + iw*istrideW);
            sum += val;
          }
        }

        /* set output to local average */
        *op = sum / kW / kH;
      }
    }
  }
}

void THNN_(SpatialAdaptiveAveragePooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int osizeW,
          int osizeH)
{
  int dimD = 0;
  int dimH = 1;
  int dimW = 2;
  int64_t sizeB = 1;
  int64_t sizeD = 0;
  int64_t isizeH = 0;
  int64_t isizeW = 0;

  int64_t istrideB = 0;
  int64_t istrideD = 0;
  int64_t istrideH = 0;
  int64_t istrideW = 0;

  scalar_t *input_data = nullptr;
  scalar_t *output_data = nullptr;


  THNN_ARGCHECK(!input->is_empty() && (input->dim() == 3 || input->dim() == 4), 2, input,
		"non-empty 3D or 4D (batch mode) tensor expected for input, but got: %s");

  if (input->dim() == 4)
  {
    istrideB = input->stride(0);
    sizeB = input->size(0);
    dimD++;
    dimH++;
    dimW++;
  }

  /* sizes */
  sizeD  = input->size(dimD);
  isizeH = input->size(dimH);
  isizeW = input->size(dimW);
  /* strides */
  istrideD = input->stride(dimD);
  istrideH = input->stride(dimH);
  istrideW = input->stride(dimW);

  /* resize output */
  if (input->dim() == 3)
  {
    THTensor_(resize3d)(output, sizeD, osizeH, osizeW);

    input_data = input->data<scalar_t>();
    output_data = output->data<scalar_t>();

    THNN_(SpatialAdaptiveAveragePooling_updateOutput_frame)(input_data, output_data,
                                                      sizeD,
                                                      isizeH, isizeW,
                                                      osizeH, osizeW,
                                                      istrideD,
                                                      istrideH, istrideW);
  }
  else
  {
    int64_t b;

    THTensor_(resize4d)(output, sizeB, sizeD, osizeH, osizeW);

    input_data = input->data<scalar_t>();
    output_data = output->data<scalar_t>();

#pragma omp parallel for private(b)
    for (b = 0; b < sizeB; b++)
    {
      THNN_(SpatialAdaptiveAveragePooling_updateOutput_frame)(input_data+b*istrideB, output_data+b*sizeD*osizeH*osizeW,
                                                      sizeD,
                                                      isizeH, isizeW,
                                                      osizeH, osizeW,
                                                      istrideD,
                                                      istrideH, istrideW);
    }
  }
}

static void THNN_(SpatialAdaptiveAveragePooling_updateGradInput_frame)(
          scalar_t *gradInput_p,
          scalar_t *gradOutput_p,
          int64_t sizeD,
          int64_t isizeH,
          int64_t isizeW,
          int64_t osizeH,
          int64_t osizeW)
{
  int64_t d;
#pragma omp parallel for private(d)
  for (d = 0; d < sizeD; d++)
  {
    scalar_t *gradInput_p_d = gradInput_p + d*isizeW*isizeH;
    scalar_t *gradOutput_p_d = gradOutput_p + d*osizeW*osizeH;

    /* calculate average */
    int64_t oh, ow;
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

        scalar_t grad_delta = gradOutput_p_d[oh*osizeW +ow] / kH / kW;

        int ih, iw;
        for(ih = istartH; ih < iendH; ih++)
        {
          for(iw = istartW; iw < iendW; iw++)
          {
            /* update gradient */
            gradInput_p_d[ih*isizeW + iw] += grad_delta;
          }
        }
      }
    }
  }
}

void THNN_(SpatialAdaptiveAveragePooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput)
{
  int dimD = 0;
  int dimH = 1;
  int dimW = 2;
  int64_t sizeB = 1;
  int sizeD;
  int isizeH;
  int isizeW;
  int osizeH;
  int osizeW;
  scalar_t *gradInput_data;
  scalar_t *gradOutput_data;

  /* get contiguous gradOutput */
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* resize */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  if (input->dim() == 4) {
    sizeB = input->size(0);
    dimD++;
    dimH++;
    dimW++;
  }

  /* sizes */
  sizeD  = input->size(dimD);
  isizeH = input->size(dimH);
  isizeW = input->size(dimW);
  osizeH = gradOutput->size(dimH);
  osizeW = gradOutput->size(dimW);

  /* get raw pointers */
  gradInput_data = gradInput->data<scalar_t>();
  gradOutput_data = gradOutput->data<scalar_t>();

  /* backprop */
  if (input->dim() == 3)
  {
    THNN_(SpatialAdaptiveAveragePooling_updateGradInput_frame)(gradInput_data, gradOutput_data,
                                                         sizeD,
                                                         isizeH, isizeW,
                                                         osizeH, osizeW);
  }
  else
  {
    int64_t b;
#pragma omp parallel for private(b)
    for (b = 0; b < sizeB; b++)
    {
      THNN_(SpatialAdaptiveAveragePooling_updateGradInput_frame)(gradInput_data+b*sizeD*isizeH*isizeW, gradOutput_data+b*sizeD*osizeH*osizeW,
                                                           sizeD,
                                                           isizeH, isizeW,
                                                           osizeH, osizeW);
    }
  }

  /* cleanup */
  c10::raw::intrusive_ptr::decref(gradOutput);
}

#endif

#undef START_IND
#undef END_IND
