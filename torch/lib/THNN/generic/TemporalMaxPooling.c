#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TemporalMaxPooling.c"
#else

static inline void THNN_(TemporalMaxPooling_shapeCheck)(
                         THNNState *state,
                         THTensor *input,
                         THTensor *gradOutput,
                         THIndexTensor *indices,
                         int kW,
                         int dW) {
  long niframe;
  long framesize;
  long noframe;

  int dimS = 0; // sequence dimension
  int dimF = 1; // feature dimension
  int ndims = input->nDimension;

  if (input->nDimension == 3)
  {
    dimS = 1;
    dimF = 2;
  }

  niframe = input->size[dimS];
  framesize = input->size[dimF];
  noframe = (niframe - kW) / dW + 1;

  THArgCheck(kW > 0, 5,
             "kernel size should be greater than zero, but got kW: %d", kW);
  THArgCheck(dW > 0, 6,
             "stride should be greater than zero, but got dW: %d", dW);

  THNN_ARGCHECK(input->nDimension == 2 || input->nDimension == 3, 2, input,
                  "2D or 3D (batch mode) tensor expected for input, but got: %s");
  THArgCheck(input->size[dimS] >= kW, 2,
             "input sequence smaller than kernel size. Got: %d, Expected: %d",
             input->size[dimS], kW);

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, ndims, dimS, noframe);
    THNN_CHECK_DIM_SIZE(gradOutput, ndims, dimF, framesize)
  }
  if (indices != NULL) {
    THNN_CHECK_DIM_SIZE_INDICES(indices, ndims, dimS, noframe);
    THNN_CHECK_DIM_SIZE_INDICES(indices, ndims, dimF, framesize);
  }
}

void THNN_(TemporalMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          int kW,
          int dW)
{
  long niframe;
  long framesize;
  long noframe;

  real *input_data;
  real *output_data;
  THIndex_t *indices_data;

  long t, y;

  int dimS = 0; // sequence dimension
  int dimF = 1; // feature dimension

  THNN_(TemporalMaxPooling_shapeCheck)(state, input, NULL, NULL, kW, dW);

  if (input->nDimension == 3)
  {
    dimS = 1;
    dimF = 2;
  }

  /* sizes */
  niframe = input->size[dimS];
  framesize = input->size[dimF];
  noframe = (niframe - kW) / dW + 1;

  /* get contiguous input */
  input = THTensor_(newContiguous)(input);

  if (input->nDimension == 2)
  {
    /* resize output */
    THTensor_(resize2d)(output, noframe, framesize);

    /* indices will contain index locations for each output point */
    THIndexTensor_(resize2d)(indices, noframe, framesize);

    /* get raw pointers */
    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THIndexTensor_(data)(indices);

    for(t = 0; t < noframe; t++)
    {
      real *ip = input_data + t*framesize*dW;
      real *op = output_data + t*framesize;
      THIndex_t *xp = indices_data + t*framesize;
#pragma omp parallel for private(y)
      for(y = 0; y < framesize; y++)
      {
        /* compute local max: */
        long maxindex = -1;
        real maxval = -THInf;
        long x;
        for(x = 0; x < kW; x++)
        {
          real val = ip[x*framesize+y];
          if (val > maxval)
          {
            maxval = val;
            maxindex = x;
          }
        }

        /* set output to local max */
        op[y] = maxval;
        xp[y] = (real)maxindex;
      }
    }
  }
  else
  {
    /* number of batch frames */
    long nbframe = input->size[0];
    long i;

    /* resize output */
    THTensor_(resize3d)(output, nbframe, noframe, framesize);

    /* indices will contain index locations for each output point */
    THIndexTensor_(resize3d)(indices, nbframe, noframe, framesize);

    /* get raw pointers */
    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THIndexTensor_(data)(indices);

    for(i = 0; i < nbframe; i++)
    {
      real *inputSample_data = input_data + i*niframe*framesize;
      real *outputSample_data = output_data + i*noframe*framesize;
      THIndex_t *indicesSample_data = indices_data + i*noframe*framesize;

      for(t = 0; t < noframe; t++)
      {
        real *ip = inputSample_data + t*framesize*dW;
        real *op = outputSample_data + t*framesize;
        THIndex_t *xp = indicesSample_data + t*framesize;

#pragma omp parallel for private(y)
        for(y = 0; y < framesize; y++)
        {
          /* compute local max: */
          long maxindex = -1;
          real maxval = -THInf;
          long x;
          for(x = 0; x < kW; x++)
          {
            real val = ip[x*framesize+y];
            if (val > maxval)
            {
              maxval = val;
              maxindex = x;
            }
          }

          /* set output to local max */
          op[y] = maxval;
          xp[y] = (real)maxindex;
        }
      }
    }
  }

  /* cleanup */
  THTensor_(free)(input);

}

void THNN_(TemporalMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          int kW,
          int dW)
{
  long niframe;
  int noframe;
  long framesize;

  real *gradInput_data;
  real *gradOutput_data;
  THIndex_t *indices_data;

  long t, y;

  THNN_(TemporalMaxPooling_shapeCheck)(state, input, gradOutput, indices, kW, dW);
  /* get contiguous gradOutput */
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* resize and zero */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  int dimS = 0; // sequence dimension
  int dimF = 1; // feature dimension

  if (input->nDimension == 3)
  {
    dimS = 1;
    dimF = 2;
  }
  /* sizes */
  niframe = input->size[dimS];
  noframe = gradOutput->size[dimS];
  framesize = gradOutput->size[dimF];

  /* get raw pointers */
  gradInput_data = THTensor_(data)(gradInput);
  gradOutput_data = THTensor_(data)(gradOutput);
  indices_data = THIndexTensor_(data)(indices);

  if (input->nDimension == 2)
  {
    for(t = 0; t < noframe; t++)
    {
      real *gip = gradInput_data + t*framesize*dW;
      real *gop = gradOutput_data + t*framesize;
      THIndex_t *xp = indices_data + t*framesize;
#pragma omp parallel for private(y)
      for(y = 0; y < framesize; y++)
      {
        /* compute local max: */
        long maxindex = (long)xp[y];
        gip[maxindex*framesize+y] += gop[y];
      }
    }
  }
  else
  {
    /* number of batch frames */
    long nbframe = input->size[0];
    long i;

    for(i = 0; i < nbframe; i++)
    {
      real *gradInputSample_data = gradInput_data + i*niframe*framesize;
      real *gradOutputSample_data = gradOutput_data + i*noframe*framesize;
      THIndex_t *indicesSample_data = indices_data + i*noframe*framesize;

      for(t = 0; t < noframe; t++)
      {
        real *gip = gradInputSample_data + t*framesize*dW;
        real *gop = gradOutputSample_data + t*framesize;
        THIndex_t *xp = indicesSample_data + t*framesize;
#pragma omp parallel for private(y)
        for(y = 0; y < framesize; y++)
        {
          /* compute local max: */
          long maxindex = (long)xp[y];
          gip[maxindex*framesize+y] += gop[y];
        }
      }
    }
  }

  /* cleanup */
  THTensor_(free)(gradOutput);
}

#endif
