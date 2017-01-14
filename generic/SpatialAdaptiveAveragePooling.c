#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialAdaptiveAveragePooling.c"
#else

#define START_IND(a,b,c) (int)floor((float)(a * c) / b)
#define END_IND(a,b,c) (int)ceil((float)((a + 1) * c) / b)
// #define START_IND(a,b,c) a * c / b
// #define END_IND(a,b,c)  (a + 1) * c / b + ((a + 1) * c % b > 0)?1:0

static void THNN_(SpatialAdaptiveAveragePooling_updateOutput_frame)(
          real *input_p,
          real *output_p,
          long nslices,
          long iwidth,
          long iheight,
          long owidth,
          long oheight,
          long stridew,
          long strideh,
          long strided)
{
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    /* loop over output */
    long i, j;
    for(i = 0; i < oheight; i++)
    {
      int y_start = START_IND(i, oheight, iheight);
      int y_end   = END_IND(i, oheight, iheight);
      int kH = y_end-y_start;

      for(j = 0; j < owidth; j++)
      {

        int x_start = START_IND(j, owidth, iwidth);
        int x_end   = END_IND(j, owidth, iwidth);
        int kW = x_end-x_start;

        /* local pointers */
        real *ip = input_p   + k*strided + y_start*strideh + x_start*stridew;
        real *op = output_p  + k*owidth*oheight + i*owidth + j;

        /* compute local average: */
        real sum = 0;
        int x,y;
        for(y = 0; y < kH; y++)
        {
          for(x = 0; x < kW; x++)
          {
            real val = *(ip + y*strideh + x*stridew);
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
          int owidth,
          int oheight)
{
  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  long nslices;
  long iheight;
  long iwidth;

  long istride_d;
  long istride_h;
  long istride_w;
  long istride_b;

  real *input_data;
  real *output_data;


  THNN_ARGCHECK(input->nDimension == 3 || input->nDimension == 4, 2, input,
		"3D or 4D (batch mode) tensor expected for input, but got: %s");

  if (input->nDimension == 4)
  {
    istride_b = input->stride[0];
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }

  /* sizes */
  nslices = input->size[dimh-1];
  iheight = input->size[dimh];
  iwidth = input->size[dimw];
  /* strides */
  istride_d = input->stride[dimh-1];
  istride_h = input->stride[dimh];
  istride_w = input->stride[dimw];

  /* resize output */
  if (input->nDimension == 3)
  {
    THTensor_(resize3d)(output, nslices, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);

    THNN_(SpatialAdaptiveAveragePooling_updateOutput_frame)(input_data, output_data,
                                                      nslices,
                                                      iwidth, iheight,
                                                      owidth, oheight,
                                                      istride_w,istride_h,
                                                      istride_d);
  }
  else
  {
    long p;

    THTensor_(resize4d)(output, nbatch, nslices, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);

#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      THNN_(SpatialAdaptiveAveragePooling_updateOutput_frame)(input_data+p*istride_b, output_data+p*nslices*owidth*oheight,
                                                        nslices,
                                                        iwidth, iheight,
                                                        owidth, oheight,
                                                        istride_w,istride_h,
                                                        istride_d);
    }
  }
}

static void THNN_(SpatialAdaptiveAveragePooling_updateGradInput_frame)(
          real *gradInput_p,
          real *gradOutput_p,
          long nslices,
          long iwidth,
          long iheight,
          long owidth,
          long oheight)
{
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    real *gradInput_p_k = gradInput_p + k*iwidth*iheight;
    real *gradOutput_p_k = gradOutput_p + k*owidth*oheight;

    /* calculate average */
    long i, j;
    for(i = 0; i < oheight; i++)
    {
      int y_start = START_IND(i, oheight, iheight);
      int y_end   = END_IND(i, oheight, iheight);
      int kH = y_end-y_start;

      for(j = 0; j < owidth; j++)
      {

        int x_start = START_IND(j, owidth, iwidth);
        int x_end   = END_IND(j, owidth, iwidth);
        int kW = x_end-x_start;

        int x,y;
        for(y = y_start; y < y_end; y++)
        {
          for(x = x_start; x < x_end; x++)
          {
            /* update gradient */
            gradInput_p_k[y*iwidth + x] += gradOutput_p_k[i*owidth + j] / kW / kH;
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
  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  int nslices;
  int iheight;
  int iwidth;
  int oheight;
  int owidth;
  real *gradInput_data;
  real *gradOutput_data;

  /* get contiguous gradOutput */
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* resize */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  if (input->nDimension == 4) {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }

  /* sizes */
  nslices = input->size[dimh-1];
  iheight = input->size[dimh];
  iwidth = input->size[dimw];
  oheight = gradOutput->size[dimh];
  owidth = gradOutput->size[dimw];

  /* get raw pointers */
  gradInput_data = THTensor_(data)(gradInput);
  gradOutput_data = THTensor_(data)(gradOutput);

  /* backprop */
  if (input->nDimension == 3)
  {
    THNN_(SpatialAdaptiveAveragePooling_updateGradInput_frame)(gradInput_data, gradOutput_data,
                                                         nslices,
                                                         iwidth, iheight,
                                                         owidth, oheight);
  }
  else
  {
    long p;
#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      THNN_(SpatialAdaptiveAveragePooling_updateGradInput_frame)(gradInput_data+p*nslices*iwidth*iheight, gradOutput_data+p*nslices*owidth*oheight,
                                                           nslices,
                                                           iwidth, iheight,
                                                           owidth, oheight);
    }
  }

  /* cleanup */
  THTensor_(free)(gradOutput);
}

#endif

#undef START_IND
#undef END_IND