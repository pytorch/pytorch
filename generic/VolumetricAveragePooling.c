#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricAveragePooling.c"
#else

static inline void THNN_(VolumetricAveragePooling_shapeCheck)(
                         THNNState *state,
                         THTensor *input,
                         THTensor *gradOutput,
                         int kT,
                         int kW,
                         int kH,
                         int dT,
                         int dW,
                         int dH) {
  long nslices;
  long itime;
  long iheight;
  long iwidth;
  long otime;
  long oheight;
  long owidth;
  int ndim = input->nDimension;
  int dimN = 0;
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  if (input->nDimension == 5)
  {
    dimN++;
    dimt++;
    dimh++;
    dimw++;
  }

  THArgCheck(kT > 0 && kW > 0 && kH > 0, 5,
             "kernel size should be greater than zero, but got kT: %d kH: %d kW: %d",
             kT, kH, kW);
  THArgCheck(dT > 0 && dW > 0 && dH > 0, 8,
             "stride should be greater than zero, but got dT: %d dH: %d dW: %d",
             dT, dH, dW);
  THNN_ARGCHECK(input->nDimension == 4 || input->nDimension == 5, 2, input,
                "4D or 5D (batch mode) tensor expected for input, but got: %s");

  THArgCheck(input->size[dimw] >= kW && input->size[dimh] >= kH
             && input->size[dimt] >= kT, 2,
             "input image (T: %d H: %d W: %d) smaller than "
             "kernel size (kT: %d kH: %d kW: %d)",
             input->size[dimt], input->size[dimh], input->size[dimw],
             kT, kH, kW);

  /* sizes */
  nslices = input->size[dimN];
  itime   = input->size[dimt];
  iheight = input->size[dimh];
  iwidth  = input->size[dimw];
  otime   = (itime   - kT) / dT + 1;
  oheight = (iheight - kH) / dH + 1;
  owidth  = (iwidth  - kW) / dW + 1;

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimN, nslices);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimt, otime);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimh, oheight);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimw, owidth);
  }
}

static void THNN_(VolumetricAveragePooling_updateOutput_frame)(
          real *input_p,
          real *output_p,
          long nslices,
          long itime,
          long iwidth,
          long iheight,
          long otime,
          long owidth,
          long oheight,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH)
{
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    /* loop over output */
    long i, j, ti;
    for (ti = 0; ti < otime; ti++)
    {
      for (i = 0; i < oheight; i++)
      {
        for (j = 0; j < owidth; j++)
        {
          /* local pointers */
          real *ip = input_p + k * itime * iwidth * iheight
            + ti * iwidth * iheight * dT +  i * iwidth * dH + j * dW;
          real *op = output_p + k * otime * owidth * oheight
            + ti * owidth * oheight + i * owidth + j;

          /* compute local sum: */
          real sum = 0.0;
          int x, y, z;

          for (z=0; z < kT; z++)
          {
            for (y = 0; y < kH; y++)
            {
              for (x = 0; x < kW; x++)
              {
                sum +=  *(ip + z * iwidth * iheight + y * iwidth + x);
              }
            }
          }

          /* set output to local max */
          *op = sum / (kT * kW * kH);
        }
      }
    }
  }
}

void THNN_(VolumetricAveragePooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH)
{
  long nslices;
  long itime;
  long iheight;
  long iwidth;
  long otime;
  long oheight;
  long owidth;
  real *input_data;
  real *output_data;

  THNN_(VolumetricAveragePooling_shapeCheck)(
        state, input, NULL, kT, kW, kH,
        dT, dW, dH);

  int dimN = 0;
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  if (input->nDimension == 5)
  {
    dimN++;
    dimt++;
    dimh++;
    dimw++;
  }

  /* sizes */
  nslices = input->size[dimN];
  itime   = input->size[dimt];
  iheight = input->size[dimh];
  iwidth  = input->size[dimw];
  otime   = (itime   - kT) / dT + 1;
  oheight = (iheight - kH) / dH + 1;
  owidth  = (iwidth  - kW) / dW + 1;

  /* get contiguous input */
  input = THTensor_(newContiguous)(input);

  if (input->nDimension == 4) /* non-batch mode */
  {
    /* resize output */
    THTensor_(resize4d)(output, nslices, otime, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);

    THNN_(VolumetricAveragePooling_updateOutput_frame)(
      input_data, output_data, nslices,
      itime, iwidth, iheight,
      otime, owidth, oheight,
      kT, kW, kH,
      dT, dW, dH
    );
  }
  else  /* batch mode */
  {
    long p;
    long nBatch = input->size[0];

    long istride = nslices * itime * iwidth * iheight;
    long ostride = nslices * otime * owidth * oheight;

    /* resize output */
    THTensor_(resize5d)(output, nBatch, nslices, otime, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);

#pragma omp parallel for private(p)
    for (p=0; p < nBatch; p++)
    {
      THNN_(VolumetricAveragePooling_updateOutput_frame)(
        input_data + p * istride, output_data + p * ostride, nslices,
        itime, iwidth, iheight,
        otime, owidth, oheight,
        kT, kW, kH,
        dT, dW, dH
      );
    }
  }

  /* cleanup */
  THTensor_(free)(input);
}

static void THNN_(VolumetricAveragePooling_updateGradInput_frame)(
          real *gradInput_p,
          real *gradOutput_p,
          long nslices,
          long itime,
          long iwidth,
          long iheight,
          long otime,
          long owidth,
          long oheight,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH)
{
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    /* loop over output */
    long i, j, ti;
    for (ti = 0; ti < otime; ti++)
    {
      for (i = 0; i < oheight; i++)
      {
        for (j = 0; j < owidth; j++)
        {
          /* local pointers */
          real *ip = gradInput_p + k * itime * iwidth * iheight
            + ti * iwidth * iheight * dT +  i * iwidth * dH + j * dW;
          real *op = gradOutput_p + k * otime * owidth * oheight
            + ti * owidth * oheight + i * owidth + j;

          /* scatter gradients out to footprint: */
          real val  = *op / (kT * kW * kH);
          int x,y,z;
          for (z=0; z < kT; z++)
          {
            for (y = 0; y < kH; y++)
            {
              for (x = 0; x < kW; x++)
              {
                *(ip + z * iwidth * iheight + y * iwidth + x) += val;
              }
            }
          }
        }
      }
    }
  }
}

void THNN_(VolumetricAveragePooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH)
{
  int nslices;
  int itime;
  int iheight;
  int iwidth;
  int otime;
  int oheight;
  int owidth;
  real *gradInput_data;
  real *gradOutput_data;

  int dimN = 0;
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  THNN_(VolumetricAveragePooling_shapeCheck)(
        state, input, gradOutput, kT, kW, kH,
        dT, dW, dH);

  /* get contiguous gradOutput */
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* resize */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  if (input->nDimension == 5)
  {
    dimN++;
    dimt++;
    dimh++;
    dimw++;
  }

  /* sizes */
  nslices = input->size[dimN];
  itime = input->size[dimt];
  iheight = input->size[dimh];
  iwidth = input->size[dimw];
  otime = gradOutput->size[dimt];
  oheight = gradOutput->size[dimh];
  owidth = gradOutput->size[dimw];

  /* get raw pointers */
  gradInput_data = THTensor_(data)(gradInput);
  gradOutput_data = THTensor_(data)(gradOutput);

  /* backprop */
  if (input->nDimension == 4) /* non-batch mode*/
  {
    THNN_(VolumetricAveragePooling_updateGradInput_frame)(
      gradInput_data, gradOutput_data, nslices,
      itime, iwidth, iheight,
      otime, owidth, oheight,
      kT, kW, kH,
      dT, dW, dH
    );
  }
  else /* batch mode */
  {
    long p;
    long nBatch = input->size[0];

    long istride = nslices * itime * iwidth * iheight;
    long ostride = nslices * otime * owidth * oheight;

#pragma omp parallel for private(p)
    for (p = 0; p < nBatch; p++)
    {
      THNN_(VolumetricAveragePooling_updateGradInput_frame)(
        gradInput_data  + p * istride, gradOutput_data + p * ostride, nslices,
        itime, iwidth, iheight,
        otime, owidth, oheight,
        kT, kW, kH,
        dT, dW, dH
      );
    }
  }

  /* cleanup */
  THTensor_(free)(gradOutput);
}

#endif
