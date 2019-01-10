#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricDilatedMaxPooling.c"
#else

static inline void THNN_(VolumetricDilatedMaxPooling_shapeCheck)(
                         THNNState *state,
                         THTensor *input,
                         THTensor *gradOutput,
                         THIndexTensor *indices,
                         int kT, int kW, int kH,
                         int dT, int dW, int dH,
                         int pT, int pW, int pH,
                         int dilationT, int dilationW, int dilationH,
                         bool ceilMode) {
  int ndim = input->nDimension;
  int dimN = 0;
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;
  int64_t nslices;
  int64_t itime;
  int64_t iheight;
  int64_t iwidth;
  int64_t otime;
  int64_t oheight;
  int64_t owidth;

  THArgCheck(kT > 0 && kW > 0 && kH > 0, 5,
             "kernel size should be greater than zero, but got kT: %d kH: %d kW: %d",
             kT, kH, kW);
  THArgCheck(dT > 0 && dW > 0 && dH > 0, 8,
             "stride should be greater than zero, but got dT: %d dH: %d dW: %d",
             dT, dH, dW);
  THArgCheck(dilationT > 0 && dilationW > 0 && dilationH > 0, 14,
             "dilation should be greater than 0, but got dilationT: %d dilationH: %d dilationW: %d",
             dilationT, dilationH, dilationW);

  THNN_ARGCHECK(input->nDimension == 4 || input->nDimension == 5, 2, input,
                "4D or 5D (batch mode) tensor expected for input, but got: %s");

  if (input->nDimension == 5)
  {
    dimN++;
    dimt++;
    dimh++;
    dimw++;
  }

  THArgCheck(kT/2 >= pT && kW/2 >= pW && kH/2 >= pH, 2,
             "pad should be smaller than half of kernel size, but got "
             "kT: %d kW: %d, kH: %d, padT: %d, padW: %d, padH: %d",
             kT, kW, kH, pT, pW, pH);

  nslices = input->size[dimN];
  itime   = input->size[dimt];
  iheight = input->size[dimh];
  iwidth  = input->size[dimw];
  if (ceilMode)
  {
    otime = (int)(ceil((float)(itime - (dilationT * (kT - 1) + 1) + 2*pT) / dT)) + 1;
    oheight = (int)(ceil((float)(iheight - (dilationH * (kH - 1) + 1) + 2*pH) / dH)) + 1;
    owidth  = (int)(ceil((float)(iwidth  - (dilationW * (kW - 1) + 1) + 2*pW) / dW)) + 1;
  }
  else
  {
    otime = (int)(floor((float)(itime - (dilationT * (kT - 1) + 1) + 2*pT) / dT)) + 1;
    oheight = (int)(floor((float)(iheight - (dilationH * (kH - 1) + 1) + 2*pH) / dH)) + 1;
    owidth  = (int)(floor((float)(iwidth  - (dilationW * (kW - 1) + 1) + 2*pW) / dW)) + 1;
  }

  if (pT || pW || pH)
  {
    // ensure that the last pooling starts inside the image
    if ((otime - 1)*dT >= itime + pT)
      --otime;
    if ((oheight - 1)*dH >= iheight + pH)
      --oheight;
    if ((owidth  - 1)*dW >= iwidth  + pW)
      --owidth;
  }

  if (otime < 1 || owidth < 1 || oheight < 1)
    THError("Given input size: (%dx%dx%dx%d). Calculated output size: (%dx%dx%dx%d). Output size is too small",
            nslices,itime,iheight,iwidth,nslices,otime,oheight,owidth);

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimN, nslices);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimt, otime);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimh, oheight);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimw, owidth);
  }
  if (indices != NULL) {
    THNN_CHECK_DIM_SIZE_INDICES(indices, ndim, dimN, nslices);
    THNN_CHECK_DIM_SIZE_INDICES(indices, ndim, dimt, otime);
    THNN_CHECK_DIM_SIZE_INDICES(indices, ndim, dimh, oheight);
    THNN_CHECK_DIM_SIZE_INDICES(indices, ndim, dimw, owidth);
  }
}

static void THNN_(VolumetricDilatedMaxPooling_updateOutput_frame)(
          real *input_p,
          real *output_p,
          THIndex_t *indz_p,
          int64_t nslices,
          int64_t itime,
          int64_t iwidth,
          int64_t iheight,
          int64_t otime,
          int64_t owidth,
          int64_t oheight,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH,
          int dilationT,
          int dilationW,
          int dilationH)
{
  int64_t k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    /* loop over output */
    int64_t i, j, ti;
    real *ip = input_p + k * itime * iwidth * iheight;
    for (ti = 0; ti < otime; ti++)
    {
      for (i = 0; i < oheight; i++)
      {
        for (j = 0; j < owidth; j++)
        {
          /* local pointers */

          int64_t start_t = ti * dT - pT;
          int64_t start_h = i * dH - pH;
          int64_t start_w = j * dW - pW;

          int64_t end_t = fminf(start_t + (kT - 1) * dilationT + 1, itime);
          int64_t end_h = fminf(start_h + (kH - 1) * dilationH + 1, iheight);
          int64_t end_w = fminf(start_w + (kW - 1) * dilationW + 1, iwidth);

          while(start_t < 0)
            start_t += dilationT;
          while(start_h < 0)
            start_h += dilationH;
          while(start_w < 0)
            start_w += dilationW;

          real *op = output_p + k * otime * owidth * oheight
            + ti * owidth * oheight + i * owidth + j;
          THIndex_t *indzp = indz_p + k * otime * owidth * oheight
            + ti * owidth * oheight + i * owidth + j;

          /* compute local max: */
          int64_t maxindex = -1;
          real maxval = -THInf;
          int64_t x,y,z;
          int64_t index = 0;

          for (z = start_t; z < end_t; z += dilationT)
          {
            for (y = start_h; y < end_h; y += dilationH)
            {
              for (x = start_w; x < end_w; x += dilationW)
              {
                index = z * iwidth * iheight + y * iwidth + x;
                real val = ip[index];
                if (val > maxval)
                {
                  maxval = val;
                  maxindex = index;
                }
              }
            }
          }

          // store location of max
          *indzp = maxindex + TH_INDEX_BASE;

          /* set output to local max */
          *op = maxval;
        }
      }
    }
  }
}

void THNN_(VolumetricDilatedMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH,
          int dilationT,
          int dilationW,
          int dilationH,
          bool ceilMode)
{
  int64_t nslices;
  int64_t itime;
  int64_t iheight;
  int64_t iwidth;
  int64_t otime;
  int64_t oheight;
  int64_t owidth;
  real *input_data;
  real *output_data;
  THIndex_t *indices_data;


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

  THNN_(VolumetricDilatedMaxPooling_shapeCheck)(
        state, input, NULL, NULL,
        kT,  kW,  kH, dT,  dW,  dH,
        pT,  pW,  pH, dilationT,  dilationW,  dilationH,
        ceilMode);

  /* sizes */
  nslices = input->size[dimN];
  itime   = input->size[dimt];
  iheight = input->size[dimh];
  iwidth  = input->size[dimw];
  if (ceilMode)
  {
    otime = (int)(ceil((float)(itime - (dilationT * (kT - 1) + 1) + 2*pT) / dT)) + 1;
    oheight = (int)(ceil((float)(iheight - (dilationH * (kH - 1) + 1) + 2*pH) / dH)) + 1;
    owidth  = (int)(ceil((float)(iwidth  - (dilationW * (kW - 1) + 1) + 2*pW) / dW)) + 1;
  }
  else
  {
    otime = (int)(floor((float)(itime - (dilationT * (kT - 1) + 1) + 2*pT) / dT)) + 1;
    oheight = (int)(floor((float)(iheight - (dilationH * (kH - 1) + 1) + 2*pH) / dH)) + 1;
    owidth  = (int)(floor((float)(iwidth  - (dilationW * (kW - 1) + 1) + 2*pW) / dW)) + 1;
  }

  if (pT || pW || pH)
  {
    // ensure that the last pooling starts inside the image
    if ((otime - 1)*dT >= itime + pT)
      --otime;
    if ((oheight - 1)*dH >= iheight + pH)
      --oheight;
    if ((owidth  - 1)*dW >= iwidth  + pW)
      --owidth;
  }

  /* get contiguous input */
  input = THTensor_(newContiguous)(input);

  if (input->nDimension == 4) /* non-batch mode */
  {
    /* resize output */
    THTensor_(resize4d)(output, nslices, otime, oheight, owidth);
    /* indices will contain ti,i,j uchar locations packed into float/double */
    THIndexTensor_(resize4d)(indices, nslices, otime, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THIndexTensor_(data)(indices);

    THNN_(VolumetricDilatedMaxPooling_updateOutput_frame)(
      input_data, output_data,
      indices_data,
      nslices,
      itime, iwidth, iheight,
      otime, owidth, oheight,
      kT, kW, kH,
      dT, dW, dH,
      pT, pW, pH,
      dilationT, dilationW, dilationH
    );
  }
  else /* batch mode */
  {
    int64_t p;
    int64_t nBatch = input->size[0];

    int64_t istride = nslices * itime * iwidth * iheight;
    int64_t ostride = nslices * otime * owidth * oheight;

    /* resize output */
    THTensor_(resize5d)(output, nBatch, nslices, otime, oheight, owidth);
    /* indices will contain ti,i,j locations for each output point */
    THIndexTensor_(resize5d)(indices, nBatch, nslices, otime, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THIndexTensor_(data)(indices);

#pragma omp parallel for private(p)
    for (p=0; p < nBatch; p++)
    {
      THNN_(VolumetricDilatedMaxPooling_updateOutput_frame)(
        input_data   + p * istride,
        output_data  + p * ostride,
        indices_data + p * ostride,
        nslices,
        itime, iwidth, iheight,
        otime, owidth, oheight,
        kT, kW, kH,
        dT, dW, dH,
        pT, pW, pH,
        dilationT, dilationW, dilationH
      );
    }
  }

  /* cleanup */
  THTensor_(free)(input);
}

static void THNN_(VolumetricDilatedMaxPooling_updateGradInput_frame)(
          real *gradInput_p,
          real *gradOutput_p,
          THIndex_t *indz_p,
          int64_t nslices,
          int64_t itime,
          int64_t iwidth,
          int64_t iheight,
          int64_t otime,
          int64_t owidth,
          int64_t oheight,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH,
          int dilationT,
          int dilationW,
          int dilationH)
{
  int64_t k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    real *gradInput_p_k  = gradInput_p  + k * itime * iwidth * iheight;
    real *gradOutput_p_k = gradOutput_p + k * otime * owidth * oheight;
    THIndex_t *indz_p_k = indz_p + k * otime * owidth * oheight;

    /* calculate max points */
    int64_t ti, i, j;
    for (ti = 0; ti < otime; ti++)
    {
      for (i = 0; i < oheight; i++)
      {
        for (j = 0; j < owidth; j++)
        {
          /* retrieve position of max */
          int64_t index = ti * oheight * owidth + i * owidth + j;
          int64_t maxp = indz_p_k[index] - TH_INDEX_BASE;

          if (maxp != -1) {
            /* update gradient */
            gradInput_p_k[maxp] += gradOutput_p_k[index];
          }
        }
      }
    }
  }
}

void THNN_(VolumetricDilatedMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH,
          int dilationT,
          int dilationW,
          int dilationH,
          bool ceilMode)
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
  THIndex_t *indices_data;

  int dimN = 0;
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  THNN_(VolumetricDilatedMaxPooling_shapeCheck)(
        state, input, gradOutput, indices,
        kT,  kW,  kH, dT,  dW,  dH,
        pT,  pW,  pH, dilationT,  dilationW,  dilationH,
        ceilMode);

  // TODO: gradOutput shape check
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
  indices_data = THIndexTensor_(data)(indices);

  /* backprop */
  if (input->nDimension == 4) /* non-batch mode*/
  {
    THNN_(VolumetricDilatedMaxPooling_updateGradInput_frame)(
      gradInput_data, gradOutput_data,
      indices_data,
      nslices,
      itime, iwidth, iheight,
      otime, owidth, oheight,
      dT, dW, dH,
      pT, pW, pH,
      dilationT, dilationW, dilationH
    );
  }
  else /* batch mode */
  {
    int64_t p;
    int64_t nBatch = input->size[0];

    int64_t istride = nslices * itime * iwidth * iheight;
    int64_t ostride = nslices * otime * owidth * oheight;

#pragma omp parallel for private(p)
    for (p = 0; p < nBatch; p++)
    {
      THNN_(VolumetricDilatedMaxPooling_updateGradInput_frame)(
        gradInput_data + p * istride,
        gradOutput_data + p * ostride,
        indices_data + p * ostride,
        nslices,
        itime, iwidth, iheight,
        otime, owidth, oheight,
        dT, dW, dH,
        pT, pW, pH,
        dilationT, dilationW, dilationH
      );
    }
  }

  /* cleanup */
  THTensor_(free)(gradOutput);
}

#endif
