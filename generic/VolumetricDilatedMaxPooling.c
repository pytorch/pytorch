#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricDilatedMaxPooling.c"
#else

static void THNN_(VolumetricDilatedMaxPooling_updateOutput_frame)(
          real *input_p,
          real *output_p,
          THIndex_t *indz_p,
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
          int dH,
          int pT,
          int pW,
          int pH,
          int dilationT,
          int dilationW,
          int dilationH)
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

          long start_t = ti * dT - pT;
          long start_h = i * dH - pH;
          long start_w = j * dW - pW;

          long kernel_t = fminf(kT, kT + start_t);
          long kernel_h = fminf(kH, kH + start_h);
          long kernel_w = fminf(kW, kW + start_w);

          while(start_t < 0)
            start_t += dilationT;
          while(start_h < 0)
            start_h += dilationH;
          while(start_w < 0)
            start_w += dilationW;

          real *ip = input_p + k * itime * iwidth * iheight
            + start_t * iwidth * iheight + start_h * iwidth + start_w;
          real *op = output_p + k * otime * owidth * oheight
            + ti * owidth * oheight + i * owidth + j;
          THIndex_t *indzp = indz_p + k * otime * owidth * oheight
            + ti * owidth * oheight + i * owidth + j;

          /* compute local max: */
          real maxval = -THInf;
          int x,y,z;
          int mx, my, mz;

          for (z = 0; z < kernel_t; z++)
          {
            for (y = 0; y < kernel_h; y++)
            {
              for (x = 0; x < kernel_w; x++)
              {
                if ((start_t + z * dilationT < itime) && (start_h + y * dilationH < iheight) && (start_w + x * dilationW < iwidth))
                {
                  real val = *(ip + z * dilationT * iwidth * iheight + y * dilationH * iwidth + x * dilationW);
                  if (val > maxval)
                  {
                    maxval = val;
                    // Store indices w.r.t the kernel dimension
                    mz = z + (kT - kernel_t);
                    my = y + (kH - kernel_h);
                    mx = x + (kW - kernel_w);
                  }
                }
              }
            }
          }

          // set max values
          ((unsigned char*)(indzp))[0] = mz;
          ((unsigned char*)(indzp))[1] = my;
          ((unsigned char*)(indzp))[2] = mx;
          ((unsigned char*)(indzp))[3] = 0;

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
  long nslices;
  long itime;
  long iheight;
  long iwidth;
  long otime;
  long oheight;
  long owidth;
  real *input_data;
  real *output_data;
  THIndex_t *indices_data;

  THNN_ARGCHECK(input->nDimension == 4 || input->nDimension == 5, 2, input,
		"4D or 5D (batch mode) tensor expected for input, but got: %s");

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

  THArgCheck(input->size[dimw] >= kW && input->size[dimh] >= kH
	     && input->size[dimt] >= kT, 2,
	     "input image (T: %d H: %d W: %d) smaller than "
	     "kernel size (kT: %d kH: %d kW: %d)",
	     input->size[dimt], input->size[dimh], input->size[dimw],
	     kT, kH, kW);

  THArgCheck(kT/2 >= pT && kW/2 >= pW && kH/2 >= pH, 2,
    "pad should be smaller than half of kernel size"
  );

  THArgCheck(dilationT > 0 && dilationW > 0 && dilationH > 0, 14,
      "dilation should be greater than 0");

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

  if (otime < 1 || owidth < 1 || oheight < 1)
    THError("Given input size: (%dx%dx%dx%d). Calculated output size: (%dx%dx%dx%d). Output size is too small",
            nslices,itime,iheight,iwidth,nslices,otime,oheight,owidth);

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
    long p;
    long nBatch = input->size[0];

    long istride = nslices * itime * iwidth * iheight;
    long ostride = nslices * otime * owidth * oheight;

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
          long nslices,
          long itime,
          long iwidth,
          long iheight,
          long otime,
          long owidth,
          long oheight,
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
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    real *gradInput_p_k  = gradInput_p  + k * itime * iwidth * iheight;
    real *gradOutput_p_k = gradOutput_p + k * otime * owidth * oheight;
    THIndex_t *indz_p_k = indz_p + k * otime * owidth * oheight;

    /* calculate max points */
    long ti, i, j;
    for (ti = 0; ti < otime; ti++)
    {
      for (i = 0; i < oheight; i++)
      {
        for (j = 0; j < owidth; j++)
        {
          /* retrieve position of max */
          THIndex_t * indzp = &indz_p_k[ti * oheight * owidth + i * owidth + j];
          long maxti = ((unsigned char*)(indzp))[0] * dilationT + ti * dT - pT;
          long maxi  = ((unsigned char*)(indzp))[1] * dilationH + i * dH - pH;
          long maxj  = ((unsigned char*)(indzp))[2] * dilationW + j * dW - pW;

          /* update gradient */
          gradInput_p_k[maxti * iheight * iwidth + maxi * iwidth + maxj] +=
            gradOutput_p_k[ti * oheight * owidth + i * owidth + j];
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
    long p;
    long nBatch = input->size[0];

    long istride = nslices * itime * iwidth * iheight;
    long ostride = nslices * otime * owidth * oheight;

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
