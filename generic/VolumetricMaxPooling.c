#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricMaxPooling.c"
#else

static void nn_(VolumetricMaxPooling_updateOutput_frame)(
  real *input_p, real *output_p, real *indz_p,
  long nslices, long itime, long iwidth, long iheight,
  long otime, long owidth, long oheight,
  int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH) {
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    /* loop over output */
    long i, j, ti;
    for(ti = 0; ti < otime; ti++) {
      for(i = 0; i < oheight; i++) {
        for(j = 0; j < owidth; j++) {
          /* local pointers */
          
          long start_t = ti * dT - padT;
          long start_h = i * dH - padH;
          long start_w = j * dW - padW;
          
          long kernel_t = fminf(kT, kT + start_t);
          long kernel_h = fminf(kH, kH + start_h);
          long kernel_w = fminf(kW, kW + start_w);
          
          start_t = fmaxf(start_t, 0);
          start_h = fmaxf(start_h, 0);
          start_w = fmaxf(start_w, 0);
          
          real *ip = input_p + k * itime * iwidth * iheight
            + start_t * iwidth * iheight + start_h * iwidth + start_w;
          real *op = output_p + k * otime * owidth * oheight
            + ti * owidth * oheight + i * owidth + j;
          real *indzp = indz_p + k * otime * owidth * oheight
            + ti * owidth * oheight + i * owidth + j;

          /* compute local max: */
          real maxval = -THInf;
          int x,y,z;
          int mx, my, mz;

          for(z = 0; z < kernel_t; z++) {
            for(y = 0; y < kernel_h; y++) {
              for(x = 0; x < kernel_w; x++) {
                if ((start_t + z < itime) && (start_h + y < iheight) && (start_w + x < iwidth))
                {
                  real val = *(ip + z * iwidth * iheight + y * iwidth + x);
                  if (val > maxval) {
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

static int nn_(VolumetricMaxPooling_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int kT = luaT_getfieldcheckint(L, 1, "kT");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dT = luaT_getfieldcheckint(L, 1, "dT");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int padT = luaT_getfieldcheckint(L, 1, "padT");
  int padW = luaT_getfieldcheckint(L, 1, "padW");
  int padH = luaT_getfieldcheckint(L, 1, "padH");
  int ceil_mode = luaT_getfieldcheckboolean(L,1,"ceil_mode");
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  long nslices;
  long itime;
  long iheight;
  long iwidth;
  long otime;
  long oheight;
  long owidth;
  real *input_data;
  real *output_data;
  real *indices_data;

  luaL_argcheck(L, input->nDimension == 4 || input->nDimension == 5, 2,
                "4D or 5D (batch-mode) tensor expected");

  int dimN = 0;
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  if (input->nDimension == 5) {
    dimN++;
    dimt++;
    dimh++;
    dimw++;
  }

  luaL_argcheck(L, input->size[dimw] >= kW &&
                input->size[dimh] >= kH && input->size[dimt] >= kT, 2,
                "input image smaller than kernel size");

  luaL_argcheck(L, kT/2 >= padT && kW/2 >= padW && kH/2 >= padH, 2, "pad should be smaller than half of kernel size");

  /* sizes */
  nslices = input->size[dimN];
  itime   = input->size[dimt];
  iheight = input->size[dimh];
  iwidth  = input->size[dimw];
  if (ceil_mode) {
    otime   = (int)(ceil((float)(itime   - kT + 2 * padT) / dT) + 1);
    oheight = (int)(ceil((float)(iheight - kH + 2 * padH) / dH) + 1);
    owidth  = (int)(ceil((float)(iwidth  - kW + 2 * padW) / dW) + 1);
  } else {
    otime   = (int)(floor((float)(itime   - kT + 2 * padT) / dT) + 1);
    oheight = (int)(floor((float)(iheight - kH + 2 * padH) / dH) + 1);
    owidth  = (int)(floor((float)(iwidth  - kW + 2 * padW) / dW) + 1);
  }

  if (padT || padW || padH)
  {
    // ensure that the last pooling starts inside the image
    if ((otime - 1)*dT >= itime + padT)
      --otime;
    if ((oheight - 1)*dH >= iheight + padH)
      --oheight;
    if ((owidth  - 1)*dW >= iwidth  + padW)
      --owidth;
  }

  /* get contiguous input */
  input = THTensor_(newContiguous)(input);

  if (input->nDimension == 4) { /* non-batch mode */
    /* resize output */
    THTensor_(resize4d)(output, nslices, otime, oheight, owidth);
    /* indices will contain ti,i,j uchar locations packed into float/double */
    THTensor_(resize4d)(indices, nslices, otime, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THTensor_(data)(indices);

    nn_(VolumetricMaxPooling_updateOutput_frame)(input_data, output_data,
                                                 indices_data,
                                                 nslices,
                                                 itime, iwidth, iheight,
                                                 otime, owidth, oheight,
                                                 kT, kW, kH, dT, dW, dH, padT, padW, padH);
  } else { /* batch mode */
    long p;
    long nBatch = input->size[0];

    long istride = nslices * itime * iwidth * iheight;
    long ostride = nslices * otime * owidth * oheight;

    /* resize output */
    THTensor_(resize5d)(output, nBatch, nslices, otime, oheight, owidth);
    /* indices will contain ti,i,j locations for each output point */
    THTensor_(resize5d)(indices, nBatch, nslices, otime, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THTensor_(data)(indices);

#pragma omp parallel for private(p)
    for (p=0; p < nBatch; p++) {
      nn_(VolumetricMaxPooling_updateOutput_frame)(
        input_data   + p * istride,
        output_data  + p * ostride,
        indices_data + p * ostride,
        nslices,
        itime, iwidth, iheight,
        otime, owidth, oheight,
        kT, kW, kH, dT, dW, dH, padT, padW, padH);
    }
  }

  /* cleanup */
  THTensor_(free)(input);
  return 1;
}

static void nn_(VolumetricMaxPooling_updateGradInput_frame)(
  real *gradInput_p, real *gradOutput_p, real *indz_p,
  long nslices,
  long itime, long iwidth, long iheight,
  long otime, long owidth, long oheight,
  int dT, int dW, int dH,
  int padT, int padW, int padH) {
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++) {
    real *gradInput_p_k  = gradInput_p  + k * itime * iwidth * iheight;
    real *gradOutput_p_k = gradOutput_p + k * otime * owidth * oheight;
    real *indz_p_k = indz_p + k * otime * owidth * oheight;

    /* calculate max points */
    long ti, i, j;
    for(ti = 0; ti < otime; ti++) {
      for(i = 0; i < oheight; i++) {
        for(j = 0; j < owidth; j++) {
          /* retrieve position of max */
          real * indzp = &indz_p_k[ti * oheight * owidth + i * owidth + j];
          long maxti = ((unsigned char*)(indzp))[0] + ti * dT - padT;
          long maxi  = ((unsigned char*)(indzp))[1] + i * dH - padH;
          long maxj  = ((unsigned char*)(indzp))[2] + j * dW - padW;

          /* update gradient */
          gradInput_p_k[maxti * iheight * iwidth + maxi * iwidth + maxj] +=
            gradOutput_p_k[ti * oheight * owidth + i * owidth + j];
        }
      }
    }
  }
}

static int nn_(VolumetricMaxPooling_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  int dT = luaT_getfieldcheckint(L, 1, "dT");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int padT = luaT_getfieldcheckint(L, 1, "padT");
  int padW = luaT_getfieldcheckint(L, 1, "padW");
  int padH = luaT_getfieldcheckint(L, 1, "padH");
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  int nslices;
  int itime;
  int iheight;
  int iwidth;
  int otime;
  int oheight;
  int owidth;
  real *gradInput_data;
  real *gradOutput_data;
  real *indices_data;

  int dimN = 0;
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;


  /* get contiguous gradOutput */
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* resize */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  if (input->nDimension == 5) {
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
  indices_data = THTensor_(data)(indices);

  /* backprop */
  if (input->nDimension == 4) { /* non-batch mode*/
    nn_(VolumetricMaxPooling_updateGradInput_frame)(
      gradInput_data, gradOutput_data,
      indices_data,
      nslices,
      itime, iwidth, iheight,
      otime, owidth, oheight,
      dT, dW, dH, padT, padW, padH);
  }
  else { /* batch mode */
    long p;
    long nBatch = input->size[0];

    long istride = nslices * itime * iwidth * iheight;
    long ostride = nslices * otime * owidth * oheight;

#pragma omp parallel for private(p)
    for (p = 0; p < nBatch; p++) {
      nn_(VolumetricMaxPooling_updateGradInput_frame)(
        gradInput_data + p * istride,
        gradOutput_data + p * ostride,
        indices_data + p * ostride,
        nslices,
        itime, iwidth, iheight,
        otime, owidth, oheight,
        dT, dW, dH, padT, padW, padH);
    }
  }

  /* cleanup */
  THTensor_(free)(gradOutput);
  return 1;
}

static const struct luaL_Reg nn_(VolumetricMaxPooling__) [] = {
  {"VolumetricMaxPooling_updateOutput", nn_(VolumetricMaxPooling_updateOutput)},
  {"VolumetricMaxPooling_updateGradInput", nn_(VolumetricMaxPooling_updateGradInput)},
  {NULL, NULL}
};

static void nn_(VolumetricMaxPooling_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(VolumetricMaxPooling__), "nn");
  lua_pop(L,1);
}

#endif
