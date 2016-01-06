#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialAdaptiveMaxPooling.c"
#else

static void nn_(SpatialAdaptiveMaxPooling_updateOutput_frame)(real *input_p,real *output_p,
                                                              real *indx_p, real *indy_p,
                                                              long nslices,
                                                              long iwidth, long iheight,
                                                              long owidth, long oheight,
                                                              long stridew,long strideh,
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
      int y_start = (int)floor((float)i / oheight * iheight);
      int y_end   = (int)ceil((float)(i + 1) / oheight * iheight);
      int kH = y_end-y_start;

      for(j = 0; j < owidth; j++)
      {
        
        int x_start = (int)floor((float)j / owidth * iwidth);
        int x_end   = (int)ceil((float)(j + 1) / owidth * iwidth);
        int kW = x_end-x_start;

        /* local pointers */
        real *ip = input_p   + k*strided + y_start*strideh + x_start*stridew;
        real *op = output_p  + k*owidth*oheight + i*owidth + j;
        real *indyp = indy_p + k*owidth*oheight + i*owidth + j;
        real *indxp = indx_p + k*owidth*oheight + i*owidth + j;

        /* compute local max: */
        long maxindex = -1;
        real maxval = -FLT_MAX;
        long tcntr = 0;
        int x,y;
        for(y = 0; y < kH; y++)
        {
          for(x = 0; x < kW; x++)
          {
            real val = *(ip + y*strideh + x*stridew);
            if (val > maxval)
            {
              maxval = val;
              maxindex = tcntr;
            }
            tcntr++;
          }
        }

        /* set output to local max */
        *op = maxval;

        /* store location of max (x,y) */
        *indyp = (int)(maxindex / kW)+1;
        *indxp = (maxindex % kW) +1;
      }
    }
  }
}

static int nn_(SpatialAdaptiveMaxPooling_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  long oheight = luaT_getfieldcheckint(L, 1, "H");
  long owidth = luaT_getfieldcheckint(L, 1, "W");
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
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
  real *indices_data;


  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4 , 2, "3D or 4D (batch mode) tensor expected");

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
    /* indices will contain i,j locations for each output point */
    THTensor_(resize4d)(indices, 2, nslices, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THTensor_(data)(indices);

    nn_(SpatialAdaptiveMaxPooling_updateOutput_frame)(input_data, output_data,
                                                      indices_data+nslices*owidth*oheight, indices_data,
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
    /* indices will contain i,j locations for each output point */
    THTensor_(resize5d)(indices, 2, nbatch, nslices, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THTensor_(data)(indices);

#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      nn_(SpatialAdaptiveMaxPooling_updateOutput_frame)(input_data+p*istride_b, output_data+p*nslices*owidth*oheight,
                                                        indices_data+(p+nbatch)*nslices*owidth*oheight, indices_data+p*nslices*owidth*oheight,
                                                        nslices,
                                                        iwidth, iheight,
                                                        owidth, oheight,
                                                        istride_w,istride_h,
                                                        istride_d);
    }
  }

  return 1;
}



static void nn_(SpatialAdaptiveMaxPooling_updateGradInput_frame)(real *gradInput_p, real *gradOutput_p,
                                                                 real *indx_p, real *indy_p,
                                                                 long nslices,
                                                                 long iwidth, long iheight,
                                                                 long owidth, long oheight)
{
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    real *gradInput_p_k = gradInput_p + k*iwidth*iheight;
    real *gradOutput_p_k = gradOutput_p + k*owidth*oheight;
    real *indx_p_k = indx_p + k*owidth*oheight;
    real *indy_p_k = indy_p + k*owidth*oheight;
    
    /* calculate max points */
    long i, j;
    for(i = 0; i < oheight; i++)
    {
      int y_start = (int)floor((float) i / oheight * iheight);
      for(j = 0; j < owidth; j++)
      {
        int x_start = (int)floor((float) j / owidth * iwidth);
        /* retrieve position of max */
        long maxi = indy_p_k[i*owidth + j] - 1 + y_start;
        long maxj = indx_p_k[i*owidth + j] - 1 + x_start;
        
        /* update gradient */
        gradInput_p_k[maxi*iwidth + maxj] += gradOutput_p_k[i*owidth + j];
      }
    }
  }
}

static int nn_(SpatialAdaptiveMaxPooling_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
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
  real *indices_data;

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
  indices_data = THTensor_(data)(indices);

  /* backprop */
  if (input->nDimension == 3)
  {
    nn_(SpatialAdaptiveMaxPooling_updateGradInput_frame)(gradInput_data, gradOutput_data,
                                                         indices_data+nslices*owidth*oheight, indices_data,
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
      nn_(SpatialAdaptiveMaxPooling_updateGradInput_frame)(gradInput_data+p*nslices*iwidth*iheight, gradOutput_data+p*nslices*owidth*oheight,
                                                           indices_data+(p+nbatch)*nslices*owidth*oheight, indices_data+p*nslices*owidth*oheight,
                                                           nslices,
                                                           iwidth, iheight,
                                                           owidth, oheight);
    }
  }

  /* cleanup */
  THTensor_(free)(gradOutput);

  return 1;
}

static const struct luaL_Reg nn_(SpatialAdaptiveMaxPooling__) [] = {
  {"SpatialAdaptiveMaxPooling_updateOutput", nn_(SpatialAdaptiveMaxPooling_updateOutput)},
  {"SpatialAdaptiveMaxPooling_updateGradInput", nn_(SpatialAdaptiveMaxPooling_updateGradInput)},
  {NULL, NULL}
};

static void nn_(SpatialAdaptiveMaxPooling_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SpatialAdaptiveMaxPooling__), "nn");
  lua_pop(L,1);
}

#endif

