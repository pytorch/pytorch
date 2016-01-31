#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialMaxUnpooling.c"
#else

static void nn_(SpatialMaxUnpooling_updateOutput_frame)(real *input_p, real *output_p,
                                                      real *ind_p,
                                                      long nslices,
                                                      long iwidth, long iheight,
                                                      long owidth, long oheight)
{
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {    
    real *output_p_k = output_p + k*owidth*oheight;
    real *input_p_k = input_p + k*iwidth*iheight;
    real *ind_p_k = ind_p + k*iwidth*iheight;

    long i, j, maxp;
    for(i = 0; i < iheight; i++)
    {
      for(j = 0; j < iwidth; j++)
      {
        maxp = ind_p_k[i*iwidth + j] - 1;  /* retrieve position of max */
        if(maxp<0 || maxp>=owidth*oheight){
            THError("invalid max index %d, owidth= %d, oheight= %d",maxp,owidth,oheight);
        }
        output_p_k[maxp] = input_p_k[i*iwidth + j]; /* update output */
      }
    }
  }
}

static int nn_(SpatialMaxUnpooling_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  int owidth = luaT_getfieldcheckint(L, 1, "owidth");
  int oheight = luaT_getfieldcheckint(L, 1, "oheight");
  int dimw = 2;
  int dimh = 1;
  int nbatch = 1;
  int nslices;
  int iheight;
  int iwidth;
  real *input_data;
  real *output_data;
  real *indices_data;


  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4 , 2, "3D or 4D (batch mode) tensor expected");
  if (!THTensor_(isSameSizeAs)(input, indices)){
    THError("Invalid input size w.r.t current indices size");
  }  

  if (input->nDimension == 4) 
  {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }

  /* sizes */
  nslices = input->size[dimh-1];
  iheight = input->size[dimh];
  iwidth = input->size[dimw];

  /* get contiguous input and indices */
  input = THTensor_(newContiguous)(input);
  indices = THTensor_(newContiguous)(indices);

  /* resize output */
  if (input->nDimension == 3)
  {
    THTensor_(resize3d)(output, nslices, oheight, owidth);
    THTensor_(zero)(output);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THTensor_(data)(indices);

    nn_(SpatialMaxUnpooling_updateOutput_frame)(input_data, output_data,
                                              indices_data,
                                              nslices,
                                              iwidth, iheight,
                                              owidth, oheight);
  }
  else
  {
    long p;

    THTensor_(resize4d)(output, nbatch, nslices, oheight, owidth);
    THTensor_(zero)(output);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);
    indices_data = THTensor_(data)(indices);

#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      nn_(SpatialMaxUnpooling_updateOutput_frame)(input_data+p*nslices*iwidth*iheight, output_data+p*nslices*owidth*oheight,
                                                indices_data+p*nslices*iwidth*iheight,
                                                nslices,
                                                iwidth, iheight,
                                                owidth, oheight);
    }
  }

  /* cleanup */
  THTensor_(free)(input);
  THTensor_(free)(indices);

  return 1;
}

static void nn_(SpatialMaxUnpooling_updateGradInput_frame)(real *gradInput_p, real *gradOutput_p,
                                                         real *ind_p,
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
    real *ind_p_k = ind_p + k*iwidth*iheight;

    long i, j, maxp;
    for(i = 0; i < iheight; i++)
    {
      for(j = 0; j < iwidth; j++)
      {        
        maxp = ind_p_k[i*iwidth + j] - 1; /* retrieve position of max */         
        if(maxp<0 || maxp>=owidth*oheight){
            THError("invalid max index %d, owidth= %d, oheight= %d",maxp,owidth,oheight);
        }  
        gradInput_p_k[i*iwidth + j] = gradOutput_p_k[maxp]; /* update gradient */
      }
    }
  }
}

static int nn_(SpatialMaxUnpooling_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  int owidth = luaT_getfieldcheckint(L, 1, "owidth");
  int oheight = luaT_getfieldcheckint(L, 1, "oheight");
  int dimw = 2;
  int dimh = 1;
  int nbatch = 1;
  int nslices;
  int iheight;
  int iwidth;
  real *gradInput_data;
  real *gradOutput_data;
  real *indices_data;

  if (!THTensor_(isSameSizeAs)(input, indices)){
    THError("Invalid input size w.r.t current indices size");
  } 

  /* get contiguous gradOutput and indices */
  gradOutput = THTensor_(newContiguous)(gradOutput);
  indices = THTensor_(newContiguous)(indices);

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

  if(owidth!=gradOutput->size[dimw] || oheight!=gradOutput->size[dimh]){
    THError("Inconsistent gradOutput size. oheight= %d, owidth= %d, gradOutput: %dx%d", oheight, owidth,gradOutput->size[dimh],gradOutput->size[dimw]);
  }

  /* get raw pointers */
  gradInput_data = THTensor_(data)(gradInput);
  gradOutput_data = THTensor_(data)(gradOutput);
  indices_data = THTensor_(data)(indices);

  /* backprop */
  if (input->nDimension == 3)
  {
    nn_(SpatialMaxUnpooling_updateGradInput_frame)(gradInput_data, gradOutput_data,
                                                 indices_data,
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
      nn_(SpatialMaxUnpooling_updateGradInput_frame)(gradInput_data+p*nslices*iwidth*iheight, gradOutput_data+p*nslices*owidth*oheight,
                                                   indices_data+p*nslices*iwidth*iheight,
                                                   nslices,
                                                   iwidth, iheight,
                                                   owidth, oheight);
    }
  }

  /* cleanup */
  THTensor_(free)(gradOutput);
  THTensor_(free)(indices);

  return 1;
}

static const struct luaL_Reg nn_(SpatialMaxUnpooling__) [] = {
  {"SpatialMaxUnpooling_updateOutput", nn_(SpatialMaxUnpooling_updateOutput)},
  {"SpatialMaxUnpooling_updateGradInput", nn_(SpatialMaxUnpooling_updateGradInput)},
  {NULL, NULL}
};

static void nn_(SpatialMaxUnpooling_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SpatialMaxUnpooling__), "nn");
  lua_pop(L,1);
}

#endif
