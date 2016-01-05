#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LookupTable.c"
#else

static void nn_(LookupTable_resetCount)(int *count_data, THLongTensor *input)
{
  int i;
  long *input_data = THLongTensor_data(input);
  long numel = THLongTensor_nElement(input);

  for (i = 0; i<numel; i++)
  {
    long k = input_data[i] - 1;
    count_data[k] = 0;
  }
  for (i = 0; i<numel; i++)
  {
    long k = input_data[i] - 1;
    count_data[k]++;
  }
}

static int nn_(LookupTable_accGradParameters)(lua_State *L)
{
  long i;
  THLongTensor *input = luaT_checkudata(L, 2, "torch.LongTensor");
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  real lr = luaL_optnumber(L, 4, 1);
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  int *count_data = NULL;

  if (luaT_getfieldcheckboolean(L, 1, "shouldScaleGradByFreq"))
  {
    THIntTensor *count = luaT_getfieldcheckudata(L, 1, "_count", "torch.IntTensor");
    THIntTensor_resize1d(count, gradWeight->size[0]);
    count_data = THIntTensor_data(count);
  }

  if (!THTensor_(isContiguous)(gradWeight))
    luaL_error(L, "gradWeight must be contiguous");
  if (!THLongTensor_isContiguous(input))
    luaL_error(L, "input must be contiguous");
  if (input->nDimension != 1 && input->nDimension != 2)
    luaL_error(L, "input must be a vector or matrix");

  long *input_data = THLongTensor_data(input);
  long numel = THLongTensor_nElement(input);
  long numw = THTensor_(size)(gradWeight, 0);

  // check that inputs are all within range
  for (i=0; i<numel; i++)
    if (input_data[i] < 1 || input_data[i] > numw)
      THError("input out of range");

  gradOutput = THTensor_(newContiguous)(gradOutput);

  real *gw = THTensor_(data)(gradWeight);
  real *go = THTensor_(data)(gradOutput);
  long stride = THTensor_(stride)(gradWeight, 0);

  if (count_data)
    nn_(LookupTable_resetCount)(count_data, input);

#ifdef _OPENMP
  if (numel > 1000)
  {
    // The strategy is to parallelize over sections of the vocabulary, so that
    // thread 1 handles updates to gradWeight[0..nVocab/nThreads]. Every thread
    // has to traverse the entire input, but the dominating factor is the axpy
    // BLAS call.
    #pragma omp parallel private(i)
    {
      int tid = omp_get_thread_num();
      int nthreads = omp_get_num_threads();

      long start = tid * (numw/nthreads + 1);
      long end = start + (numw/nthreads + 1);
      for (i=0; i<numel; i++)
      {
        long k = input_data[i] - 1;
        if (k >= start && k < end)
        {
          real scale = lr;
          if (count_data) scale /= count_data[k];
          THBlas_(axpy)(stride, scale, go + i*stride, 1, gw + k*stride, 1);
        }
      }
    }

    THTensor_(free)(gradOutput);
    return 0;
  }
#endif

  for (i=0; i<numel; i++)
  {
    long k = input_data[i] - 1;
    real scale = lr;
    if (count_data) scale /= count_data[k];
    THBlas_(axpy)(stride, scale, go + i*stride, 1, gw + k*stride, 1);
  }

  THTensor_(free)(gradOutput);
  return 0;
}

static const struct luaL_Reg nn_(LookupTable__) [] = {
  {"LookupTable_accGradParameters", nn_(LookupTable_accGradParameters)},
  {NULL, NULL}
};

static void nn_(LookupTable_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(LookupTable__), "nn");
  lua_pop(L,1);
}

#endif
