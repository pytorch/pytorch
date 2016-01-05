#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LogSoftMax.c"
#else

static int nn_(LogSoftMax_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  real *input_data, *output_data;
  long nframe = 0, dim = 0;
  long t, d;

  if(input->nDimension == 1)
  {
    nframe = 1;
    dim = input->size[0];
  }
  else if(input->nDimension == 2)
  {
    nframe = input->size[0];
    dim = input->size[1];
  }
  else
    THArgCheck(0, 2, "vector or matrix expected");

  input = THTensor_(newContiguous)(input);
  THTensor_(resizeAs)(output, input);

  real* input_data0 = THTensor_(data)(input);
  real* output_data0 = THTensor_(data)(output);

  accreal logsum;
  real maxInput;
#pragma omp parallel for private(t, d, maxInput, logsum, input_data, \
                                 output_data)
  for(t = 0; t < nframe; t++)
  {
    logsum = 0;
    maxInput = -THInf;
    input_data = input_data0 + dim*t;
    output_data = output_data0 + dim*t;

    for(d = 0; d < dim; d++)
      maxInput = THMax(maxInput, input_data[d]);

    for(d = 0; d < dim; d++)
      logsum += THExpMinusApprox(maxInput-input_data[d]);
    logsum = maxInput + log(logsum);

    for(d = 0; d < dim; d++)
      output_data[d] = input_data[d] - logsum;
  }

  THTensor_(free)(input);

  return 1;
}

static int nn_(LogSoftMax_updateGradInput)(lua_State *L)
{
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  real *gradInput_data, *gradOutput_data, *output_data;
  long nframe = 0, dim = 0;
  long t, d;

  if(output->nDimension == 1)
  {
    nframe = 1;
    dim = output->size[0];
  }
  else if(output->nDimension == 2)
  {
    nframe = output->size[0];
    dim = output->size[1];
  }
  else
    THError("vector or matrix expected");

  THTensor_(resizeAs)(gradInput, output);
  real* gradInput_data0 = THTensor_(data)(gradInput);
  real* output_data0 = THTensor_(data)(output);
  real* gradOutput_data0 = THTensor_(data)(gradOutput);
  accreal sum;
#pragma omp parallel for private(t, sum, d, gradInput_data, output_data, \
                                 gradOutput_data)
  for(t = 0; t < nframe; t++)
  {
    sum = 0;
    gradInput_data = gradInput_data0 + dim*t;
    output_data = output_data0 + dim*t;
    gradOutput_data = gradOutput_data0 + dim*t;

    for(d = 0; d < dim; d++)
      sum += gradOutput_data[d];

    for(d = 0; d < dim; d++)
      gradInput_data[d] = gradOutput_data[d] - exp(output_data[d])*sum;
  }

  return 1;
}

static const struct luaL_Reg nn_(LogSoftMax__) [] = {
  {"LogSoftMax_updateOutput", nn_(LogSoftMax_updateOutput)},
  {"LogSoftMax_updateGradInput", nn_(LogSoftMax_updateGradInput)},
  {NULL, NULL}
};

void nn_(LogSoftMax_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(LogSoftMax__), "nn");
  lua_pop(L,1);
}

#endif
