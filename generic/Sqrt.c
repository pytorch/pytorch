#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Sqrt.c"
#else

static int nn_(Sqrt_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  real bias = luaT_getfieldchecknumber(L,1,"eps");
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  THTensor_(resizeAs)(output, input);
  THTensor_(sqrt)(output, input);
  return 1;
}

static int nn_(Sqrt_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THTensor_(resizeAs)(gradInput, input);

  if (output->nDimension == 1 || 
      !THTensor_(isContiguous)(output) || 
      !THTensor_(isContiguous)(gradOutput) ||
      !THTensor_(isContiguous)(gradInput))
  {
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output, \
                     *gradInput_data = ((*output_data == 0.0) ? 0.0 : \
                                        (0.5 * (*gradOutput_data / *output_data))););
  }
  else
  {
    real* gradOutput_data = THTensor_(data)(gradOutput);
    real* gradInput_data  = THTensor_(data)(gradInput);
    real* output_data     = THTensor_(data)(output);
    long i;
#pragma omp parallel for private(i)
    for(i = 0; i < THTensor_(nElement)(output); i++)
      if (output_data[i] == 0.0) {
        gradInput_data[i] = 0.0;
      } else {
        gradInput_data[i] = 0.5 * (gradOutput_data[i] / output_data[i]);
      }
  }
  return 1;
}

static const struct luaL_Reg nn_(Sqrt__) [] = {
  {"Sqrt_updateOutput", nn_(Sqrt_updateOutput)},
  {"Sqrt_updateGradInput", nn_(Sqrt_updateGradInput)},
  {NULL, NULL}
};

static void nn_(Sqrt_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(Sqrt__), "nn");
  lua_pop(L,1);
}

#endif
