#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Square.c"
#else

static int nn_(Square_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  THTensor_(resizeAs)(output, input);
  
  if (input->nDimension == 1 || !THTensor_(isContiguous)(input) || !THTensor_(isContiguous)(output))
  {
    TH_TENSOR_APPLY2(real, output, real, input,   \
         *output_data = (*input_data) * (*input_data););
  }
  else
  {
    real* output_data = THTensor_(data)(output);
    real* input_data  = THTensor_(data)(input);
    long i;
#pragma omp parallel for private(i)
    for(i = 0; i < THTensor_(nElement)(input); i++)
      output_data[i] = input_data[i]*input_data[i];
  }
  return 1;
}

static int nn_(Square_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THTensor_(resizeAs)(gradInput, input);

  if (input->nDimension == 1 || 
      !THTensor_(isContiguous)(input) || 
      !THTensor_(isContiguous)(gradOutput) ||
      !THTensor_(isContiguous)(gradInput))
  {
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,  \
         *gradInput_data  = 2.0 * (*gradOutput_data) * (*input_data););
  }
  else
  {
    real* gradOutput_data = THTensor_(data)(gradOutput);
    real* gradInput_data  = THTensor_(data)(gradInput);
    real* input_data  = THTensor_(data)(input);
    long i;
#pragma omp parallel for private(i)
    for(i = 0; i < THTensor_(nElement)(gradInput); i++)
      gradInput_data[i] = 2.0 * gradOutput_data[i] * input_data[i];
  }
  return 1;
}

static const struct luaL_Reg nn_(Square__) [] = {
  {"Square_updateOutput", nn_(Square_updateOutput)},
  {"Square_updateGradInput", nn_(Square_updateGradInput)},
  {NULL, NULL}
};

static void nn_(Square_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(Square__), "nn");
  lua_pop(L,1);
}

#endif
