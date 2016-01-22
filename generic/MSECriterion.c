#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/MSECriterion.c"
#else

static int nn_(MSECriterion_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);  
  THTensor *target = luaT_checkudata(L, 3, torch_Tensor);  
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  real sum;

  sum = 0;
  TH_TENSOR_APPLY2(real, input, real, target,
                   real z = (*input_data - *target_data);
                   sum += z*z;)

  if(sizeAverage)
    sum /= THTensor_(nElement)(input);

  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");

  lua_pushnumber(L, sum);
  return 1;
}

static int nn_(MSECriterion_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *target = luaT_checkudata(L, 3, torch_Tensor);
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  real norm = (sizeAverage ? 2./((real)THTensor_(nElement)(input)) : 2.);

  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
                   *gradInput_data = norm * (*input_data - *target_data);)
  return 1;
}

static const struct luaL_Reg nn_(MSECriterion__) [] = {
  {"MSECriterion_updateOutput", nn_(MSECriterion_updateOutput)},
  {"MSECriterion_updateGradInput", nn_(MSECriterion_updateGradInput)},
  {NULL, NULL}
};

static void nn_(MSECriterion_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(MSECriterion__), "nn");
  lua_pop(L,1);
}

#endif
