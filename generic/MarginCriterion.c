#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/MarginCriterion.c"
#else

static int nn_(MarginCriterion_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);  
  THTensor *target = luaT_checkudata(L, 3, torch_Tensor);  
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  real margin = luaT_getfieldchecknumber(L, 1, "margin");
  real sum;

  sum = 0;
  TH_TENSOR_APPLY2(real, input, real, target,
                   real z = (margin - *input_data* *target_data);
                   sum += z>0 ? z : 0;)

  if(sizeAverage)
    sum /= THTensor_(nElement)(input);

  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");

  lua_pushnumber(L, sum);
  return 1;
}

static int nn_(MarginCriterion_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *target = luaT_checkudata(L, 3, torch_Tensor);
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  real margin = luaT_getfieldchecknumber(L, 1, "margin");
  real norm = (sizeAverage ? 1./((real)THTensor_(nElement)(input)) : 1.);

  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
                   *gradInput_data = (*input_data * *target_data) < margin ? -norm* *target_data : 0;)
  return 1;
}

static const struct luaL_Reg nn_(MarginCriterion__) [] = {
  {"MarginCriterion_updateOutput", nn_(MarginCriterion_updateOutput)},
  {"MarginCriterion_updateGradInput", nn_(MarginCriterion_updateGradInput)},
  {NULL, NULL}
};

static void nn_(MarginCriterion_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(MarginCriterion__), "nn");
  lua_pop(L,1);
}

#endif
