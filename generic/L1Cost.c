#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/L1Cost.c"
#else

static int nn_(L1Cost_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);  
  accreal sum;

  sum = 0;
  TH_TENSOR_APPLY(real, input, sum += fabs(*input_data););

  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");

  lua_pushnumber(L, sum);
  return 1;
}

static int nn_(L1Cost_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY2(real, gradInput, real, input,
                   if (*input_data > 0)
                     *gradInput_data = 1;
                   else if (*input_data < 0)
                     *gradInput_data = -1;
                   else
                     *gradInput_data = 0;);
  return 1;
}

static const struct luaL_Reg nn_(L1Cost__) [] = {
  {"L1Cost_updateOutput", nn_(L1Cost_updateOutput)},
  {"L1Cost_updateGradInput", nn_(L1Cost_updateGradInput)},
  {NULL, NULL}
};

static void nn_(L1Cost_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(L1Cost__), "nn");
  lua_pop(L,1);
}

#endif
