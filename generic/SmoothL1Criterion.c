#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SmoothL1Criterion.c"
#else

static int nn_(SmoothL1Criterion_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *target = luaT_checkudata(L, 3, torch_Tensor);
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  real sum;

  sum = 0;
  TH_TENSOR_APPLY2(real, input, real, target,
                   real z = fabs(*input_data - *target_data);
                   sum += z < 1 ? 0.5*z*z : z - 0.5;)

  if(sizeAverage)
    sum /= THTensor_(nElement)(input);

  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");

  lua_pushnumber(L, sum);
  return 1;
}

static int nn_(SmoothL1Criterion_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *target = luaT_checkudata(L, 3, torch_Tensor);
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  real norm = (sizeAverage ? 1./((real)THTensor_(nElement)(input)) : 1.);

  THTensor_(resizeAs)(gradInput, input);
  TH_TENSOR_APPLY3(real, gradInput, real, input, real, target,
                   real x = *input_data - *target_data;
                   if(x < -1.)
                     *gradInput_data = - norm;
                   else if(x > 1.)
                     *gradInput_data = norm;
                   else
                     *gradInput_data = norm * x;)
  return 1;
}

static const struct luaL_Reg nn_(SmoothL1Criterion__) [] = {
  {"SmoothL1Criterion_updateOutput", nn_(SmoothL1Criterion_updateOutput)},
  {"SmoothL1Criterion_updateGradInput", nn_(SmoothL1Criterion_updateGradInput)},
  {NULL, NULL}
};

static void nn_(SmoothL1Criterion_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SmoothL1Criterion__), "nn");
  lua_pop(L,1);
}

#endif
