#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SoftPlus.c"
#else

static int nn_(SoftPlus_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  real beta = luaT_getfieldchecknumber(L, 1, "beta");
  real threshold = luaT_getfieldchecknumber(L, 1, "threshold");

  THTensor_(resizeAs)(output, input);

  /* f(x) = 1/beta * log(1 + exp(beta * x)) */

  TH_TENSOR_APPLY2(real, output, real, input,               \
    *output_data = (*input_data * beta) > threshold ? *input_data : THLog1p(exp(*input_data * beta)) / beta;)
    
    return 1;
}

static int nn_(SoftPlus_updateGradInput)(lua_State *L)
{
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  real beta = luaT_getfieldchecknumber(L, 1, "beta");
  real threshold = luaT_getfieldchecknumber(L, 1, "threshold");

  /* d/dx[log(1+exp(k*x))/k] = exp(kx) / (exp(kx) + 1)
     SINCE
     y = (1/k)*log(1+exp(k*x)) --> x = (1/k)*log(exp(k*y)-1)
     THEREFORE:
     d/dx(f(x)) = (exp(k*y) - 1) / exp(k*y) */

  THTensor_(resizeAs)(gradInput, output);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,    \
                   real z = exp(*output_data * beta);                  \
                   *gradInput_data = (*output_data * beta) > threshold ? *gradOutput_data : *gradOutput_data * (z - 1.)/z;)
    return 1;
}

static const struct luaL_Reg nn_(SoftPlus__) [] = {
  {"SoftPlus_updateOutput", nn_(SoftPlus_updateOutput)},
  {"SoftPlus_updateGradInput", nn_(SoftPlus_updateGradInput)},
  {NULL, NULL}
};

static void nn_(SoftPlus_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SoftPlus__), "nn");
  lua_pop(L,1);
}

#endif
