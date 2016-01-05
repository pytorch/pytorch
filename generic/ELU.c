#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/ELU.c"
#else

static int nn_(ELU_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  real alpha = luaT_getfieldchecknumber(L, 1, "alpha");
  
  THTensor_(resizeAs)(output, input);
  TH_TENSOR_APPLY2(real, input, real, output, \
                   *output_data = *input_data <= 0 ? (exp(*input_data)-1)*alpha : *input_data; \
                   );
  
    
  return 1;
}

static int nn_(ELU_updateGradInput)(lua_State *L)
{
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  real alpha = luaT_getfieldchecknumber(L, 1, "alpha");
   
  THTensor_(resizeAs)(gradInput, output);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output, \
                   *gradInput_data = (*output_data) <= 0 ? (*gradOutput_data * (*output_data + alpha)) : (*gradOutput_data); \
                   );
  
  return 1;
}

static const struct luaL_Reg nn_(ELU__) [] = {
  { "ELU_updateOutput", nn_(ELU_updateOutput) },
  { "ELU_updateGradInput", nn_(ELU_updateGradInput) },
  { NULL, NULL }
};

static void nn_(ELU_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(ELU__), "nn");
  lua_pop(L, 1);
}

#endif
