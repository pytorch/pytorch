#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/HardTanh.c"
#else

static int nn_(HardTanh_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  real min_val = luaT_getfieldchecknumber(L, 1, "min_val");
  real max_val = luaT_getfieldchecknumber(L, 1, "max_val");
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  THTensor_(resizeAs)(output, input);
  
  if (input->nDimension == 1 || !THTensor_(isContiguous)(input) || !THTensor_(isContiguous)(output))
  {
    TH_TENSOR_APPLY2(real, output, real, input,     \
         if(*input_data < min_val)     \
           *output_data = min_val;   \
         else if(*input_data <= max_val)    \
           *output_data = *input_data;  \
         else       \
           *output_data = max_val;);
  }
  else
  {
    real* ptr_output = THTensor_(data)(output);
    real* ptr_input  = THTensor_(data)(input);
    long i;

#pragma omp parallel for private(i)
    for (i = 0; i < THTensor_(nElement)(input); i++)
    {
      if(ptr_input[i] < min_val)
        ptr_output[i] = min_val;
      else if (ptr_input[i] <= max_val)
        ptr_output[i] = ptr_input[i];
      else
        ptr_output[i] = max_val;
    }
  }
  return 1;
}

static int nn_(HardTanh_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  real min_val = luaT_getfieldchecknumber(L, 1, "min_val");
  real max_val = luaT_getfieldchecknumber(L, 1, "max_val");
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THTensor_(resizeAs)(gradInput, input);

  if (input->nDimension == 1 || 
      !THTensor_(isContiguous)(input) || 
      !THTensor_(isContiguous)(gradOutput) ||
      !THTensor_(isContiguous)(gradInput))
  {
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,  \
         if(*input_data < min_val || *input_data > max_val)    \
           *gradInput_data = 0;                             \
         else           \
           *gradInput_data = *gradOutput_data;);
  }
  else
  {
    real* ptr_gradOutput = THTensor_(data)(gradOutput);
    real* ptr_gradInput  = THTensor_(data)(gradInput);
    real* ptr_input      = THTensor_(data)(input);
    long i;

#pragma omp parallel for private(i)
    for (i = 0; i < THTensor_(nElement)(input); i++)
    {
      if(ptr_input[i] < min_val || ptr_input[i] > max_val)
        ptr_gradInput[i] = 0;
      else
        ptr_gradInput[i] = ptr_gradOutput[i];
    }
  }
  return 1;
}

static const struct luaL_Reg nn_(HardTanh__) [] = {
  {"HardTanh_updateOutput", nn_(HardTanh_updateOutput)},
  {"HardTanh_updateGradInput", nn_(HardTanh_updateGradInput)},
  {NULL, NULL}
};

static void nn_(HardTanh_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(HardTanh__), "nn");
  lua_pop(L,1);
}

#endif
