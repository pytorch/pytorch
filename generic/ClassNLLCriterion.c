#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/ClassNLLCriterion.c"
#else


static int nn_(ClassNLLCriterion_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 1, torch_Tensor);
  THLongTensor *target = luaT_checkudata(L, 2, "torch.LongTensor");
  THTensor *weights = NULL;
  if (!lua_isnil(L, 3)) {
    weights = luaT_checkudata(L, 3, torch_Tensor);
  }
  int n_dims = THTensor_(nDimension)(input);
  int n_classes = THTensor_(size)(input, n_dims - 1);

  int sizeAverage = lua_toboolean(L, 4);
  THTensor *output = luaT_checkudata(L, 5, torch_Tensor);
  THTensor *total_weight = luaT_checkudata(L, 6, torch_Tensor);

  if (THLongTensor_nDimension(target) > 1) {
    THError("multi-target not supported");
  }
  if (THTensor_(nDimension)(input) > 2) {
    THError("input tensor should be 1D or 2D");
  }

  input = THTensor_(newContiguous)(input);
  target = THLongTensor_newContiguous(target);
  weights = weights ? THTensor_(newContiguous)(weights) : NULL;

  real *input_data = THTensor_(data)(input);
  long *target_data = THLongTensor_data(target);
  real *weights_data = weights ? THTensor_(data)(weights) : NULL;
  real *output_data = THTensor_(data)(output);
  real *total_weight_data = THTensor_(data)(total_weight);

  output_data[0] = total_weight_data[0] = 0.0;

  if (THTensor_(nDimension)(input) == 1) {
    int cur_target = target_data[0] - 1;
    THAssert(cur_target >= 0 && cur_target < n_classes);
    total_weight_data[0] = weights ? weights_data[cur_target] : 1.0f;
    output_data[0] = -input_data[cur_target] * total_weight_data[0];
  } else if (THTensor_(nDimension)(input) == 2) {
    int batch_size = THTensor_(size)(input, 0);
    int n_target = THTensor_(size)(input, 1);

    int i;
    for (i = 0; i < batch_size; i++) {
      int cur_target = target_data[i] - 1;
      THAssert(cur_target >= 0 && cur_target < n_classes);

      real cur_weight = weights ? weights_data[cur_target] : 1.0f;
      total_weight_data[0] += cur_weight;
      output_data[0] -= input_data[i * n_target + cur_target] * cur_weight;
    }
  }

  if (sizeAverage && total_weight_data[0]) {
    output_data[0] /= total_weight_data[0];
  }

  if (weights) {
    THTensor_(free)(weights);
  }
  THTensor_(free)(input);
  THLongTensor_free(target);

  return 0;
}

static int nn_(ClassNLLCriterion_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 1, torch_Tensor);
  THLongTensor *target = luaT_checkudata(L, 2, "torch.LongTensor");
  THTensor *weights = NULL;
  if (!lua_isnil(L, 3)) {
    weights = luaT_checkudata(L, 3, torch_Tensor);
  }

  int n_dims = THTensor_(nDimension)(input);
  int n_classes = THTensor_(size)(input, n_dims - 1);

  int sizeAverage = lua_toboolean(L, 4);
  THTensor *total_weight = luaT_checkudata(L, 5, torch_Tensor);
  THTensor *gradInput = luaT_checkudata(L, 6, torch_Tensor);
  luaL_argcheck(
    L,
    THTensor_(isContiguous)(gradInput),
    6,
    "gradInput must be contiguous"
  );

  real* total_weight_data = THTensor_(data)(total_weight);

  if (!(*total_weight_data > 0)) {
    return 0;
  }

  if (THLongTensor_nDimension(target) > 1) {
    THError("multi-target not supported");
  }

  if (THTensor_(nDimension)(input) > 2) {
    THError("input tensor should be 1D or 2D");
  }

  target = THLongTensor_newContiguous(target);
  weights = weights ? THTensor_(newContiguous)(weights) : NULL;

  long *target_data = THLongTensor_data(target);
  real *weights_data = weights ? THTensor_(data)(weights) : NULL;
  real *gradInput_data = THTensor_(data)(gradInput);

  if (THTensor_(nDimension)(input) == 1) {
    int cur_target = target_data[0] - 1;
    THAssert(cur_target >= 0 && cur_target < n_classes);

    gradInput_data[cur_target] =
      (!sizeAverage && weights) ? -weights_data[cur_target] : -1;

  } else if (THTensor_(nDimension)(input) == 2) {
    int batch_size = THTensor_(size)(input, 0);
    int n_target = THTensor_(size)(input, 1);

    int i;
    for(i = 0; i < batch_size; i++){
      int cur_target = target_data[i] - 1;

      THAssert(cur_target >= 0 && cur_target < n_classes);

      gradInput_data[i * n_target + cur_target] =
        -(weights ? weights_data[cur_target] : 1.0f);

      if (sizeAverage && *total_weight_data) {
        gradInput_data[i * n_target + cur_target] /= *total_weight_data;
      }
    }
  }

  THLongTensor_free(target);
  if (weights) {
    THTensor_(free)(weights);
  }

  return 0;
}

static const struct luaL_Reg nn_(ClassNLLCriterion__) [] = {
  {"ClassNLLCriterion_updateOutput", nn_(ClassNLLCriterion_updateOutput)},
  {"ClassNLLCriterion_updateGradInput", nn_(ClassNLLCriterion_updateGradInput)},
  {NULL, NULL}
};

static void nn_(ClassNLLCriterion_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(ClassNLLCriterion__), "nn");
  lua_pop(L,1);
}

#endif
