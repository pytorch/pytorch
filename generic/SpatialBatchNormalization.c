#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialBatchNormalization.c"
#else

static int nn_(SpatialBatchNormalization_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 1, torch_Tensor);
  THTensor *output = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *weight = luaT_toudata(L, 3, torch_Tensor);
  THTensor *bias = luaT_toudata(L, 4, torch_Tensor);
  int train = lua_toboolean(L, 5);
  double eps = lua_tonumber(L, 6);
  double momentum = lua_tonumber(L, 7);
  THTensor *running_mean = luaT_checkudata(L, 8, torch_Tensor);
  THTensor *running_var = luaT_checkudata(L, 9, torch_Tensor);
  THTensor *save_mean = luaT_toudata(L, 10, torch_Tensor);
  THTensor *save_std = luaT_toudata(L, 11, torch_Tensor);

  long nBatch = THTensor_(size)(input, 0);
  long nFeature = THTensor_(size)(input, 1);
  long iH = THTensor_(size)(input, 2);
  long iW = THTensor_(size)(input, 3);
  long n = nBatch * iH * iW;

  #pragma parallel for
  for (long f = 0; f < nFeature; ++f) {
    THTensor *in = THTensor_(newSelect)(input, 1, f);
    THTensor *out = THTensor_(newSelect)(output, 1, f);

    real mean, invstd;

    if (train) {
      // compute mean per feature plane
      accreal sum = 0;
      TH_TENSOR_APPLY(real, in, sum += *in_data;);

      mean = (real) sum / n;
      THTensor_(set1d)(save_mean, f, (real) mean);

      // compute variance per feature plane
      sum = 0;
      TH_TENSOR_APPLY(real, in,
        sum += (*in_data - mean) * (*in_data - mean););

      if (sum == 0 && eps == 0.0) {
        invstd = 0;
      } else {
        invstd = (real) (1 / sqrt(sum/n + eps));
      }
      THTensor_(set1d)(save_std, f, (real) invstd);

      // update running averages
      THTensor_(set1d)(running_mean, f,
        (real) (momentum * mean + (1 - momentum) * THTensor_(get1d)(running_mean, f)));

      accreal unbiased_var = sum / (n - 1);
      THTensor_(set1d)(running_var, f,
        (real) (momentum * unbiased_var + (1 - momentum) * THTensor_(get1d)(running_var, f)));
    } else {
      mean = THTensor_(get1d)(running_mean, f);
      invstd = 1 / sqrt(THTensor_(get1d)(running_var, f) + eps);
    }

    // compute output
    real w = weight ? THTensor_(get1d)(weight, f) : 1;
    real b = bias ? THTensor_(get1d)(bias, f) : 0;

    TH_TENSOR_APPLY2(real, in, real, out,
      *out_data = (real) (((*in_data - mean) * invstd) * w + b););

    THTensor_(free)(out);
    THTensor_(free)(in);
  }

  return 0;
}

static int nn_(SpatialBatchNormalization_backward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 1, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradInput = luaT_toudata(L, 3, torch_Tensor);
  THTensor *gradWeight = luaT_toudata(L, 4, torch_Tensor);
  THTensor *gradBias = luaT_toudata(L, 5, torch_Tensor);
  THTensor *weight = luaT_toudata(L, 6, torch_Tensor);
  THTensor *save_mean = luaT_toudata(L, 7, torch_Tensor);
  THTensor *save_std = luaT_toudata(L, 8, torch_Tensor);
  double scale = lua_tonumber(L, 9);

  long nBatch = THTensor_(size)(input, 0);
  long nFeature = THTensor_(size)(input, 1);
  long iH = THTensor_(size)(input, 2);
  long iW = THTensor_(size)(input, 3);
  long n = nBatch * iH * iW;

  // Q(X) = X - E[x] ; i.e. input centered to zero mean
  // Y = Q(X) / σ    ; i.e. BN output before weight and bias
  // dL/dX = (Q(dL/dY) - dot(Y, dL/dY) * Y) / σ

  #pragma parallel for
  for (long f = 0; f < nFeature; ++f) {
    THTensor *in = THTensor_(newSelect)(input, 1, f);
    THTensor *gradOut = THTensor_(newSelect)(gradOutput, 1, f);
    real mean = THTensor_(get1d)(save_mean, f);
    real invstd = THTensor_(get1d)(save_std, f);
    real w = weight ? THTensor_(get1d)(weight, f) : 1;

    // sum over all gradOutput in feature plane
    accreal sum = 0;
    TH_TENSOR_APPLY(real, gradOut, sum += *gradOut_data;);

    // dot product of the Q(X) and gradOuput
    accreal dotp = 0;
    TH_TENSOR_APPLY2(real, in, real, gradOut,
      dotp += (*in_data - mean) * (*gradOut_data););

    if (gradInput) {
      THTensor *gradIn = THTensor_(newSelect)(gradInput, 1, f);

      // projection of gradOutput on to output scaled by std
      real k = (real) dotp * invstd * invstd / n;
      TH_TENSOR_APPLY2(real, gradIn, real, in,
        *gradIn_data = (*in_data - mean) * k;);

      accreal gradMean = sum / n;
      TH_TENSOR_APPLY2(real, gradIn, real, gradOut,
        *gradIn_data = (*gradOut_data - gradMean - *gradIn_data) * invstd * w;);

      THTensor_(free)(gradIn);
    }

    if (gradWeight) {
      real val = THTensor_(get1d)(gradWeight, f);
      THTensor_(set1d)(gradWeight, f, val + scale * dotp * invstd);
    }

    if (gradBias) {
      real val = THTensor_(get1d)(gradBias, f);
      THTensor_(set1d)(gradBias, f, val + scale * sum);
    }

    THTensor_(free)(gradOut);
    THTensor_(free)(in);
  }

  return 0;
}

static const struct luaL_Reg nn_(SpatialBatchNormalization__) [] = {
  {"SpatialBatchNormalization_updateOutput", nn_(SpatialBatchNormalization_updateOutput)},
  {"SpatialBatchNormalization_backward", nn_(SpatialBatchNormalization_backward)},
  {NULL, NULL}
};

static void nn_(SpatialBatchNormalization_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SpatialBatchNormalization__), "nn");
  lua_pop(L,1);
}

#endif
