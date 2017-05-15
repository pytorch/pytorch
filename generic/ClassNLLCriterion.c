#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/ClassNLLCriterion.c"
#else

void THNN_(ClassNLLCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *output,
          bool sizeAverage,
          THTensor *weights,
          THTensor *total_weight)
{
  THNN_CHECK_DIM_SIZE(output, 1, 0, 1);
  THNN_CHECK_DIM_SIZE(total_weight, 1, 0, 1);
  int n_dims = THTensor_(nDimension)(input);
  int n_classes = THTensor_(size)(input, n_dims - 1);

  if (THIndexTensor_(nDimension)(target) > 1) {
    THError("multi-target not supported");
  }
  if (THTensor_(nDimension)(input) > 2) {
    THError("input tensor should be 1D or 2D");
  }
  if (weights && THTensor_(nElement)(weights) != n_classes) {
    THDescBuff s1 = THTensor_(sizeDesc)(weights);
    THError("weight tensor should be defined either for all %d classes or no classes"
	    " but got weight tensor of shape: %s", n_classes, s1.str);
  }

  input = THTensor_(newContiguous)(input);
  target = THIndexTensor_(newContiguous)(target);
  weights = weights ? THTensor_(newContiguous)(weights) : NULL;

  real *input_data = THTensor_(data)(input);
  THIndex_t *target_data = THIndexTensor_(data)(target);
  real *weights_data = weights ? THTensor_(data)(weights) : NULL;
  real *output_data = THTensor_(data)(output);
  real *total_weight_data = THTensor_(data)(total_weight);

  output_data[0] = total_weight_data[0] = 0.0;

  if (THTensor_(nDimension)(input) == 1) {
    int cur_target = target_data[0] - TH_INDEX_BASE;
    THAssert(cur_target >= 0 && cur_target < n_classes);
    total_weight_data[0] = weights ? weights_data[cur_target] : 1.0f;
    output_data[0] = -input_data[cur_target] * total_weight_data[0];
  } else if (THTensor_(nDimension)(input) == 2) {
    int batch_size = THTensor_(size)(input, 0);
    THAssert(THIndexTensor_(size)(target, 0) == batch_size);

    int n_target = THTensor_(size)(input, 1);

    int i;
    for (i = 0; i < batch_size; i++) {
      int cur_target = target_data[i] - TH_INDEX_BASE;
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
  THIndexTensor_(free)(target);
}

void THNN_(ClassNLLCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *gradInput,
          bool sizeAverage,
          THTensor *weights,
          THTensor *total_weight)
{
  int n_dims = THTensor_(nDimension)(input);
  int n_classes = THTensor_(size)(input, n_dims - 1);

  if (!THTensor_(isContiguous)(gradInput)) {
    THError("gradInput must be contiguous");
  }

  real *total_weight_data = THTensor_(data)(total_weight);

  if (!(*total_weight_data > 0)) {
    return;
  }

  if (THIndexTensor_(nDimension)(target) > 1) {
    THError("multi-target not supported");
  }

  if (THTensor_(nDimension)(input) > 2) {
    THError("input tensor should be 1D or 2D");
  }
  
  if (weights && THTensor_(nElement)(weights) != n_classes) {
    THError("weight tensor should be defined either for all or no classes");
  }

  target = THIndexTensor_(newContiguous)(target);
  weights = weights ? THTensor_(newContiguous)(weights) : NULL;

  THIndex_t *target_data = THIndexTensor_(data)(target);
  real *weights_data = weights ? THTensor_(data)(weights) : NULL;
  real *gradInput_data = THTensor_(data)(gradInput);

  if (THTensor_(nDimension)(input) == 1) {
    int cur_target = target_data[0] - TH_INDEX_BASE;
    THAssert(cur_target >= 0 && cur_target < n_classes);

    gradInput_data[cur_target] =
      (!sizeAverage && weights) ? -weights_data[cur_target] : -1;

  } else if (THTensor_(nDimension)(input) == 2) {
    int batch_size = THTensor_(size)(input, 0);
    THAssert(THIndexTensor_(size)(target, 0) == batch_size);

    int n_target = THTensor_(size)(input, 1);

    int i;
    for (i = 0; i < batch_size; i++){
      int cur_target = target_data[i] - TH_INDEX_BASE;

      THAssert(cur_target >= 0 && cur_target < n_classes);

      gradInput_data[i * n_target + cur_target] =
        -(weights ? weights_data[cur_target] : 1.0f);

      if (sizeAverage && *total_weight_data) {
        gradInput_data[i * n_target + cur_target] /= *total_weight_data;
      }
    }
  }

  THIndexTensor_(free)(target);
  if (weights) {
    THTensor_(free)(weights);
  }
}

#endif
