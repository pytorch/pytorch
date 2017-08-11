#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialClassNLLCriterion.c"
#else

#define INITIAL_CHECK                                                            \
  THArgCheck(THIndexTensor_(nDimension)(target) == 3, 3,                         \
    "only batches of spatial targets supported (3D tensors)"		         \
	     " but got targets of dimension: %d",			         \
	     THIndexTensor_(nDimension)(target));			         \
  THArgCheck(THTensor_(nDimension)(input) == 4, 2,			         \
	     "only batches of spatial inputs supported (4D tensors), "	         \
	     "but got input of dimension: %d", THTensor_(nDimension)(input));    \
  if (weights && THTensor_(nElement)(weights) != THTensor_(size)(input, 1)) {    \
    THError("weight tensor should be defined either for all or no classes");     \
  }                                                                              \
                                                                                 \
  {                                                                              \
    int64_t input0 = THTensor_(size)(input, 0);                                     \
    int64_t input1 = THTensor_(size)(input, 1);                                     \
    int64_t input2 = THTensor_(size)(input, 2);                                     \
    int64_t input3 = THTensor_(size)(input, 3);                                     \
    int64_t target0 = THIndexTensor_(size)(target, 0);                              \
    int64_t target1 = THIndexTensor_(size)(target, 1);                              \
    int64_t target2 = THIndexTensor_(size)(target, 2);                              \
    THAssertMsg(input0 == target0 && input2 == target1 && input3 == target2,     \
              "size mismatch (got input: %ldx%ldx%ldx%ld, target: %ldx%ldx%ld)", \
              input0, input1, input2, input3, target0, target1, target2);        \
  }

void THNN_(SpatialClassNLLCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *output,
          bool sizeAverage,
          THTensor *weights,
          THTensor *total_weight,
          int64_t ignore_index)
{
  INITIAL_CHECK;

  input = THTensor_(newContiguous)(input);
  target = THIndexTensor_(newContiguous)(target);
  weights = weights ? THTensor_(newContiguous)(weights) : NULL;

  real *input_data = THTensor_(data)(input);
  THIndex_t *target_data = THIndexTensor_(data)(target);
  real *weights_data = weights ? THTensor_(data)(weights) : NULL;
  real *output_data = THTensor_(data)(output);
  real *total_weight_data = THTensor_(data)(total_weight);

  int64_t batch_size = THTensor_(size)(input, 0);
  int64_t n_classes = THTensor_(size)(input, 1);
  int64_t map_size = THTensor_(size)(input, 2) * THTensor_(size)(input, 3);
  int64_t sample_size = map_size * n_classes;

  real total_weight_acc = 0;
  real output_acc = 0;
  for (int b = 0; b < batch_size; b++) {
    for (int elem = 0; elem < map_size; elem++) {
      int cur_target = target_data[b * map_size + elem] - TH_INDEX_BASE;
      if (cur_target == ignore_index) continue;
      THAssert(cur_target >= 0 && cur_target < n_classes);

      real cur_weight = weights ? weights_data[cur_target] : 1.0f;
      total_weight_acc += cur_weight;
      output_acc -= input_data[b * sample_size + cur_target * map_size + elem] * cur_weight;
    }
  }
  *total_weight_data = total_weight_acc;
  *output_data = output_acc;

  if (sizeAverage && *total_weight_data)
    *output_data /= *total_weight_data;

  THTensor_(free)(input);
  THIndexTensor_(free)(target);
  if (weights)
    THTensor_(free)(weights);
}

void THNN_(SpatialClassNLLCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *gradInput,
          bool sizeAverage,
          THTensor *weights,
          THTensor *total_weight,
          int64_t ignore_index)
{
  INITIAL_CHECK;
  THArgCheck(THTensor_(isContiguous)(gradInput), 4,
              "gradInput must be contiguous");

  real *total_weight_data = THTensor_(data)(total_weight);
  if (*total_weight_data <= 0)
    return;

  target = THIndexTensor_(newContiguous)(target);
  weights = weights ? THTensor_(newContiguous)(weights) : NULL;

  THIndex_t *target_data = THIndexTensor_(data)(target);
  real *weights_data = weights ? THTensor_(data)(weights) : NULL;
  real *gradInput_data = THTensor_(data)(gradInput);

  int64_t batch_size = THTensor_(size)(input, 0);
  int64_t n_classes = THTensor_(size)(input, 1);
  int64_t map_size = THTensor_(size)(input, 2) * THTensor_(size)(input, 3);
  int64_t sample_size = map_size * n_classes;

  real normalize = sizeAverage ? *total_weight_data : 1.0f;

  int b;
  #pragma omp parallel for
  for (b = 0; b < batch_size; b++) {
    int elem;
    for (elem = 0; elem < map_size; elem++) {
      int cur_target = target_data[b * map_size + elem] - TH_INDEX_BASE;
      if (cur_target == ignore_index) continue;
      THAssert(cur_target >= 0 && cur_target < n_classes);

      gradInput_data[b * sample_size + cur_target * map_size + elem] =
        -(weights ? weights_data[cur_target] : 1.0f) / normalize;
    }
  }

  THIndexTensor_(free)(target);
  if (weights)
    THTensor_(free)(weights);
}

#undef INITIAL_CHECK

#endif
