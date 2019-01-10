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

#define GRADOUTPUT_SHAPE_CHECK                                                \
  THArgCheck(THTensor_(nDimension)(gradOutput) == 3, 3,                       \
    "gradOutput must have same dimension as target (3)"                       \
	     " but got dimension: %d",			                                        \
	     THTensor_(nDimension)(gradOutput));			                              \
  {                                                                           \
    int64_t gradOutput0 = THTensor_(size)(gradOutput, 0);                     \
    int64_t gradOutput1 = THTensor_(size)(gradOutput, 1);                     \
    int64_t gradOutput2 = THTensor_(size)(gradOutput, 2);                     \
    int64_t target0 = THIndexTensor_(size)(target, 0);                        \
    int64_t target1 = THIndexTensor_(size)(target, 1);                        \
    int64_t target2 = THIndexTensor_(size)(target, 2);                        \
    THAssertMsg(                                                              \
        gradOutput0 == target0 && gradOutput1 == target1 && gradOutput2 == target2, \
        "size mismatch (got gradOutput: %ldx%ldx%ld, target: %ldx%ldx%ld)",   \
        gradOutput0, gradOutput1, gradOutput2, target0, target1, target2);    \
  }


void THNN_(SpatialClassNLLCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *output,
          bool sizeAverage,
          THTensor *weights,
          THTensor *total_weight,
          int64_t ignore_index,
          bool reduce)
{
  INITIAL_CHECK;
  THTensor_(resize1d)(output, 1);
  THTensor_(resize1d)(total_weight, 1);
  ignore_index -= TH_INDEX_BASE;

  if (!reduce) {
    int64_t batch_size = THTensor_(size)(input, 0);
    int64_t H = THTensor_(size)(input, 2);
    int64_t W = THTensor_(size)(input, 3);
    THTensor_(resize3d)(output, batch_size, H, W);

    int64_t b, h, w;
    #pragma omp parallel for private(b, h, w)
    for (b = 0; b < batch_size; b++) {
      for (h = 0; h < H; h++) {
        for (w = 0; w < W; w++) {
          int64_t cur_target = (int64_t)THIndexTensor_(get3d)(target, b, h, w) - TH_INDEX_BASE;
          if (cur_target == ignore_index) {
            THTensor_fastSet3d(output, b, h, w, 0.0f);
            continue;
          }
          real value = THTensor_fastGet4d(input, b, cur_target, h, w);
          real weight = weights ? THTensor_fastGet1d(weights, cur_target) : 1.0f;
          THTensor_fastSet3d(output, b, h, w, -value * weight);
        }
      }
    }
    return;
  }

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
          THTensor *gradOutput,
          THTensor *gradInput,
          bool sizeAverage,
          THTensor *weights,
          THTensor *total_weight,
          int64_t ignore_index,
          bool reduce)
{
  INITIAL_CHECK;
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);
  THArgCheck(THTensor_(isContiguous)(gradInput), 4,
              "gradInput must be contiguous");
  THNN_CHECK_SHAPE(input, gradInput);
  ignore_index -= TH_INDEX_BASE;

  if (!reduce) {
    GRADOUTPUT_SHAPE_CHECK;

    int64_t batch_size = THTensor_(size)(input, 0);
    int64_t H = THTensor_(size)(input, 2);
    int64_t W = THTensor_(size)(input, 3);

    int64_t b, h, w;
    #pragma omp parallel for private(b, h, w)
    for (b = 0; b < batch_size; b++) {
      for (h = 0; h < H; h++) {
        for (w = 0; w < W; w++) {
          int64_t cur_target = (int64_t)THIndexTensor_(get3d)(target, b, h, w) - TH_INDEX_BASE;
          if (cur_target == ignore_index) {
            continue;
          }
          real value = -(weights ? THTensor_fastGet1d(weights, cur_target) : 1.0f);
          real gradOutput_value = THTensor_fastGet3d(gradOutput, b, h, w);
          THTensor_fastSet4d(gradInput, b, cur_target, h, w, value * gradOutput_value);
        }
      }
    }
    return;
  }

  THNN_CHECK_DIM_SIZE(gradOutput, 1, 0, 1);

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

  real normalize = (sizeAverage) ? *total_weight_data : 1.0f;

  int b;
  #pragma omp parallel for
  for (b = 0; b < batch_size; b++) {
    int elem;
    for (elem = 0; elem < map_size; elem++) {
      int cur_target = target_data[b * map_size + elem] - TH_INDEX_BASE;
      if (cur_target == ignore_index) continue;
      THAssert(cur_target >= 0 && cur_target < n_classes);

      int index = b * sample_size + cur_target * map_size + elem;
      gradInput_data[index] =
        -(weights ? weights_data[cur_target] : 1.0f) / normalize * THTensor_fastGet1d(gradOutput, 0);
    }
  }

  THIndexTensor_(free)(target);
  if (weights)
    THTensor_(free)(weights);
}

#undef INITIAL_CHECK

#endif
