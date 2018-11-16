#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialClassNLLCriterion.c"
#else

#define INITIAL_CHECK                                                            \
  THArgCheck(THIndexTensor_(nDimensionLegacyAll)(target) == 3, 3,                         \
    "only batches of spatial targets supported (3D tensors)"		         \
	     " but got targets of dimension: %d",			         \
	     THIndexTensor_(nDimensionLegacyAll)(target));			         \
  THArgCheck(THTensor_(nDimensionLegacyAll)(input) == 4, 2,			         \
	     "only batches of spatial inputs supported (4D tensors), "	         \
	     "but got input of dimension: %d", THTensor_(nDimensionLegacyAll)(input));    \
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
  THArgCheck(THTensor_(nDimensionLegacyAll)(gradOutput) == 3, 3,                       \
    "gradOutput must have same dimension as target (3)"                       \
	     " but got dimension: %d",			                                        \
	     THTensor_(nDimensionLegacyAll)(gradOutput));			                              \
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
          int64_t reduction,
          THTensor *weights,
          THTensor *total_weight,
          int64_t ignore_index)
{
  INITIAL_CHECK;
  THTensor_(resize1d)(output, 1);
  THTensor_(resize1d)(total_weight, 1);
  ignore_index -= TH_INDEX_BASE;

  if (reduction == Reduction::None) {
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
            THTensor_(fastSet3d)(output, b, h, w, 0.0f);
            continue;
          }
          scalar_t value = THTensor_(fastGet4d)(input, b, cur_target, h, w);
          scalar_t weight = weights ? THTensor_(fastGetLegacy1dNoScalars)(weights, cur_target) : 1.0f;
          THTensor_(fastSet3d)(output, b, h, w, -value * weight);
        }
      }
    }
    return;
  }

  input = THTensor_(newContiguous)(input);
  target = THIndexTensor_(newContiguous)(target);
  weights = weights ? THTensor_(newContiguous)(weights) : NULL;

  scalar_t *input_data = input->data<scalar_t>();
  THIndex_t *target_data = THIndexTensor_(data)(target);
  scalar_t *weights_data = weights ? weights->data<scalar_t>() : NULL;
  scalar_t *output_data = output->data<scalar_t>();
  scalar_t *total_weight_data = total_weight->data<scalar_t>();

  int64_t batch_size = THTensor_(size)(input, 0);
  int64_t n_classes = THTensor_(size)(input, 1);
  int64_t map_size = THTensor_(size)(input, 2) * THTensor_(size)(input, 3);
  int64_t sample_size = map_size * n_classes;

  scalar_t total_weight_acc = 0;
  scalar_t output_acc = 0;
  for (int b = 0; b < batch_size; b++) {
    for (int elem = 0; elem < map_size; elem++) {
      int cur_target = target_data[b * map_size + elem] - TH_INDEX_BASE;
      if (cur_target == ignore_index) continue;
      THAssert(cur_target >= 0 && cur_target < n_classes);

      scalar_t cur_weight = weights ? weights_data[cur_target] : 1.0f;
      total_weight_acc += cur_weight;
      output_acc -= input_data[b * sample_size + cur_target * map_size + elem] * cur_weight;
    }
  }
  *total_weight_data = total_weight_acc;
  *output_data = output_acc;

  if (reduction == Reduction::Mean && *total_weight_data)
    *output_data /= *total_weight_data;

  c10::raw::intrusive_ptr::decref(input);
  THIndexTensor_(free)(target);
  if (weights)
    c10::raw::intrusive_ptr::decref(weights);
}

void THNN_(SpatialClassNLLCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *gradOutput,
          THTensor *gradInput,
          int64_t reduction,
          THTensor *weights,
          THTensor *total_weight,
          int64_t ignore_index)
{
  INITIAL_CHECK;
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);
  THArgCheck(THTensor_(isContiguous)(gradInput), 4,
              "gradInput must be contiguous");
  THNN_CHECK_SHAPE(input, gradInput);
  ignore_index -= TH_INDEX_BASE;

  if (reduction == Reduction::None) {
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
          scalar_t value = -(weights ? THTensor_(fastGetLegacy1dNoScalars)(weights, cur_target) : 1.0f);
          scalar_t gradOutput_value = THTensor_(fastGet3d)(gradOutput, b, h, w);
          THTensor_(fastSet4d)(gradInput, b, cur_target, h, w, value * gradOutput_value);
        }
      }
    }
    return;
  }

  THNN_CHECK_DIM_SIZE(gradOutput, 1, 0, 1);

  scalar_t *total_weight_data = total_weight->data<scalar_t>();
  if (*total_weight_data <= 0)
    return;

  target = THIndexTensor_(newContiguous)(target);
  weights = weights ? THTensor_(newContiguous)(weights) : NULL;

  THIndex_t *target_data = THIndexTensor_(data)(target);
  scalar_t *weights_data = weights ? weights->data<scalar_t>() : NULL;
  scalar_t *gradInput_data = gradInput->data<scalar_t>();

  int64_t batch_size = THTensor_(size)(input, 0);
  int64_t n_classes = THTensor_(size)(input, 1);
  int64_t map_size = THTensor_(size)(input, 2) * THTensor_(size)(input, 3);
  int64_t sample_size = map_size * n_classes;

  scalar_t normalize = (reduction == Reduction::Mean) ? *total_weight_data : 1.0f;

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
        -(weights ? weights_data[cur_target] : 1.0f) / normalize * THTensor_(fastGetLegacy1dNoScalars)(gradOutput, 0);
    }
  }

  THIndexTensor_(free)(target);
  if (weights)
    c10::raw::intrusive_ptr::decref(weights);
}

#undef INITIAL_CHECK

#endif
