#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialClassNLLCriterion.cu"
#else

void THNN_(SpatialClassNLLCriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCIndexTensor *target,
           THCTensor *output,
           bool sizeAverage,
           THCTensor *weights,
           THCTensor *total_weight)
{
  THArgCheck(THCIndexTensor_(nDimension)(state, target) == 3, 1,
             "only batches of spatial targets supported (3D tensors)" \
             " but got targets of dimension: %d",
             THCIndexTensor_(nDimension)(state, target));
  THArgCheck(THCTensor_(nDimension)(state, input) == 4, 2,
             "only batches of spatial inputs supported (4D tensors), "      \
             "but got input of dimension: %d", THCTensor_(nDimension)(state, input));

  long input_batch_size = THCTensor_(size)(state, input, 0);
  THCIndex_t batch_size = THCIndexTensor_(size)(state, target, 0);
  THArgCheck(input_batch_size == batch_size, 2,
             "mismatch between the batch size of input " \
             "(%ld) and that of target (%ld)",
             input_batch_size, batch_size);

  long input_map_height = THCTensor_(size)(state, input, 2);
  THCIndex_t map_height = THCIndexTensor_(size)(state, target, 1);
  THArgCheck(input_map_height == map_height, 2,
             "mismatch between the map height of input " \
             "(%ld) and that of target (%ld)",
             input_map_height, map_height);

  long input_map_width = THCTensor_(size)(state, input, 3);
  THCIndex_t map_width = THCIndexTensor_(size)(state, target, 2);
  THArgCheck(input_map_width == map_width, 2,
             "mismatch between the map width of input " \
             "(%ld) and that of target (%ld)",
             input_map_width, map_width);

  if (weights && THCTensor_(nElement)(state, weights) != THCTensor_(size)(state, input, 1)) {
    THError("weight tensor should be defined either for all or no classes");
  }

  if (weights)
    THCUNN_assertSameGPU(state, 5, input, target, weights, output, total_weight);
  else
    THCUNN_assertSameGPU(state, 4, input, target, output, total_weight);

  input = THCTensor_(newContiguous)(state, input);
  weights = weights ? THCTensor_(newContiguous)(state, weights) : NULL;
  target = THCIndexTensor_(newContiguous)(state, target);

  real *input_data = THCTensor_(data)(state, input);
  real *weights_data = weights ? THCTensor_(data)(state, weights) : NULL;
  THCIndex_t  *target_data = THCIndexTensor_(data)(state, target);
  real *output_data = THCTensor_(data)(state, output);
  real *total_weight_data = THCTensor_(data)(state, total_weight);

  int blocks_per_sample = GET_BLOCKS(map_height * map_width) / 128;
  blocks_per_sample = (blocks_per_sample == 0) ? 1 : blocks_per_sample;
  int total_blocks = blocks_per_sample * batch_size;

  THCTensor_(fill)(state, output, ScalarConvert<int, real>::to(0));
  THCTensor_(fill)(state, total_weight, ScalarConvert<int, real>::to(0));

  cunn_SpatialClassNLLCriterion_updateOutput_kernel<real, accreal>
    <<<total_blocks, CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      output_data,
      total_weight_data,
      input_data,
      target_data,
      weights_data,
      sizeAverage,
      THCTensor_(size)(state, input, 0),
      THCTensor_(size)(state, input, 1),
      THCTensor_(size)(state, input, 2) * THCTensor_(size)(state, input, 3),
      blocks_per_sample
  );
  THCudaCheck(cudaGetLastError());

  if (weights)
    THCTensor_(free)(state, weights);
  THCIndexTensor_(free)(state, target);
  THCTensor_(free)(state, input);
}

void THNN_(SpatialClassNLLCriterion_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCIndexTensor *target,
           THCTensor *gradInput,
           bool sizeAverage,
           THCTensor *weights,
           THCTensor *total_weight)
{
  THArgCheck(THCIndexTensor_(nDimension)(state, target) == 3, 1,
             "only batches of spatial targets supported (3D tensors)");
  THArgCheck(THCTensor_(nDimension)(state, input) == 4, 2,
             "only batches of spatial inputs supported (4D tensors)");
  THArgCheck(THCTensor_(isContiguous)(state, gradInput), 4,
             "gradInput must be contiguous");

  long input_batch_size = THCTensor_(size)(state, input, 0);
  THCIndex_t batch_size = THCIndexTensor_(size)(state, target, 0);
  THArgCheck(input_batch_size == batch_size, 2,
             "mismatch between the batch size of input " \
             "(%ld) and that of target (%ld)",
             input_batch_size, batch_size);

  long input_map_height = THCTensor_(size)(state, input, 2);
  THCIndex_t map_height = THCIndexTensor_(size)(state, target, 1);
  THArgCheck(input_map_height == map_height, 2,
             "mismatch between the map height of input " \
             "(%ld) and that of target (%ld)",
             input_map_height, map_height);

  long input_map_width = THCTensor_(size)(state, input, 3);
  THCIndex_t map_width = THCIndexTensor_(size)(state, target, 2);
  THArgCheck(input_map_width == map_width, 2,
             "mismatch between the map width of input " \
             "(%ld) and that of target (%ld)",
             input_map_width, map_width);

  if (weights && THCTensor_(nElement)(state, weights) != THCTensor_(size)(state, input, 1)) {
    THError("weight tensor should be defined either for all or no classes");
  }

  if (weights)
    THCUNN_assertSameGPU(state, 5, weights, input, target, gradInput, total_weight);
  else
    THCUNN_assertSameGPU(state, 4, input, target, gradInput, total_weight);

  input = THCTensor_(newContiguous)(state, input);
  weights = weights ? THCTensor_(newContiguous)(state, weights) : NULL;
  target = THCIndexTensor_(newContiguous)(state, target);

  real *weights_data = weights ? THCTensor_(data)(state, weights) : NULL;
  real *gradInput_data = THCTensor_(data)(state, gradInput);
  THCIndex_t *target_data = THCIndexTensor_(data)(state, target);
  real *total_weight_data = THCTensor_(data)(state, total_weight);

  int blocks_per_sample = GET_BLOCKS(map_height * map_width) / 128;
  blocks_per_sample = (blocks_per_sample == 0) ? 1 : blocks_per_sample;
  int total_blocks = blocks_per_sample * batch_size;

  cunn_SpatialClassNLLCriterion_updateGradInput_kernel
    <<<total_blocks, CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      gradInput_data,
      target_data,
      weights_data,
      total_weight_data,
      sizeAverage,
      THCTensor_(size)(state, input, 0),
      THCTensor_(size)(state, input, 1),
      THCTensor_(size)(state, input, 2) *THCTensor_(size)(state, input, 3),
      blocks_per_sample
  );
  THCudaCheck(cudaGetLastError());

  if (weights)
    THCTensor_(free)(state, weights);
  THCIndexTensor_(free)(state, target);
  THCTensor_(free)(state, input);
}

#endif
