#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/ClassNLLCriterion.cu"
#else

void THNN_(ClassNLLCriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCIndexTensor *target,
           THCTensor *output,
           bool sizeAverage,
           THCTensor *weights,
           THCTensor *total_weight,
           int64_t ignore_index,
           bool reduce) {
  if (THCIndexTensor_(nDimension)(state, target) > 1) {
    THError("multi-target not supported");
  }

  int n_dims = THCTensor_(nDimension)(state, input);
  int n_classes = THCTensor_(size)(state, input, n_dims - 1);
  ignore_index -= TH_INDEX_BASE;

  if (weights) {
    THCUNN_assertSameGPU(
      state, 5, input, target, weights, output, total_weight
    );
  } else {
    THCUNN_assertSameGPU(
      state, 4, input, target, output, total_weight
    );
  }

  THArgCheck(n_dims <= 2 && n_dims > 0, 2, "vector or matrix expected");

  int64_t batch_size = n_dims == 1 ? 1 : THCTensor_(size)(state, input, 0);
  int64_t num_targets = THCudaLongTensor_size(state, target, 0);
  THArgCheck(batch_size == num_targets,
      2, "mismatch between the batch size of input (%ld) and that of target (%ld)",
      batch_size, num_targets);

  if (weights && THCTensor_(nElement)(state, weights) != n_classes) {
    THCDescBuff s1 = THCTensor_(sizeDesc)(state, weights);
    THError("weight tensor should be defined either for all %d classes or no classes"
            " but got weight tensor of shape: %s", n_classes, s1.str);
  }

  if (!reduce && n_dims == 2) {
    THCTensor_(resize1d)(state, output, batch_size);
    if (weights) {
      weights = THCTensor_(newContiguous)(state, weights);
    }

    ClassNLLCriterion_updateOutput_no_reduce_kernel<real>
      <<<GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
        batch_size,
        toDeviceTensor<real, 2>(state, input),
        toDeviceTensor<THCIndex_t, 1>(state, target),
        toDeviceTensor<real, 1>(state, output),
        weights ? THCTensor_(data)(state, weights) : NULL,
        n_classes,
        ignore_index);

    THCudaCheck(cudaGetLastError());

    if (weights) {
      THCTensor_(free)(state, weights);
    }
    return;
  }

  if (!reduce && n_dims <= 1) {
    sizeAverage = false;
  }

  THCTensor_(resize1d)(state, output, 1);
  THCTensor_(resize1d)(state, total_weight, 1);

  input = THCTensor_(newContiguous)(state, input);
  weights = weights ? THCTensor_(newContiguous)(state, weights) : NULL;
  target = THCIndexTensor_(newContiguous)(state, target);

  real *input_data = THCTensor_(data)(state, input);
  real *weights_data = weights ? THCTensor_(data)(state, weights) : NULL;
  THCIndex_t  *target_data = THCIndexTensor_(data)(state, target);
  real *output_data = THCTensor_(data)(state, output);
  real *total_weight_data = THCTensor_(data)(state, total_weight);

  if (THCTensor_(nDimension)(state, input) == 1) {
    cunn_ClassNLLCriterion_updateOutput_kernel1<real>
      <<<1, 1, 0, THCState_getCurrentStream(state)>>>(
        output_data,
        total_weight_data,
        input_data,
        target_data,
        weights_data,
        sizeAverage,
        n_classes,
        ignore_index
    );

  } else if (THCTensor_(nDimension)(state, input) == 2) {
    cunn_ClassNLLCriterion_updateOutput_kernel<real, accreal>
      <<<1, NTHREADS, 0, THCState_getCurrentStream(state)>>>(
        output_data,
        total_weight_data,
        input_data,
        target_data,
        weights_data,
        sizeAverage,
        THCTensor_(size)(state, input, 0),
        THCTensor_(size)(state, input, 1),
        n_classes,
        ignore_index
    );
  }
  THCudaCheck(cudaGetLastError());

  if (weights) {
    THCTensor_(free)(state, weights);
  }
  THCIndexTensor_(free)(state, target);
  THCTensor_(free)(state, input);
}

void THNN_(ClassNLLCriterion_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCIndexTensor *target,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           bool sizeAverage,
           THCTensor *weights,
           THCTensor *total_weight,
           int64_t ignore_index,
           bool reduce) {
  if (THCIndexTensor_(nDimension)(state, target) > 1) {
    THError("multi-target not supported");
  }

  int n_dims = THCTensor_(nDimension)(state, input);
  int n_classes = THCTensor_(size)(state, input, n_dims - 1);

  THCTensor_(resizeAs)(state, gradInput, input);
  THCTensor_(zero)(state, gradInput);
  THArgCheck(THCTensor_(isContiguous)(state, gradInput), 4, "gradInput must be contiguous");

  if (weights) {
    THCUNN_assertSameGPU(
      state, 5, weights, input, target, gradInput, total_weight
    );
  }
  else {
    THCUNN_assertSameGPU(
      state, 4, input, target, gradInput, total_weight
    );
  }

  THArgCheck(n_dims <= 2 && n_dims > 0, 2, "vector or matrix expected");

  int64_t batch_size = n_dims == 1 ? 1 : THCTensor_(size)(state, input, 0);
  int64_t num_targets = THCudaLongTensor_size(state, target, 0);
  THArgCheck(batch_size == num_targets,
      2, "mismatch between the batch size of input (%ld) and that of target (%ld)",
      batch_size, num_targets);

  if (weights && THCTensor_(nElement)(state, weights) != n_classes) {
    THError("weight tensor should be defined either for all or no classes");
  }

  if (!reduce && n_dims == 2) {
    THCUNN_check_dim_size(state, gradOutput, 1, 0, batch_size);
    if (weights) {
      weights = THCTensor_(newContiguous)(state, weights);
    }

    ClassNLLCriterion_updateGradInput_no_reduce_kernel<real>
      <<<GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
        batch_size,
        toDeviceTensor<THCIndex_t, 1>(state, target),
        toDeviceTensor<real, 1>(state, gradOutput),
        toDeviceTensor<real, 2>(state, gradInput),
        weights ? THCTensor_(data)(state, weights) : NULL,
        n_classes,
        ignore_index);

    THCudaCheck(cudaGetLastError());

    if (weights) {
      THCTensor_(free)(state, weights);
    }
    return;
  }

  if (!reduce && n_dims <= 1) {
    sizeAverage = false;
  }

  ignore_index -= TH_INDEX_BASE;

  weights = weights ? THCTensor_(newContiguous)(state, weights) : NULL;
  target = THCIndexTensor_(newContiguous)(state, target);

  THCUNN_check_dim_size(state, gradOutput, 1, 0, 1);
  real *gradOutput_data = THCTensor_(data)(state, gradOutput);
  real *weights_data = weights ? THCTensor_(data)(state, weights) : NULL;
  real *gradInput_data = THCTensor_(data)(state, gradInput);
  THCIndex_t  *target_data = THCIndexTensor_(data)(state, target);
  real *total_weight_data = THCTensor_(data)(state, total_weight);

  if (THCTensor_(nDimension)(state, input) == 1) {
    cunn_ClassNLLCriterion_updateGradInput_kernel1<real>
      <<<1, 1, 0, THCState_getCurrentStream(state)>>>(
        gradInput_data,
        gradOutput_data,
        weights_data,
        target_data,
        total_weight_data,
        sizeAverage,
        n_classes,
        ignore_index
    );
  } else {
    cunn_ClassNLLCriterion_updateGradInput_kernel<real>
      <<<1, NTHREADS, 0, THCState_getCurrentStream(state)>>>(
        gradInput_data,
        gradOutput_data,
        target_data,
        weights_data,
        total_weight_data,
        sizeAverage,
        THCTensor_(size)(state, input, 0),
        THCTensor_(size)(state, input, 1),
        n_classes,
        ignore_index
    );
  }
  THCudaCheck(cudaGetLastError());

  if (weights) {
    THCTensor_(free)(state, weights);
  }
  THCIndexTensor_(free)(state, target);
}

#endif
