#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/ClassNLLCriterion.cu"
#else

void THNN_(ClassNLLCriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCIndexTensor *target,
           THCTensor *output,
           int64_t reduction,
           THCTensor *weights,
           THCTensor *total_weight,
           int64_t ignore_index) {
  if (THCIndexTensor_(nDimension)(state, target) > 1) {
    THError("multi-target not supported");
  }

  int n_dims = THCTensor_(nDimensionLegacyNoScalars)(state, input);
  int n_classes = THCTensor_(sizeLegacyNoScalars)(state, input, n_dims - 1);

  if (weights) {
    THCUNN_assertSameGPU(
      state, 5, input, target, weights, output, total_weight
    );
  } else {
    THCUNN_assertSameGPU(
      state, 4, input, target, output, total_weight
    );
  }

  if (n_dims != 1 && n_dims != 2) {
    THError("input tensor should be 1D or 2D");
  }

  int64_t batch_size = n_dims == 1 ? 1 : THCTensor_(sizeLegacyNoScalars)(state, input, 0);
  int64_t num_targets = THCudaLongTensor_sizeLegacyNoScalars(state, target, 0);
  THArgCheck(batch_size == num_targets,
      2, "mismatch between the batch size of input (%ld) and that of target (%ld)",
      batch_size, num_targets);

  if (weights && THCTensor_(nElement)(state, weights) != n_classes) {
    THCDescBuff s1 = THCTensor_(sizeDesc)(state, weights);
    THError("weight tensor should be defined either for all %d classes or no classes"
            " but got weight tensor of shape: %s", n_classes, s1.str);
  }

  if (reduction == at::Reduction::None && n_dims == 2) {
    THCTensor_(resize1d)(state, output, batch_size);
    if (batch_size == 0) {
      // This guards from unnecessary operations and launching CUDA kernel with 0 blocks.
      return;
    }
    if (weights) {
      weights = THCTensor_(newContiguous)(state, weights);
    }

    ClassNLLCriterion_updateOutput_no_reduce_kernel<scalar_t>
      <<<GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
        batch_size,
        toDeviceTensor<scalar_t, 2>(state, input),
        toDeviceTensor<THCIndex_t, 1>(state, target),
        toDeviceTensor<scalar_t, 1>(state, output),
        weights ? THCTensor_(data)(state, weights) : NULL,
        n_classes,
        ignore_index);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (weights) {
      THCTensor_(free)(state, weights);
    }
    return;
  }

  THCTensor_(resize0d)(state, output);
  THCTensor_(resize0d)(state, total_weight);

  input = THCTensor_(newContiguous)(state, input);
  weights = weights ? THCTensor_(newContiguous)(state, weights) : NULL;
  target = THCIndexTensor_(newContiguous)(state, target);

  scalar_t *input_data = THCTensor_(data)(state, input);
  scalar_t *weights_data = weights ? THCTensor_(data)(state, weights) : NULL;
  THCIndex_t  *target_data = THCIndexTensor_(data)(state, target);
  scalar_t *output_data = THCTensor_(data)(state, output);
  scalar_t *total_weight_data = THCTensor_(data)(state, total_weight);

  if (THCTensor_(nDimensionLegacyNoScalars)(state, input) == 1) {
    cunn_ClassNLLCriterion_updateOutput_kernel1<scalar_t>
      <<<1, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(
        output_data,
        total_weight_data,
        input_data,
        target_data,
        weights_data,
        reduction == at::Reduction::Mean,
        n_classes,
        ignore_index
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if (THCTensor_(nDimensionLegacyNoScalars)(state, input) == 2) {
    cunn_ClassNLLCriterion_updateOutput_kernel<scalar_t, accreal>
      <<<1, NTHREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
        output_data,
        total_weight_data,
        input_data,
        target_data,
        weights_data,
        reduction == at::Reduction::Mean,
        THCTensor_(size)(state, input, 0),
        THCTensor_(size)(state, input, 1),
        n_classes,
        ignore_index
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

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
           int64_t reduction,
           THCTensor *weights,
           THCTensor *total_weight,
           int64_t ignore_index) {
  if (THCIndexTensor_(nDimensionLegacyNoScalars)(state, target) > 1) {
    THError("multi-target not supported");
  }

  int n_dims = THCTensor_(nDimensionLegacyNoScalars)(state, input);
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

  if (n_dims != 1 && n_dims != 2) {
    THError("input tensor should be 1D or 2D");
  }

  int64_t batch_size = n_dims == 1 ? 1 : THCTensor_(size)(state, input, 0);
  int64_t num_targets = THCudaLongTensor_sizeLegacyNoScalars(state, target, 0);
  THArgCheck(batch_size == num_targets,
      2, "mismatch between the batch size of input (%ld) and that of target (%ld)",
      batch_size, num_targets);

  if (weights && THCTensor_(nElement)(state, weights) != n_classes) {
    THError("weight tensor should be defined either for all or no classes");
  }

  if (reduction == at::Reduction::None && n_dims == 2) {
    THCUNN_check_dim_size(state, gradOutput, 1, 0, batch_size);
    if (batch_size == 0) {
      // This guards from unnecessary operations and launching CUDA kernel with 0 blocks.
      return;
    }
    if (weights) {
      weights = THCTensor_(newContiguous)(state, weights);
    }

    ClassNLLCriterion_updateGradInput_no_reduce_kernel<scalar_t>
      <<<GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
        batch_size,
        toDeviceTensor<THCIndex_t, 1>(state, target),
        toDeviceTensor<scalar_t, 1>(state, gradOutput),
        toDeviceTensor<scalar_t, 2>(state, gradInput),
        weights ? THCTensor_(data)(state, weights) : NULL,
        n_classes,
        ignore_index);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (weights) {
      THCTensor_(free)(state, weights);
    }
    return;
  }

  weights = weights ? THCTensor_(newContiguous)(state, weights) : NULL;
  target = THCIndexTensor_(newContiguous)(state, target);

  THCUNN_check_dim_size(state, gradOutput, 1, 0, 1);
  scalar_t *gradOutput_data = THCTensor_(data)(state, gradOutput);
  scalar_t *weights_data = weights ? THCTensor_(data)(state, weights) : NULL;
  scalar_t *gradInput_data = THCTensor_(data)(state, gradInput);
  THCIndex_t  *target_data = THCIndexTensor_(data)(state, target);
  scalar_t *total_weight_data = THCTensor_(data)(state, total_weight);

  if (THCTensor_(nDimensionLegacyNoScalars)(state, input) == 1) {
    cunn_ClassNLLCriterion_updateGradInput_kernel1<scalar_t>
      <<<1, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(
        gradInput_data,
        gradOutput_data,
        weights_data,
        target_data,
        total_weight_data,
        reduction == at::Reduction::Mean,
        n_classes,
        ignore_index
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    cunn_ClassNLLCriterion_updateGradInput_kernel<scalar_t>
      <<<1, NTHREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
        gradInput_data,
        gradOutput_data,
        target_data,
        weights_data,
        total_weight_data,
        reduction == at::Reduction::Mean,
        THCTensor_(size)(state, input, 0),
        THCTensor_(size)(state, input, 1),
        n_classes,
        ignore_index
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  if (weights) {
    THCTensor_(free)(state, weights);
  }
  THCIndexTensor_(free)(state, target);
}

#endif
