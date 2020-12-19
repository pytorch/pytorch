#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/MultiMarginCriterion.cu"
#else

static inline void THNN_(MultiMarginCriterion_shapeCheck)(
  THCState *state,
  THCTensor *input, THCTensor *target) {
if (input->dim() <= 1) {
int dim = input->dim() == 0 ? 1 : input->size(0);
int target_size = target->dim() == 0 ? 1 : target->size(0);
TORCH_CHECK(!target->is_empty() && (target->dim() <= 1) && (target_size == dim),
  "inconsistent target size: ", target->sizes(), " for input of size: ", input->sizes());
} else if (input->dim() == 2) {
  int nframe = input->size(0);
  int dim = input->size(1);
  TORCH_CHECK((input->size(1) != 0) && (target->dim() == 2) && (target->size(0) == nframe) && (target->size(1) == dim),
  "inconsistent target size: ", target->sizes(), " for input of size: ", input->sizes());
} else {
  TORCH_CHECK(false, "non-empty vector or matrix expected, got size: ", input->sizes());
}
}

// TODO: improve error messages
void THNN_(MultiMarginCriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCIndexTensor *target,
           THCTensor *output,
           int64_t reduction,
           int p,
           THCTensor *weights,
           accreal margin_)
{
  THNN_(MultiMarginCriterion_shapeCheck)(state, input, target);
  if (input->numel() == 0) {
    return;
  }
  scalar_t margin = ScalarConvert<accreal, scalar_t>::to(margin_);
  THCUNN_assertSameGPU(state, 2, input, target);
  input = THCTensor_(newContiguous)(state, input);
  if(weights)
    weights = THCTensor_(newContiguous)(state, weights);
  if (THTensor_nDimensionLegacyNoScalars(input) == 1)
  {
    int nframe = 1;
    THArgCheck(!target->is_empty() && (THTensor_nDimensionLegacyNoScalars(target) == 1) && (THTensor_sizeLegacyNoScalars(target, 0) == nframe), 3,
               "inconsistent target size");
    dim3 blocks(1);
    dim3 threads(MULTIMARGIN_THREADS);
    if (reduction == at::Reduction::None) {
      THCTensor_(resizeAs)(state, output, target);
    } else {
      THCTensor_(resize0d)(state, output);
    }
    if (p == 1)
    {
      cunn_MultiMarginCriterion_updateOutput_kernel<1, scalar_t, accreal> <<<blocks,threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        THCTensor_(data)(state, output),
        THCTensor_(data)(state, input),
        THCIndexTensor_(data)(state, target),
        weights ? THCTensor_(data)(state, weights) : NULL,
        1, THTensor_sizeLegacyNoScalars(input, 0),
        reduction == at::Reduction::Mean,
        margin
      );
    }
    else if (p == 2)
    {
      cunn_MultiMarginCriterion_updateOutput_kernel<2, scalar_t, accreal> <<<blocks,threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        THCTensor_(data)(state, output),
        THCTensor_(data)(state, input),
        THCIndexTensor_(data)(state, target),
        weights ? THCTensor_(data)(state, weights) : NULL,
        1, THTensor_sizeLegacyNoScalars(input, 0),
        reduction == at::Reduction::Mean,
        margin
      );
    }
    THCudaCheck(cudaGetLastError());
  }
  else if (input->dim() == 2)
  {
    int nframe = input->size(0);
    // allow zero-dim target for 2D input.
    THArgCheck((input->size(1) != 0) && (THTensor_nDimensionLegacyNoScalars(target) == 1) && (THTensor_sizeLegacyNoScalars(target, 0) == nframe), 3,
               "inconsistent target size");
    dim3 blocks(input->size(0));
    dim3 threads(MULTIMARGIN_THREADS);

    if (reduction == at::Reduction::None)
    {
      THCTensor_(resizeAs)(state, output, target);
      if (p == 1)
      {
        cunn_MultiMarginCriterion_updateOutput_kernel<1, scalar_t, accreal> <<<blocks,threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
          THCTensor_(data)(state, output),
          THCTensor_(data)(state, input),
          THCIndexTensor_(data)(state, target),
          weights ? THCTensor_(data)(state, weights) : NULL,
          nframe, input->size(1),
          false,
          margin
        );
      }
      else if (p == 2)
      {
        cunn_MultiMarginCriterion_updateOutput_kernel<2, scalar_t, accreal> <<<blocks,threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
          THCTensor_(data)(state, output),
          THCTensor_(data)(state, input),
          THCIndexTensor_(data)(state, target),
          weights ? THCTensor_(data)(state, weights) : NULL,
          nframe, input->size(1),
          false,
          margin
        );
      }
      THCudaCheck(cudaGetLastError());
    }
    else
    {
      THCTensor_(resize0d)(state, output);
      THCTensor *output_ = THCTensor_(newWithSize1d)(state, input->size(0));  // tmp output buffer
      if (p == 1)
      {
        cunn_MultiMarginCriterion_updateOutput_kernel<1, scalar_t, accreal> <<<blocks,threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
          THCTensor_(data)(state, output_),
          THCTensor_(data)(state, input),
          THCIndexTensor_(data)(state, target),
          weights ? THCTensor_(data)(state, weights) : NULL,
          nframe, input->size(1),
          reduction == at::Reduction::Mean,
          margin
        );
      }
      else if (p == 2)
      {
        cunn_MultiMarginCriterion_updateOutput_kernel<2, scalar_t, accreal> <<<blocks,threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
          THCTensor_(data)(state, output_),
          THCTensor_(data)(state, input),
          THCIndexTensor_(data)(state, target),
          weights ? THCTensor_(data)(state, weights) : NULL,
          input->size(0), input->size(1),
          reduction == at::Reduction::Mean,
          margin
        );
      }
      THCudaCheck(cudaGetLastError());
      auto t = THTensor_wrap(output_);
      auto r = THTensor_wrap(output);
      at::native::sum_out(r, t, at::IntArrayRef(std::vector<int64_t>{}), false, r.scalar_type());
      THCTensor_(free)(state, output_);
    }
  }
  else
  {
    TORCH_CHECK(false, "Expected 2D input with optional zero batch dim, or 1D input with non-zero dims, but got sizes: ",
    input->sizes());
  }

  THCTensor_(free)(state, input);
  if(weights)
    THCTensor_(free)(state, weights);
}

void THNN_(MultiMarginCriterion_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCIndexTensor *target,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int64_t reduction,
           int p,
           THCTensor *weights,
           accreal margin_)
{
  THNN_(MultiMarginCriterion_shapeCheck)(state, input, target);
  input = THCTensor_(newContiguous)(state, input);
  THCTensor_(resizeAs)(state, gradInput, input);
  if (input->numel() == 0) {
    THCTensor_(free)(state, input);
    return;
  }
  scalar_t margin = ScalarConvert<accreal, scalar_t>::to(margin_);
  THCUNN_assertSameGPU(state, 3, input, gradInput, target);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  if(weights)
    weights = THCTensor_(newContiguous)(state, weights);

  if (THTensor_nDimensionLegacyNoScalars(input) == 1)
  {
    dim3 blocks(1);
    dim3 threads(MULTIMARGIN_THREADS);

    if (p == 1)
    {
      cunn_MultiMarginCriterion_updateGradInput_kernel<1, scalar_t, accreal> <<<blocks,threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        THCTensor_(data)(state, gradInput),
        THCTensor_(data)(state, gradOutput),
        THCTensor_(data)(state, input),
        THCIndexTensor_(data)(state, target),
        weights ? THCTensor_(data)(state, weights) : NULL,
        1, THTensor_sizeLegacyNoScalars(gradInput, 0),
        reduction == at::Reduction::Mean,
        margin,
        reduction != at::Reduction::None
      );
    }
    else if (p == 2)
    {
      cunn_MultiMarginCriterion_updateGradInput_kernel<2, scalar_t, accreal> <<<blocks,threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        THCTensor_(data)(state, gradInput),
        THCTensor_(data)(state, gradOutput),
        THCTensor_(data)(state, input),
        THCIndexTensor_(data)(state, target),
        weights ? THCTensor_(data)(state, weights) : NULL,
        1, THTensor_sizeLegacyNoScalars(gradInput, 0),
        reduction == at::Reduction::Mean,
        margin,
        reduction != at::Reduction::None
      );
    }
    THCudaCheck(cudaGetLastError());
  }
  else if (input->dim() == 2)
  {
    int nframe = gradInput->size(0);
    THArgCheck((input->size(1) != 0) && (THTensor_nDimensionLegacyNoScalars(target) == 1) && (THTensor_sizeLegacyNoScalars(target, 0) == nframe), 3,
               "inconsistent target size");
    dim3 blocks(gradInput->size(0));
    dim3 threads(MULTIMARGIN_THREADS);

    if (p == 1)
    {
      cunn_MultiMarginCriterion_updateGradInput_kernel<1, scalar_t, accreal> <<<blocks,threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        THCTensor_(data)(state, gradInput),
        THCTensor_(data)(state, gradOutput),
        THCTensor_(data)(state, input),
        THCIndexTensor_(data)(state, target),
        weights ? THCTensor_(data)(state, weights) : NULL,
        nframe, gradInput->size(1),
        reduction == at::Reduction::Mean,
        margin,
        reduction != at::Reduction::None
      );
    }
    else if (p == 2)
    {
      cunn_MultiMarginCriterion_updateGradInput_kernel<2, scalar_t, accreal> <<<blocks,threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        THCTensor_(data)(state, gradInput),
        THCTensor_(data)(state, gradOutput),
        THCTensor_(data)(state, input),
        THCIndexTensor_(data)(state, target),
        weights ? THCTensor_(data)(state, weights) : NULL,
        nframe, gradInput->size(1),
        reduction == at::Reduction::Mean,
        margin,
        reduction != at::Reduction::None
      );
    }
    THCudaCheck(cudaGetLastError());
  }
  else
  {
    TORCH_CHECK(false, "Expected 2D input with optional zero batch dim, or 1D input with non-zero dims, but got sizes: ", 
    input->sizes());
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
  if(weights)
    THCTensor_(free)(state, weights);
}

#endif
