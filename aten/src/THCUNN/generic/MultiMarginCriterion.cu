#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/MultiMarginCriterion.cu"
#else

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
    THArgCheck(!target->is_empty() && (THTensor_nDimensionLegacyNoScalars(target) == 1) && (THTensor_sizeLegacyNoScalars(target, 0) == nframe), 3,
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
      float sum = THCTensor_(sumall)(state, output_);
      THCTensor_(set0d)(state, output, ScalarConvert<accreal, scalar_t>::to(sum));
      THCTensor_(free)(state, output_);
    }
  }
  else
  {
    AT_ERROR("non-empty vector or matrix expected, got sizes: ", input->sizes());
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
  scalar_t margin = ScalarConvert<accreal, scalar_t>::to(margin_);
  THCUNN_assertSameGPU(state, 3, input, gradInput, target);
  input = THCTensor_(newContiguous)(state, input);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  THCTensor_(resizeAs)(state, gradInput, input);
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
    THArgCheck(!target->is_empty() && (THTensor_nDimensionLegacyNoScalars(target) == 1) && (THTensor_sizeLegacyNoScalars(target, 0) == nframe), 3,
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
    AT_ERROR("non-empty vector or matrix expected, got ", input->sizes());
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
  if(weights)
    THCTensor_(free)(state, weights);
}

#endif
