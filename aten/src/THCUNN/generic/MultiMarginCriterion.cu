#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/MultiMarginCriterion.cu"
#else

// TODO: improve error messages
void THNN_(MultiMarginCriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCIndexTensor *target,
           THCTensor *output,
           bool sizeAverage,
           int p,
           THCTensor *weights,
           accreal margin_,
           bool reduce)
{
  real margin = ScalarConvert<accreal, real>::to(margin_);
  THCUNN_assertSameGPU(state, 2, input, target);
  input = THCTensor_(newContiguous)(state, input);
  if(weights)
    weights = THCTensor_(newContiguous)(state, weights);
  if (input->nDimension == 1)
  {
    dim3 blocks(1);
    dim3 threads(MULTIMARGIN_THREADS);
    THCTensor_(resize1d)(state, output, 1);
    if (p == 1)
    {
      cunn_MultiMarginCriterion_updateOutput_kernel<1, real, accreal> <<<blocks,threads, 0, THCState_getCurrentStream(state)>>>(
        THCTensor_(data)(state, output),
        THCTensor_(data)(state, input),
        THCIndexTensor_(data)(state, target),
        weights ? THCTensor_(data)(state, weights) : NULL,
        1, input->size[0],
        sizeAverage,
        margin
      );
    }
    else if (p == 2)
    {
      cunn_MultiMarginCriterion_updateOutput_kernel<2, real, accreal> <<<blocks,threads, 0, THCState_getCurrentStream(state)>>>(
        THCTensor_(data)(state, output),
        THCTensor_(data)(state, input),
        THCIndexTensor_(data)(state, target),
        weights ? THCTensor_(data)(state, weights) : NULL,
        1, input->size[0],
        sizeAverage,
        margin
      );
    }
    THCudaCheck(cudaGetLastError());
  }
  else if (input->nDimension == 2)
  {
    int nframe = input->size[0];
    THArgCheck((target->nDimension == 1) && (target->size[0] == nframe), 3,
               "inconsistent target size");
    dim3 blocks(input->size[0]);
    dim3 threads(MULTIMARGIN_THREADS);

    if (!reduce)
    {
      THCTensor_(resize1d)(state, output, input->size[0]);
      if (p == 1)
      {
        cunn_MultiMarginCriterion_updateOutput_kernel<1, real, accreal> <<<blocks,threads, 0, THCState_getCurrentStream(state)>>>(
          THCTensor_(data)(state, output),
          THCTensor_(data)(state, input),
          THCIndexTensor_(data)(state, target),
          weights ? THCTensor_(data)(state, weights) : NULL,
          nframe, input->size[1],
          false,
          margin
        );
      }
      else if (p == 2)
      {
        cunn_MultiMarginCriterion_updateOutput_kernel<2, real, accreal> <<<blocks,threads, 0, THCState_getCurrentStream(state)>>>(
          THCTensor_(data)(state, output),
          THCTensor_(data)(state, input),
          THCIndexTensor_(data)(state, target),
          weights ? THCTensor_(data)(state, weights) : NULL,
          nframe, input->size[1],
          false,
          margin
        );
      }
      THCudaCheck(cudaGetLastError());
    }
    else
    {
      THCTensor_(resize1d)(state, output, 1);
      THCTensor *output_ = THCTensor_(newWithSize1d)(state, input->size[0]);  // tmp output buffer
      if (p == 1)
      {
        cunn_MultiMarginCriterion_updateOutput_kernel<1, real, accreal> <<<blocks,threads, 0, THCState_getCurrentStream(state)>>>(
          THCTensor_(data)(state, output_),
          THCTensor_(data)(state, input),
          THCIndexTensor_(data)(state, target),
          weights ? THCTensor_(data)(state, weights) : NULL,
          nframe, input->size[1],
          sizeAverage,
          margin
        );
      }
      else if (p == 2)
      {
        cunn_MultiMarginCriterion_updateOutput_kernel<2, real, accreal> <<<blocks,threads, 0, THCState_getCurrentStream(state)>>>(
          THCTensor_(data)(state, output_),
          THCTensor_(data)(state, input),
          THCIndexTensor_(data)(state, target),
          weights ? THCTensor_(data)(state, weights) : NULL,
          input->size[0], input->size[1],
          sizeAverage,
          margin
        );
      }
      THCudaCheck(cudaGetLastError());
      float sum = THCTensor_(sumall)(state, output_);
      THCTensor_(set1d)(state, output, 0, ScalarConvert<accreal, real>::to(sum));
      THCTensor_(free)(state, output_);
    }
  }
  else
  {
    THError("vector or matrix expected");
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
           bool sizeAverage,
           int p,
           THCTensor *weights,
           accreal margin_,
           bool reduce)
{
  real margin = ScalarConvert<accreal, real>::to(margin_);
  THCUNN_assertSameGPU(state, 3, input, gradInput, target);
  input = THCTensor_(newContiguous)(state, input);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  THCTensor_(resizeAs)(state, gradInput, input);
  if(weights)
    weights = THCTensor_(newContiguous)(state, weights);

  if (input->nDimension == 1)
  {
    dim3 blocks(1);
    dim3 threads(MULTIMARGIN_THREADS);

    if (p == 1)
    {
      cunn_MultiMarginCriterion_updateGradInput_kernel<1, real, accreal> <<<blocks,threads, 0, THCState_getCurrentStream(state)>>>(
        THCTensor_(data)(state, gradInput),
        THCTensor_(data)(state, gradOutput),
        THCTensor_(data)(state, input),
        THCIndexTensor_(data)(state, target),
        weights ? THCTensor_(data)(state, weights) : NULL,
        1, gradInput->size[0],
        sizeAverage,
        margin,
        reduce
      );
    }
    else if (p == 2)
    {
      cunn_MultiMarginCriterion_updateGradInput_kernel<2, real, accreal> <<<blocks,threads, 0, THCState_getCurrentStream(state)>>>(
        THCTensor_(data)(state, gradInput),
        THCTensor_(data)(state, gradOutput),
        THCTensor_(data)(state, input),
        THCIndexTensor_(data)(state, target),
        weights ? THCTensor_(data)(state, weights) : NULL,
        1, gradInput->size[0],
        sizeAverage,
        margin,
        reduce
      );
    }
    THCudaCheck(cudaGetLastError());
  }
  else if (input->nDimension == 2)
  {
    int nframe = gradInput->size[0];
    THArgCheck((target->nDimension == 1) && (target->size[0] == nframe), 3,
               "inconsistent target size");
    dim3 blocks(gradInput->size[0]);
    dim3 threads(MULTIMARGIN_THREADS);

    if (p == 1)
    {
      cunn_MultiMarginCriterion_updateGradInput_kernel<1, real, accreal> <<<blocks,threads, 0, THCState_getCurrentStream(state)>>>(
        THCTensor_(data)(state, gradInput),
        THCTensor_(data)(state, gradOutput),
        THCTensor_(data)(state, input),
        THCIndexTensor_(data)(state, target),
        weights ? THCTensor_(data)(state, weights) : NULL,
        nframe, gradInput->size[1],
        sizeAverage,
        margin,
        reduce
      );
    }
    else if (p == 2)
    {
      cunn_MultiMarginCriterion_updateGradInput_kernel<2, real, accreal> <<<blocks,threads, 0, THCState_getCurrentStream(state)>>>(
        THCTensor_(data)(state, gradInput),
        THCTensor_(data)(state, gradOutput),
        THCTensor_(data)(state, input),
        THCIndexTensor_(data)(state, target),
        weights ? THCTensor_(data)(state, weights) : NULL,
        nframe, gradInput->size[1],
        sizeAverage,
        margin,
        reduce
      );
    }
    THCudaCheck(cudaGetLastError());
  }
  else
  {
    THError("vector or matrix expected");
  }

  THCTensor_(free)(state, input);
  if(weights)
    THCTensor_(free)(state, weights);
}

#endif
