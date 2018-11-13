#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/MultiLabelMarginCriterion.cu"
#else

// TODO: improve error messages
void THNN_(MultiLabelMarginCriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCIndexTensor *target,
           THCTensor *output,
           THCTensor *istarget,
           int64_t reduction)
{
  input = THCTensor_(newContiguous)(state, input);
  target = THCIndexTensor_(newContiguous)(state, target);
  istarget = THCTensor_(newContiguous)(state, istarget);
  THCTensor_(resizeAs)(state, istarget, input);

  if(input->dim() == 1)
  {
    int dim = input->size(0);
    THArgCheck(!target->is_empty() && (target->dim() == 1) && (target->size(0) == dim), 3,
        "inconsistent target size");
    THCTensor_(resize1d)(state, output, 1);

    dim3 blocks(1);
    dim3 threads(MULTILABELMARGIN_THREADS);

    cunn_MultiLabelMarginCriterion_updateOutput_kernel<scalar_t, accreal>
      <<<blocks, threads, 0, THCState_getCurrentStream(state)>>>(
        THCTensor_(data)(state, output),
        THCTensor_(data)(state, input),
        THCIndexTensor_(data)(state, target),
        THCTensor_(data)(state, istarget),
        1, dim,
        reduction == Reduction::Mean
        );
    THCudaCheck(cudaGetLastError());
  }
  else if(input->dim() == 2)
  {
    int nframe = input->size(0);
    int dim = input->size(1);
    THArgCheck(!target->is_empty() && (target->dim() == 2) && (target->size(0) == nframe)
               && (target->size(1) == dim), 3, "inconsistent target size");

    dim3 blocks(input->size(0));
    dim3 threads(MULTILABELMARGIN_THREADS);

    if (reduction != Reduction::None)
    {
      THCTensor *output_tmp = THCTensor_(newWithSize1d)(state, input->size(0));
      THCTensor_(resize1d)(state, output, 1);

      cunn_MultiLabelMarginCriterion_updateOutput_kernel<scalar_t, accreal>
        <<<blocks, threads, 0, THCState_getCurrentStream(state)>>>(
          THCTensor_(data)(state, output_tmp),
          THCTensor_(data)(state, input),
          THCIndexTensor_(data)(state, target),
          THCTensor_(data)(state, istarget),
          nframe, dim,
          reduction == Reduction::Mean
          );
      THCudaCheck(cudaGetLastError());
      THCTensor_(set1d)(state, output, 0, ScalarConvert<accreal, scalar_t>::to(THCTensor_(sumall)(state, output_tmp)));
      THCTensor_(free)(state, output_tmp);
    }
    else
    {
    THCTensor_(resize1d)(state, output, input->size(0));

    cunn_MultiLabelMarginCriterion_updateOutput_kernel<scalar_t, accreal>
      <<<blocks, threads, 0, THCState_getCurrentStream(state)>>>(
        THCTensor_(data)(state, output),
        THCTensor_(data)(state, input),
        THCIndexTensor_(data)(state, target),
        THCTensor_(data)(state, istarget),
        nframe, dim,
        false
        );
    THCudaCheck(cudaGetLastError());
    }
  }
  else
    AT_ERROR("non-empty vector or matrix expected, got size: ", input->sizes());

  THCTensor_(free)(state, input);
  THCIndexTensor_(free)(state, target);
  THCTensor_(free)(state, istarget);
}

void THNN_(MultiLabelMarginCriterion_updateGradInput)(
            THCState *state,
            THCTensor *input,
            THCIndexTensor *target,
            THCTensor *gradOutput,
            THCTensor *gradInput,
            THCTensor *istarget,
            int64_t reduction)
{
  input = THCTensor_(newContiguous)(state, input);
  target = THCIndexTensor_(newContiguous)(state, target);
  istarget = THCTensor_(newContiguous)(state, istarget);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  THCTensor_(resizeAs)(state, gradInput, input);

  if(gradInput->dim() == 1)
  {
    int dim = gradInput->size(0);
    THArgCheck(!target->is_empty() && (target->dim() == 1) && (target->size(0) == dim), 3,
               "inconsistent target size");
    THArgCheck(!istarget->is_empty() && (istarget->dim() == 1) && (istarget->size(0) == dim), 3,
               "inconsistent isTarget size");
    dim3 blocks(1);
    dim3 threads(MULTILABELMARGIN_THREADS);

    cunn_MultiLabelMarginCriterion_updateGradInput_kernel<scalar_t, accreal>
      <<<blocks, threads, 0, THCState_getCurrentStream(state)>>>(
        THCTensor_(data)(state, gradInput),
        THCTensor_(data)(state, gradOutput),
        THCTensor_(data)(state, input),
        THCIndexTensor_(data)(state, target),
        THCTensor_(data)(state, istarget),
        1, gradInput->size(0),
        reduction == Reduction::Mean,
        reduction != Reduction::None);

  }
  else if(gradInput->dim() == 2)
  {
    int nframe = gradInput->size(0);
    int dim = gradInput->size(1);
    THArgCheck(!target->is_empty() && (target->dim() == 2) && (target->size(0) == nframe)
               && (target->size(1) == dim), 3, "inconsistent target size");
    THArgCheck(!istarget->is_empty() && (istarget->dim() == 2) && (istarget->size(0) == nframe)
               && (istarget->size(1) == dim), 3, "inconsistent isTarget size");
    dim3 blocks(gradInput->size(0));
    dim3 threads(MULTILABELMARGIN_THREADS);

    cunn_MultiLabelMarginCriterion_updateGradInput_kernel<scalar_t, accreal>
      <<<blocks, threads, 0, THCState_getCurrentStream(state)>>>(
        THCTensor_(data)(state, gradInput),
        THCTensor_(data)(state, gradOutput),
        THCTensor_(data)(state, input),
        THCIndexTensor_(data)(state, target),
        THCTensor_(data)(state, istarget),
        gradInput->size(0), gradInput->size(1),
        reduction == Reduction::Mean,
        reduction != Reduction::None);
  }
  else
    AT_ERROR("non-empty vector or matrix expected, got size: ", gradInput->sizes());

  THCudaCheck(cudaGetLastError());

  THCTensor_(free)(state, input);
  THCIndexTensor_(free)(state, target);
  THCTensor_(free)(state, istarget);
  THCTensor_(free)(state, gradOutput);
}

#endif
